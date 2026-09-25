"""
Text normalization for business names and addresses.

Handles the noise patterns described in CLAUDE.md §1:
- Abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited, etc.)
- Legal suffix removal for name_core
- Address abbreviations (Rd/Road, St/Street, French: bd, av, r→rue, etc.)
- Unicode NFKD + accent stripping (important for French)
- & → and, punctuation stripping, whitespace collapsing
- Postal code extraction (US ZIP, India PIN, France CP)

All normalization is country-agnostic to handle unseen countries at test time.

Scale: normalize_frame() works on whole columns in parallel worker processes
and returns compact string columns (no per-record Python dicts), so ~12M
records per split fit in memory.
"""

import multiprocessing as mp
import re
import unicodedata

import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Name abbreviation expansions (generic, multilingual)
# ──────────────────────────────────────────────────────────────────────

NAME_ABBREVIATIONS = {
    # English
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "co": "company",
    "intl": "international",
    "natl": "national",
    "mfg": "manufacturing",
    "svcs": "services",
    "svc": "service",
    "grp": "group",
    "assoc": "associates",
    "assn": "association",
    "dept": "department",
    "div": "division",
    "engg": "engineering",
    "engr": "engineering",
    "tech": "technology",
    "techs": "technologies",
    "mgmt": "management",
    "enterp": "enterprises",
    "ent": "enterprises",
    "infra": "infrastructure",
    "soln": "solutions",
    "solns": "solutions",
    "sys": "systems",
    "ind": "industries",
    "inds": "industries",
    "pharm": "pharmaceuticals",
    "pharma": "pharmaceuticals",
    "fin": "financial",
    "govt": "government",
    "inst": "institute",
    "univ": "university",
    "hosp": "hospital",
    "labs": "laboratories",
    "lab": "laboratory",
    "bros": "brothers",
    "bro": "brother",
    # French
    "ste": "societe",
    "ets": "etablissements",
    "cie": "compagnie",
}

# ──────────────────────────────────────────────────────────────────────
# Legal suffixes to strip for name_core (order matters: longest first)
# ──────────────────────────────────────────────────────────────────────

LEGAL_SUFFIXES = [
    # English (multi-word first)
    "private limited",
    "pvt ltd",
    "pvt limited",
    "private ltd",
    "limited liability partnership",
    "limited liability company",
    "limited liability",
    "public limited company",
    "public limited",
    "limited partnership",
    "general partnership",
    "incorporated",
    "corporation",
    "limited",
    "company",
    "llp",
    "llc",
    "lp",
    "plc",
    "inc",
    "corp",
    "ltd",
    "co",
    # French
    "societe a responsabilite limitee",
    "societe par actions simplifiee",
    "societe anonyme",
    "entreprise unipersonnelle a responsabilite limitee",
    "sarl",
    "sas",
    "sasu",
    "sa",
    "eurl",
    "sci",
    "snc",
    "sca",
    "scs",
    # German
    "gesellschaft mit beschrankter haftung",
    "gmbh",
    "ag",
    "ohg",
    "kg",
    # Indian
    "opc",  # One Person Company
    "nidhi",
]

# ──────────────────────────────────────────────────────────────────────
# Address abbreviation expansions
# ──────────────────────────────────────────────────────────────────────

ADDRESS_ABBREVIATIONS = {
    # English
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "pkwy": "parkway",
    "hwy": "highway",
    "cir": "circle",
    "trl": "trail",
    "ter": "terrace",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "rm": "room",
    "po": "post office",
    "rte": "route",
    # Directions
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    # Indian
    "nagar": "nagar",
    "marg": "marg",
    "dist": "district",
    "opp": "opposite",
    "nr": "near",
    "extn": "extension",
    "ext": "extension",
    # French
    "bd": "boulevard",
    "r": "rue",
    "ch": "chemin",
    "imp": "impasse",
    "all": "allee",
    "rte": "route",
    "pl": "place",
    "crs": "cours",
    "fbg": "faubourg",
    "pte": "porte",
    "pass": "passage",
    "rsd": "residence",
    "bat": "batiment",
    "zac": "zone d amenagement concerte",
    "zi": "zone industrielle",
    "za": "zone artisanale",
    "bp": "boite postale",
    "cs": "cedex",
}

# ──────────────────────────────────────────────────────────────────────
# Regex patterns
# ──────────────────────────────────────────────────────────────────────

# Postal code patterns (generic, not country-specific)
# Matches: US 5-digit ZIP, India 6-digit PIN, France 5-digit CP, etc.
POSTAL_CODE_RE = re.compile(r"\b(\d{5,6})\b")

# House/street number at start of address
HOUSE_NUMBER_RE = re.compile(r"^(\d+[\w-]*)\b")

# Any number in an address (house, street, unit, postal code, ...)
NUMBER_RE = re.compile(r"\d+")

# Punctuation to strip (keep hyphens in IDs)
PUNCTUATION_RE = re.compile(r"[^\w\s-]")

# Multiple spaces
MULTI_SPACE_RE = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """
    Remove diacritical marks from text using Unicode NFKD decomposition.

    Important for French text normalization (e.g., résumé → resume,
    Société → societe).

    Args:
        text: Input string.

    Returns:
        String with accents removed.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def normalize_text(text: str) -> str:
    """
    Apply base normalization to any text field.

    Steps: lowercase → NFKD + strip accents → & → and → strip punctuation
    → collapse whitespace → strip.

    Args:
        text: Raw text string.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    text = text.lower()
    text = strip_accents(text)
    text = text.replace("&", " and ")
    text = PUNCTUATION_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def expand_abbreviations(text: str, abbrev_dict: dict) -> str:
    """
    Expand known abbreviations in text using a dictionary.

    Operates on whole words only (token-level replacement).

    Args:
        text: Normalized (lowercased, cleaned) text.
        abbrev_dict: Mapping from abbreviation → full form.

    Returns:
        Text with abbreviations expanded.
    """
    tokens = text.split()
    expanded = []
    for token in tokens:
        expanded.append(abbrev_dict.get(token, token))
    return " ".join(expanded)


def normalize_name(name: str) -> str:
    """
    Normalize a business name.

    Applies base normalization, then expands name abbreviations.

    Args:
        name: Raw business name string.

    Returns:
        Normalized business name.
    """
    text = normalize_text(name)
    text = expand_abbreviations(text, NAME_ABBREVIATIONS)
    return text


def extract_name_core(normalized_name: str) -> str:
    """
    Remove legal suffixes from a normalized business name to get the core name.

    This helps compare the actual business identity without legal form noise
    (e.g., "acme corporation" and "acme corp" both become "acme").

    Legal suffixes are matched greedily from longest to shortest and stripped
    repeatedly, so stacked forms like "abc company private limited" → "abc".
    A suffix is only removed when a word precedes it, so the core is never
    empty (a name that is only a legal form, e.g. "sas", is kept as is).

    Args:
        normalized_name: Already-normalized business name.

    Returns:
        Name with legal suffixes stripped and trimmed.
    """
    name = normalized_name.strip()
    stripped = True
    while stripped:
        stripped = False
        for suffix in LEGAL_SUFFIXES:
            if name.endswith(" " + suffix):
                name = name[: -(len(suffix) + 1)].strip()
                stripped = True
                break
    return name


# Placeholder words that appear literally in real addresses ("New Delhi, null, A-68")
NULL_TOKENS = {"null", "none", "nan"}


def normalize_address(address: str) -> str:
    """
    Normalize a business address.

    Applies base normalization, expands address abbreviations and drops
    placeholder tokens such as a literal "null".

    Args:
        address: Raw business address string.

    Returns:
        Normalized business address.
    """
    text = normalize_text(address)
    text = expand_abbreviations(text, ADDRESS_ABBREVIATIONS)
    return " ".join(t for t in text.split() if t not in NULL_TOKENS)


def extract_postal_codes(text: str) -> list:
    """
    Extract all potential postal codes (5-6 digit numbers) from text.

    Works for US ZIP (5 digits), India PIN (6 digits), and France CP (5 digits)
    without requiring country knowledge.

    Args:
        text: Normalized text (name or address).

    Returns:
        List of postal code strings found.
    """
    return POSTAL_CODE_RE.findall(text)


def extract_house_number(address: str) -> str:
    """
    Extract the leading house/street number from an address.

    Args:
        address: Normalized address text.

    Returns:
        House number string, or empty string if not found.
    """
    match = HOUSE_NUMBER_RE.match(address)
    return match.group(1) if match else ""


def get_tokens(text: str) -> set:
    """
    Split text into a set of unique tokens.

    Args:
        text: Normalized text.

    Returns:
        Set of token strings.
    """
    if not text:
        return set()
    return set(text.split())


def normalize_record(row: dict) -> dict:
    """
    Normalize a single record (for notebooks / debugging one pair).

    The pipeline itself uses normalize_frame, which produces the same fields
    as columns for millions of records.

    Args:
        row: Dict with business_name, business_address, country.

    Returns:
        Dict with the original fields plus every NORMALIZED_COLUMNS field.
    """
    out = _normalize_chunk(([row.get("business_name", "")],
                            [row.get("business_address", "")],
                            [row.get("country", "")]))
    return {**row, **{col: values[0] for col, values in out.items()}}


# Columns added by normalize_frame. Multi-valued fields (postal codes, all
# numbers) are stored as space-separated strings to stay compact.
NORMALIZED_COLUMNS = [
    "name_norm",        # lowercased, accents stripped, abbreviations expanded
    "name_core",        # name_norm without legal suffixes
    "address_norm",     # normalized address
    "country_norm",     # normalized country string (open set)
    "postal_codes",     # 5-6 digit numbers in the address, space-separated
    "house_number",     # leading number of the address ("" if none)
    "address_numbers",  # every distinct number in the address, space-separated
]


def _normalize_chunk(chunk: tuple) -> dict:
    """
    Normalize one chunk of records (runs inside a worker process).

    Args:
        chunk: Tuple of (names, addresses, countries) lists.

    Returns:
        Dict column name → list of values, one per record.
    """
    names, addresses, countries = chunk
    out = {col: [] for col in NORMALIZED_COLUMNS}
    for name, address, country in zip(names, addresses, countries):
        name_norm = normalize_name(name or "")
        address_norm = normalize_address(address or "")
        out["name_norm"].append(name_norm)
        out["name_core"].append(extract_name_core(name_norm))
        out["address_norm"].append(address_norm)
        out["country_norm"].append(normalize_text(country or ""))
        out["postal_codes"].append(" ".join(extract_postal_codes(address_norm)))
        out["house_number"].append(extract_house_number(address_norm))
        out["address_numbers"].append(" ".join(sorted(set(NUMBER_RE.findall(address_norm)))))
    return out


def normalize_frame(df: pd.DataFrame, n_jobs: int = 1,
                    chunk_size: int = 250_000) -> pd.DataFrame:
    """
    Normalize every record of a source DataFrame, in parallel.

    Args:
        df: DataFrame with business_name, business_address, country.
        n_jobs: Worker processes (1 = run in this process).
        chunk_size: Records per worker task.

    Returns:
        Copy of df with the NORMALIZED_COLUMNS added as compact string columns.
    """
    names = df["business_name"].tolist()
    addresses = df["business_address"].tolist()
    countries = df["country"].tolist()
    chunks = [
        (names[i:i + chunk_size], addresses[i:i + chunk_size], countries[i:i + chunk_size])
        for i in range(0, len(names), chunk_size)
    ]

    if n_jobs > 1 and len(chunks) > 1:
        with mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn") \
                .Pool(min(n_jobs, len(chunks))) as pool:
            results = pool.map(_normalize_chunk, chunks)
    else:
        results = [_normalize_chunk(c) for c in chunks]

    string_dtype = df["business_name"].dtype
    out = df.copy()
    for col in NORMALIZED_COLUMNS:
        values = [v for part in results for v in part[col]]
        out[col] = pd.array(values, dtype=string_dtype)
    return out
