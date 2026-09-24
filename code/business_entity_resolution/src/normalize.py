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
"""

import re
import unicodedata


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


def normalize_address(address: str) -> str:
    """
    Normalize a business address.

    Applies base normalization, then expands address abbreviations.

    Args:
        address: Raw business address string.

    Returns:
        Normalized business address.
    """
    text = normalize_text(address)
    text = expand_abbreviations(text, ADDRESS_ABBREVIATIONS)
    return text


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
    Normalize all fields of a single business record and produce derived fields.

    Input row must have: entity_id, business_name, business_address, country.
    Output adds: name_norm, name_core, address_norm, country_norm,
                 postal_codes, house_number, address_numbers, name_tokens,
                 address_tokens.

    Args:
        row: Dict with raw record fields.

    Returns:
        Dict with original fields plus all normalized/derived fields.
    """
    result = dict(row)

    # Normalize core fields
    result["name_norm"] = normalize_name(row.get("business_name", ""))
    result["name_core"] = extract_name_core(result["name_norm"])
    result["address_norm"] = normalize_address(row.get("business_address", ""))
    result["country_norm"] = normalize_text(row.get("country", ""))

    # Extract structured bits
    result["postal_codes"] = extract_postal_codes(result["address_norm"])
    result["house_number"] = extract_house_number(result["address_norm"])
    result["address_numbers"] = set(NUMBER_RE.findall(result["address_norm"]))

    # Token sets for blocking and features
    result["name_tokens"] = get_tokens(result["name_core"])
    result["address_tokens"] = get_tokens(result["address_norm"])

    return result


def normalize_dataframe(df) -> list:
    """
    Normalize all records in a DataFrame.

    Args:
        df: pandas DataFrame with entity_id, business_name, business_address, country.

    Returns:
        List of normalized record dicts.
    """
    return [normalize_record(row) for row in df.to_dict("records")]
