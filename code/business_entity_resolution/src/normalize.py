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
import time
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc


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


# ──────────────────────────────────────────────────────────────────────
# Indic scripts → Latin letters
# ──────────────────────────────────────────────────────────────────────
# S2/S3 write many Indian names and addresses in Devanagari, Bengali,
# Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada or Malayalam
# ("मॉडर्न सॉल्यूशंस प्राइवेट लिमिटेड" = Modern Solutions Private Limited).
# These nine Unicode blocks share one layout (each is 0x80 wide, same
# offset = same sound), so a single offset table transliterates all of them.
# Without this, strip_accents deletes their vowel signs and nothing matches.

INDIC_FIRST, INDIC_LAST = 0x0900, 0x0D7F
INDIC_PATTERN = "[\u0900-\u0D7F]"
INDIC_RE = re.compile(INDIC_PATTERN)
_NUKTA, _VIRAMA = 0x3C, 0x4D

# offset within a block → (kind, latin). Kinds: "cons" (consonant with an
# inherent "a"), "vowel" (independent vowel), "sign" (vowel sign that
# replaces the inherent "a"), "mark" (nasal/aspiration), "final" (letter
# with no inherent vowel), "space", "skip".
_INDIC_OFFSETS = {
    0x00: ("mark", "n"), 0x01: ("mark", "n"), 0x02: ("mark", "n"), 0x03: ("mark", "h"),
    0x04: ("vowel", "a"), 0x05: ("vowel", "a"), 0x06: ("vowel", "a"), 0x07: ("vowel", "i"),
    0x08: ("vowel", "i"), 0x09: ("vowel", "u"), 0x0A: ("vowel", "u"), 0x0B: ("vowel", "ri"),
    0x0C: ("vowel", "li"), 0x0D: ("vowel", "e"), 0x0E: ("vowel", "e"), 0x0F: ("vowel", "e"),
    0x10: ("vowel", "ai"), 0x11: ("vowel", "o"), 0x12: ("vowel", "o"), 0x13: ("vowel", "o"),
    0x14: ("vowel", "au"),
    0x15: ("cons", "k"), 0x16: ("cons", "kh"), 0x17: ("cons", "g"), 0x18: ("cons", "gh"),
    0x19: ("cons", "n"), 0x1A: ("cons", "ch"), 0x1B: ("cons", "chh"), 0x1C: ("cons", "j"),
    0x1D: ("cons", "jh"), 0x1E: ("cons", "n"), 0x1F: ("cons", "t"), 0x20: ("cons", "th"),
    0x21: ("cons", "d"), 0x22: ("cons", "dh"), 0x23: ("cons", "n"), 0x24: ("cons", "t"),
    0x25: ("cons", "th"), 0x26: ("cons", "d"), 0x27: ("cons", "dh"), 0x28: ("cons", "n"),
    0x29: ("cons", "n"), 0x2A: ("cons", "p"), 0x2B: ("cons", "ph"), 0x2C: ("cons", "b"),
    0x2D: ("cons", "bh"), 0x2E: ("cons", "m"), 0x2F: ("cons", "y"), 0x30: ("cons", "r"),
    0x31: ("cons", "r"), 0x32: ("cons", "l"), 0x33: ("cons", "l"), 0x34: ("cons", "l"),
    0x35: ("cons", "v"), 0x36: ("cons", "sh"), 0x37: ("cons", "sh"), 0x38: ("cons", "s"),
    0x39: ("cons", "h"),
    0x3E: ("sign", "a"), 0x3F: ("sign", "i"), 0x40: ("sign", "i"), 0x41: ("sign", "u"),
    0x42: ("sign", "u"), 0x43: ("sign", "ri"), 0x44: ("sign", "ri"), 0x45: ("sign", "e"),
    0x46: ("sign", "e"), 0x47: ("sign", "e"), 0x48: ("sign", "ai"), 0x49: ("sign", "o"),
    0x4A: ("sign", "o"), 0x4B: ("sign", "o"), 0x4C: ("sign", "au"),
    0x50: ("final", "om"),
    0x58: ("cons", "q"), 0x59: ("cons", "kh"), 0x5A: ("cons", "g"), 0x5B: ("cons", "z"),
    0x5C: ("cons", "r"), 0x5D: ("cons", "rh"), 0x5E: ("cons", "f"), 0x5F: ("cons", "y"),
    0x60: ("vowel", "ri"), 0x61: ("vowel", "li"), 0x62: ("sign", "li"), 0x63: ("sign", "li"),
    0x64: ("space", " "), 0x65: ("space", " "),
    **{0x66 + d: ("final", str(d)) for d in range(10)},
    0x71: ("cons", "v"),
}

# Letters that differ between blocks, by exact code point
_INDIC_SPECIAL = {
    0x09CE: ("final", "t"),                          # Bengali khanda ta
    0x09F0: ("cons", "r"), 0x09F1: ("cons", "v"),    # Assamese ra / wa
    0x0A70: ("mark", "n"), 0x0A71: ("skip", ""),     # Gurmukhi tippi / addak
    0x0D4E: ("final", "r"),                          # Malayalam dot reph
    0x0D7A: ("final", "n"), 0x0D7B: ("final", "n"), 0x0D7C: ("final", "r"),  # Malayalam
    0x0D7D: ("final", "l"), 0x0D7E: ("final", "l"), 0x0D7F: ("final", "k"),  # chillu letters
}

# Consonant + nukta (dot below) → borrowed sound, e.g. ज़ = z, फ़ = f
_NUKTA_SOUNDS = {"j": "z", "ph": "f", "k": "q", "d": "r", "dh": "rh"}

# Malayalam writes "nt" and "tt" with the letter ṟa (ന്റ, റ്റ); rewrite them
# with plain ta before transliterating ("ഇന്റർനാഷണൽ" → "intarnashanal")
_INDIC_CLUSTERS = {"\u0d28\u0d4d\u0d31": "\u0d28\u0d4d\u0d1f",
                   "\u0d31\u0d4d\u0d31": "\u0d1f\u0d4d\u0d1f"}


def _indic_kind(char: str):
    """
    Classify one character of an Indic script.

    Args:
        char: A single character.

    Returns:
        (kind, latin) tuple, ("nukta", "") / ("virama", ""), or None when the
        character is not in an Indic block.
    """
    cp = ord(char)
    if not INDIC_FIRST <= cp <= INDIC_LAST:
        return None
    if cp in _INDIC_SPECIAL:
        return _INDIC_SPECIAL[cp]
    offset = (cp - INDIC_FIRST) & 0x7F
    if offset == _NUKTA:
        return ("nukta", "")
    if offset == _VIRAMA:
        return ("virama", "")
    return _INDIC_OFFSETS.get(offset, ("skip", ""))


def transliterate_indic(text: str) -> str:
    """
    Transliterate Indic-script characters to plain Latin letters.

    A consonant carries an inherent "a" unless a vowel sign replaces it or a
    virama cancels it; the inherent "a" of a word-final consonant is dropped
    (Hindi-style), so "मॉडर्न" → "modarn" and "लिमिटेड" → "limited".
    Characters outside the Indic blocks pass through unchanged.

    Args:
        text: Raw text, possibly mixing scripts.

    Returns:
        Text with every Indic character replaced by Latin letters.
    """
    for cluster, plain in _INDIC_CLUSTERS.items():
        text = text.replace(cluster, plain)
    out = []
    i, n = 0, len(text)
    while i < n:
        info = _indic_kind(text[i])
        if info is None:
            out.append(text[i])
            i += 1
            continue
        kind, latin = info
        i += 1
        if kind != "cons":
            out.append(latin)
            continue
        if i < n and (_indic_kind(text[i]) or ("",))[0] == "nukta":
            latin = _NUKTA_SOUNDS.get(latin, latin)
            i += 1
        following = _indic_kind(text[i]) if i < n else None
        if following and following[0] == "sign":
            out.append(latin + following[1])
            i += 1
        elif following and following[0] == "virama":
            out.append(latin)
            i += 1
        elif following and following[0] in ("cons", "vowel", "mark", "final"):
            out.append(latin + "a")
        else:
            out.append(latin)  # word end: drop the inherent "a"
    return "".join(out)


# Zero-width joiners split Malayalam/Devanagari words when they become spaces
ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")


def normalize_text(text: str) -> str:
    """
    Apply base normalization to any text field.

    Steps: Indic scripts → Latin → lowercase → NFKD + strip accents
    → & → and → strip punctuation → collapse whitespace → strip.

    Args:
        text: Raw text string.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    text = ZERO_WIDTH_RE.sub("", text)
    if INDIC_RE.search(text):
        text = transliterate_indic(text)
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
    name = WEB_JUNK_RE.sub(" ", name)
    name = ID_CODE_RE.sub(" ", name)
    text = normalize_text(name)
    text = " ".join(_fix_zero_for_o(t) for t in text.split())
    text = expand_abbreviations(text, NAME_ABBREVIATIONS)
    return text


# Website / handle wrappers around a name: "reliablethayer.com", "@nationalheritage",
# "| www.bangalore..." (the glued words are split later by snap_to_reference)
WEB_JUNK_RE = re.compile(r"(?i)https?://|www\.|@|\.(?:co\.in|com|net|org|in)\b")
WEB_PATTERN = r"(?i)https?://|www\.|@|\.(?:com|net|org|in)\b|[a-z]com\b"

# Record codes appended to names: "Big-Pvt Ltd Center #93347"
ID_CODE_RE = re.compile(r"#\s*\d+")


def _fix_zero_for_o(token: str) -> str:
    """
    Turn the digit 0 back into the letter o inside a word ("0rthopedic").

    Only for tokens made of letters plus zeros, with at least two letters, so
    numbers like "10th" or "2000" are left alone.

    Args:
        token: One normalized token.

    Returns:
        The token with 0 → o when it is a misspelled word, else unchanged.
    """
    if "0" in token and token.isalnum() and sum(c.isalpha() for c in token) >= 2 \
            and all(c.isalpha() or c == "0" for c in token):
        return token.replace("0", "o")
    return token


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
    # Legal words also show up mid-name ("creative limited private software",
    # "private first agro limited"); drop them anywhere as long as a word remains
    core = [t for t in name.split() if t not in LEGAL_WORDS_ANYWHERE]
    return " ".join(core) if core else name


# Unambiguous legal-form words (after abbreviation expansion) removed from
# name_core wherever they appear, not only at the end
LEGAL_WORDS_ANYWHERE = {"private", "limited", "incorporated", "corporation",
                        "llp", "llc", "plc", "opc"}


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
    text = normalize_text(HALF_RE.sub(" ", address))
    text = expand_abbreviations(text, ADDRESS_ABBREVIATIONS)
    text = expand_abbreviations(text, ORDINAL_WORDS)
    return " ".join(t for t in text.split() if t not in NULL_TOKENS)


# "516 1/2 201st Avenue": the half adds a spurious "1" and "2" to the numbers
HALF_RE = re.compile(r"(?<![\d/])1/2(?![\d/])")

# "309 TWELFTH ST" vs "309 12th Street"
ORDINAL_WORDS = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    "eleventh": "11th", "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th",
    "fifteenth": "15th", "sixteenth": "16th", "seventeenth": "17th",
    "eighteenth": "18th", "nineteenth": "19th", "twentieth": "20th",
}


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
        # Leading zeros dropped so "01035" and "1035" count as the same number
        numbers = {num.lstrip("0") or "0" for num in NUMBER_RE.findall(address_norm)}
        out["address_numbers"].append(" ".join(sorted(numbers)))
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


# ──────────────────────────────────────────────────────────────────────
# Snapping S2/S3 words to the S1 vocabulary (run once per split)
# ──────────────────────────────────────────────────────────────────────
# Transliterated words are spelled by sound ("modarn solyushans", "praivet")
# and website names are glued ("anilandevelopers"), so they never equal the
# English words S1 uses. S1 is the clean reference list of the same split, so
# its own words are the vocabulary: a transliterated word is replaced by the
# most frequent S1 word with the same sound-alike key, and a glued name is
# split into S1 words. Uses only the provided data (no labels, no lookups).

_PHONETIC_SUBS = (
    ("tion", "shn"), ("sion", "shn"), ("ture", "chr"),
    ("chh", "C"), ("ch", "C"), ("ph", "f"), ("ck", "k"), ("sh", "s"), ("th", "t"),
    ("kh", "k"), ("gh", "g"), ("bh", "b"), ("dh", "d"), ("jh", "j"),
    ("q", "k"), ("x", "ks"), ("z", "s"), ("w", "b"), ("v", "b"),
)
_SOFT_C_RE = re.compile(r"c(?=[eiy])")
_DROP_RE = re.compile(r"[aeiouyh]")
_REPEAT_RE = re.compile(r"(.)\1+")


def phonetic_key(word: str) -> str:
    """
    Sound-alike key of a word: consonant skeleton after merging similar sounds.

    "modern" / "modarn" → "mdrn", "private" / "praivet" / "praibhet" → "prbt",
    "solutions" / "solyushans" → "slsns", "services" / "sarvisas" → "srbs".

    Args:
        word: Lowercase alphabetic token.

    Returns:
        The key, or "" when the word has digits or the key is shorter than 2.
    """
    if not word.isalpha():
        return ""
    for old, new in _PHONETIC_SUBS:
        word = word.replace(old, new)
    word = _SOFT_C_RE.sub("s", word).replace("c", "k")
    word = _REPEAT_RE.sub(r"\1", _DROP_RE.sub("", word).replace("C", "c"))
    return word if len(word) >= 2 else ""


def _count_tokens(texts, min_len: int = 2) -> dict:
    """
    Count alphabetic tokens of at least min_len letters over many texts.

    Args:
        texts: Iterable of normalized strings.
        min_len: Minimum token length to keep.

    Returns:
        Dict token → count.
    """
    counts = Counter(t for text in texts for t in text.split())
    return {t: c for t, c in counts.items() if len(t) >= min_len and t.isalpha()}


def _key_to_word(counts: dict) -> dict:
    """
    Map each sound-alike key to the most frequent word that has it.

    Args:
        counts: Dict word → count.

    Returns:
        Dict key → word.
    """
    best = {}
    for word, count in counts.items():
        key = phonetic_key(word)
        if key and count > best.get(key, ("", 0))[1]:
            best[key] = (word, count)
    return {key: word for key, (word, _) in best.items()}


def segment_glued(token: str, vocab: set, max_parts: int = 4) -> list:
    """
    Split a glued token into vocabulary words ("anilandevelopers" → anilan developers).

    A trailing web suffix (com/net/org/in) is dropped first. At most one short
    unknown piece (≤ 4 letters, e.g. initials "pn") is allowed; the result must
    contain a known word of 4+ letters.

    Args:
        token: Lowercase token.
        vocab: Set of known words.
        max_parts: Maximum number of pieces.

    Returns:
        List of pieces, or [token] when no good split exists.
    """
    body = re.sub(r"(?:com|net|org|in)$", "", token) if len(token) > 6 else token
    n = len(body)
    # best[i] = (cost, pieces, unknown_used) for body[:i]
    best = [None] * (n + 1)
    best[0] = (0, [], False)
    for end in range(1, n + 1):
        for start in range(max(0, end - 20), end):
            prev = best[start]
            if prev is None or len(prev[1]) >= max_parts:
                continue
            piece = body[start:end]
            if piece in vocab and len(piece) >= 2:
                cand = (prev[0] + 1, prev[1] + [piece], prev[2])
            elif not prev[2] and len(piece) <= 4:
                cand = (prev[0] + 3, prev[1] + [piece], True)
            else:
                continue
            if best[end] is None or cand[0] < best[end][0]:
                best[end] = cand
    if best[n] is None or len(best[n][1]) < 2 \
            or not any(len(p) >= 4 and p in vocab for p in best[n][1]):
        return [token]
    return best[n][1]


def _snap_tokens(text: str, vocab: dict, key_map: dict, glued: bool, translit: set,
                 seg_vocab: set) -> str:
    """
    Snap the tokens of one normalized text to the reference vocabulary.

    Args:
        text: Normalized text of one record.
        vocab: Reference word counts (a word already in it is kept).
        key_map: Sound-alike key → reference word.
        glued: Whether to split glued tokens (website-style names).
        translit: Tokens that came from Indic script; only these are replaced
            by their sound-alike (plain Latin words are never touched).
        seg_vocab: Words allowed as pieces of a glued token.

    Returns:
        The text with snapped tokens.
    """
    out = []
    for token in text.split():
        if token in vocab:
            out.append(token)
            continue
        pieces = segment_glued(token, seg_vocab) if glued and len(token) >= 7 else [token]
        for piece in pieces:
            if piece in translit and piece not in vocab:
                piece = key_map.get(phonetic_key(piece), piece)
            out.append(piece)
    return " ".join(out)


def _indic_tokens(raw: str) -> set:
    """
    Normalized tokens that come from the Indic-script words of a raw text.

    Args:
        raw: Raw name or address.

    Returns:
        Set of transliterated tokens (empty when the text has no Indic script).
    """
    if not raw:
        return set()
    return {t for word in re.split(r"[\s,]+", raw) if INDIC_RE.search(word)
            for t in normalize_text(word).split()}


def snap_to_reference(tgt: pd.DataFrame, s1: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Rewrite transliterated and website-style S2/S3 names/addresses in S1 words.

    Only rows whose raw name/address contains Indic script, or whose raw name
    looks like a website/handle, are touched. name_norm, name_core and
    address_norm of those rows are recomputed.

    Args:
        tgt: Normalized S2+S3 frame (normalize_frame output).
        s1: Normalized S1 frame of the same split (all of it, before sampling).
        verbose: Whether to print a summary.

    Returns:
        tgt with the rewritten columns (same row order).
    """
    t0 = time.time()
    indic = (tgt["business_name"].str.contains(INDIC_PATTERN, regex=True)
             | tgt["business_address"].str.contains(INDIC_PATTERN, regex=True))
    web = tgt["business_name"].str.contains(WEB_PATTERN, regex=True)
    indic = indic.fillna(False).to_numpy(dtype=bool)
    web = web.fillna(False).to_numpy(dtype=bool)
    rows = np.flatnonzero(indic | web)
    if len(rows) == 0:
        return tgt

    name_counts = _count_tokens(s1["name_norm"].tolist())
    addr_counts = _count_tokens(s1["address_norm"].tolist())
    name_keys, addr_keys = _key_to_word(name_counts), _key_to_word(addr_counts)
    # S1 spells abbreviations out ("infrastructure"), transliterations keep the
    # short form ("inphra"): let the short form's key point to the full word
    for short, full in NAME_ABBREVIATIONS.items():
        key = phonetic_key(short)
        if len(key) >= 3 and full in name_counts:
            name_keys.setdefault(key, full)
    seg_vocab = set(name_counts)

    names = tgt["name_norm"].iloc[rows].tolist()
    addrs = tgt["address_norm"].iloc[rows].tolist()
    new_names, new_cores, new_addrs = [], [], []
    raw_names = tgt["business_name"].iloc[rows].tolist()
    raw_addrs = tgt["business_address"].iloc[rows].tolist()
    for name, addr, raw_name, raw_addr, is_indic, is_web in zip(
            names, addrs, raw_names, raw_addrs, indic[rows], web[rows]):
        name_translit = _indic_tokens(raw_name) if is_indic else set()
        addr_translit = _indic_tokens(raw_addr) if is_indic else set()
        name = _snap_tokens(name, name_counts, name_keys, is_web, name_translit, seg_vocab)
        name = expand_abbreviations(name, NAME_ABBREVIATIONS)
        new_names.append(name)
        new_cores.append(extract_name_core(name))
        new_addrs.append(_snap_tokens(addr, addr_counts, addr_keys, False, addr_translit, seg_vocab)
                         if addr_translit else addr)

    # Replace only the touched rows (no Python object per record for 10M rows)
    mask = pa.array(np.isin(np.arange(len(tgt)), rows))
    out = tgt.copy()
    for col, values in (("name_norm", new_names), ("name_core", new_cores),
                        ("address_norm", new_addrs)):
        column = pa.array(tgt[col].array)
        if isinstance(column, pa.ChunkedArray):
            column = column.combine_chunks()
        column = pc.replace_with_mask(column, mask, pa.array(values, type=column.type))
        out[col] = pd.array(column, dtype=tgt[col].dtype)
    if verbose:
        print(f"  Snapped {int(indic.sum()):,} Indic-script and {int(web.sum()):,} "
              f"website-style S2/S3 records to the S1 vocabulary ({time.time() - t0:.0f}s)")
    return out
