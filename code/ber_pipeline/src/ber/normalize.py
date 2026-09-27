"""Text normalisation for business names and addresses.

Source 1 is clean; Sources 2/3 add noise: other scripts, injected accents,
junk punctuation, moved or swapped legal suffixes, filler words, DBA wrappers,
domain/handle-style names, rotated address components, state-name variants,
null placeholders and house-number formatting. The functions here map every
record to the same canonical forms so that similarity features and blocking
compare like with like. Nothing here is country-specific except the optional
state maps, which fall back to "no state" for unseen countries.
"""
import re

from anyascii import anyascii

INDIC_RE = re.compile(r"[ऀ-෿]")
_PRE = str.maketrans({"°": "o ", "º": "o ", "’": "'", "`": "'", "&": " and "})
_NONALNUM = re.compile(r"[^a-z0-9]+")
_INITIALS = re.compile(r"\b(?:[a-z]\s?\.\s?){2,}")          # l.l.c.  s.a.r.l.
_ID_TAG = re.compile(r"\(?\bid\s*:\s*\d+\)?")                # (ID: 84803)
_DBA = re.compile(
    r"\b(?:d\s*/\s*b\s*/\s*a|dba|doing business as|f\s*/\s*k\s*/\s*a|fka|"
    r"formerly known as|formerly|a\s*/\s*k\s*/\s*a|aka|t\s*/\s*a|trading as)\b")
_DOMAIN = re.compile(r"^(?:www\.)?([a-z0-9\-']+)\.(?:com|c0m|net|org|in|co|fr|biz|info)$")
_LEET = str.maketrans("013458", "oleasb")

LEGAL = {
    "llc": "llc", "inc": "inc", "incorporated": "inc", "lncorporated": "inc",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "cie": "co",
    "compagnie": "co", "ltd": "ltd", "limited": "ltd", "limtid": "ltd",
    "pvt": "pvt", "private": "pvt", "praivet": "pvt", "llp": "llp", "lp": "lp",
    "plc": "plc", "pc": "pc", "pllc": "pllc", "pa": "pa", "public": "public",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci",
    "sa": "sa", "snc": "snc", "ei": "ei", "gmbh": "gmbh",
}
LEGAL_INDIC = {"pra": "pvt", "li": "ltd"}  # "प्रा. लि." after transliteration
FILLER = {
    "the", "center", "centre", "service", "services", "group", "groupe",
    "partners", "holding", "holdings", "and", "of", "fils", "france", "ets",
    "etablissements", "la", "le", "les", "de", "du", "des", "d", "l", "et",
}
HONORIFIC = {"smt", "shri", "sri", "shree", "mr", "mrs", "ms", "dr", "m", "s"}

# ---------------------------------------------------------------- addresses
ADDR_MAP = {
    "street": "st", "saint": "st", "str": "st", "road": "rd", "avenue": "ave",
    "av": "ave", "drive": "dr", "drve": "dr", "lane": "ln", "court": "ct",
    "circle": "cir", "place": "pl", "boulevard": "blvd", "bd": "blvd",
    "terrace": "ter", "highway": "hwy", "parkway": "pkwy", "turnpike": "tpke",
    "trail": "trl", "square": "sq", "north": "n", "south": "s", "east": "e",
    "west": "w", "apartment": "apt", "appt": "apt", "app": "apt",
    "building": "bldg", "floor": "fl", "suite": "ste", "sainte": "ste",
    "fort": "ft", "mount": "mt", "township": "twp", "twnship": "twp",
    "rue": "r", "allee": "all", "alee": "all", "impasse": "imp", "chemin": "ch",
    "route": "rte", "opposite": "opp", "nr": "near", "sector": "sec",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
}
ADDR_DROP = {"no", "hno", "hn", "h", "door", "house"}
_NULLS = {"", "null", "n a", "na", "none", "nan", "unknown"}
_PO_BOX = re.compile(r"\b(?:po box|p o box|pmb)\s*\d+")
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_NUM_ALPHA = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "district of columbia": "dc", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "delhi": "dl", "jammu and kashmir": "jk", "ladakh": "la", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py", "andaman and nicobar islands": "an",
    "dadra and nagar haveli and daman and diu": "dn", "lakshadweep": "ld",
}
_IN_EXTRA = {"ts": "tg", "ct": "cg", "or": "od", "ua": "uk", "orrisa": "od"}


def _state_map(names, extra=()):
    """{full state name or code -> code} lookup, plus extra aliases."""
    m = dict(names)
    m.update({code: code for code in names.values()})
    m.update(extra)
    return m


STATE_MAPS = {
    "US": _state_map(US_STATES),
    "India": _state_map(IN_STATES, _IN_EXTRA),
}


def to_ascii(s):
    """Lower-case ASCII transliteration of any script (anyascii), after symbol fixes."""
    return anyascii(s.translate(_PRE)).lower()


_ASPIRATES = (("ph", "f"), ("sh", "s"), ("kh", "k"), ("gh", "g"), ("bh", "b"),
              ("dh", "d"), ("th", "t"), ("ch", "c"), ("jh", "j"))
_NASAL = re.compile(r"m(?=[^aeiouy])")  # transliterated anusvara: "marketimg"
_WEAK = re.compile(r"[aeiouyhw]")
_REPEAT = re.compile(r"(.)\1+")


def skeleton(tokens):
    """First letter + consonants per token.

    Transliteration from Indic scripts drops inherent vowels ("skti bildrs"
    for "shakti builders"); comparing consonant skeletons lines them up.
    """
    out = []
    for t in tokens:
        if t.isdigit():
            out.append(t)
            continue
        for a, b in _ASPIRATES:
            t = t.replace(a, b)
        t = _NASAL.sub("n", t)
        t = _REPEAT.sub(r"\1", t[0] + _WEAK.sub("", t[1:]))
        out.append(t)
    return " ".join(out)


def _unleet(tok):
    """Undo digit-for-letter substitutions ("0rthopedic") in mostly-alphabetic tokens."""
    letters = sum(c.isalpha() for c in tok)
    if letters >= 3 and letters < len(tok):
        return tok.translate(_LEET)
    return tok


def norm_name(raw):
    """Return a dict of name forms and flags for one business name."""
    indic = bool(INDIC_RE.search(raw))
    s = _ID_TAG.sub(" ", to_ascii(raw)).split(" | ")[0]
    alt = ""
    m = _DBA.search(s)
    if m and s[m.end():].strip():
        alt, s = s[:m.start()], s[m.end():]
    s = _INITIALS.sub(lambda g: re.sub(r"[\s.]", "", g.group(0)) + " ", s)

    compact = False
    words = []
    for w in s.split():
        w = w.strip("@#*<>-()[]{}\"'!?;:")
        d = _DOMAIN.match(w)
        if d:
            w, compact = d.group(1), True
        elif len(w) >= 12 and w.endswith("com") and w.isalpha():
            w, compact = w[:-3], True
        words.append(w)
    raw_tokens = _NONALNUM.sub(" ", " ".join(words)).split()
    if len(raw_tokens) == 1 and len(raw_tokens[0]) >= 10:
        compact = True
    legal_map = {**LEGAL, **LEGAL_INDIC} if indic else LEGAL

    tokens, legal, core = [], [], []
    for t in raw_tokens:
        t = _unleet(t)
        lg = legal_map.get(t)
        if lg:
            legal.append(lg)
            tokens.append(lg)
            continue
        tokens.append(t)
        if t not in FILLER and t not in HONORIFIC and t not in core:
            core.append(t)
    if not core:
        core = [t for t in tokens if t not in FILLER] or tokens
    alt_tokens = _NONALNUM.sub(" ", alt).split()
    return {
        "n_full": " ".join(tokens),
        "n_core": " ".join(core),
        "n_compact": "".join(core),
        "n_skel": skeleton(core),
        "n_legal": " ".join(sorted(set(legal))),
        "n_alt": " ".join(alt_tokens),
        "n_is_compact": compact,
        "n_has_dba": bool(m),
        "n_is_indic": indic,
    }


def norm_addr(raw, country, state_aliases=None):
    """Return a dict of address forms for one address.

    state_aliases: optional {ascii component -> state code} learned from
    training data (e.g. transliterated Indic state names).
    """
    indic = bool(INDIC_RE.search(raw))
    smap = STATE_MAPS.get(country, {})
    comps = to_ascii(raw).split(",")
    state, kept = "", []
    for c in comps:
        key = " ".join(_NONALNUM.sub(" ", c).split())
        if key in _NULLS:
            continue
        code = smap.get(key) or (state_aliases or {}).get(key)
        if code:
            state = state or code
            continue
        kept.append(c)
    text = _PO_BOX.sub(" ", " ".join(kept))
    text = _ORDINAL.sub(r"\1", text)
    text = _NUM_ALPHA.sub(" ", _NONALNUM.sub(" ", text))
    tokens = []
    for t in text.split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            t = ADDR_MAP.get(t, t)
        if t not in ADDR_DROP:
            tokens.append(t)
    nums = []
    for t in tokens:
        if t.isdigit() and t not in nums:
            nums.append(t)
    return {
        "a_norm": " ".join(tokens),
        "a_state": state,
        "a_nums": " ".join(nums),
        "a_is_indic": indic,
        "a_is_empty": not tokens,
    }


def state_key(component):
    """Canonical lookup key for one raw address component."""
    return " ".join(_NONALNUM.sub(" ", to_ascii(component)).split())
