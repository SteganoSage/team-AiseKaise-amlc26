"""
Pair feature extraction for the entity resolution classifier.

For each (S1, candidate) pair, computes a feature vector. Groups can be
switched on/off in config.FEATURE_GROUPS for ablations:

- string:    rapidfuzz ratio / partial_ratio / token_sort / token_set /
             Jaro-Winkler on name_norm, name_core and address_norm.
- tokens:    token Jaccard / overlap on name_core and address tokens.
- tfidf:     TF-IDF cosine on name chars, address chars, name+address words.
- numbers:   overlap of all numbers in the address.
- structure: postal code / house number / country match flags, lengths,
             missing-field flags, source (S2 vs S3).
- rank:      how a pair compares with its competitors — rank and gap to the
             best among the S1's candidates and among the candidate's S1s,
             mutual best, candidate counts. Needs the whole batch of pairs,
             so always featurize all S1 records of a split in one call.
- blockers:  which blockers produced the pair.
- embedding: embedding cosine of names and of name+address (USE_EMBEDDINGS).

All features are country-agnostic (no one-hot country) so France generalizes.
Computation is vectorized over all pairs (rapidfuzz.process.cpdist runs in
C++ on all cores) instead of a Python loop per pair.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

import config
from blocking import BLOCKER_BITS, build_tfidf_index


# rapidfuzz scorers used on every text field; fuzz scores are 0-100, JW is 0-1
STRING_SCORERS = {
    "ratio": (fuzz.ratio, 100.0),
    "partial_ratio": (fuzz.partial_ratio, 100.0),
    "token_sort_ratio": (fuzz.token_sort_ratio, 100.0),
    "token_set_ratio": (fuzz.token_set_ratio, 100.0),
    "jaro_winkler": (JaroWinkler.similarity, 1.0),
}


# ──────────────────────────────────────────────────────────────────────
# Feature groups (each returns an ordered dict name → float32 array)
# ──────────────────────────────────────────────────────────────────────

def string_similarities(left: list, right: list, prefix: str) -> dict:
    """
    rapidfuzz similarity suite for aligned lists of strings.

    A pair where either string is empty gets 0 for every score.

    Args:
        left: S1-side strings, one per pair.
        right: Candidate-side strings, one per pair.
        prefix: Feature name prefix, e.g. "name".

    Returns:
        Dict of feature name → array in [0, 1].
    """
    both = np.array([bool(a) and bool(b) for a, b in zip(left, right)])
    out = {}
    for name, (scorer, scale) in STRING_SCORERS.items():
        scores = process.cpdist(left, right, scorer=scorer,
                                dtype=np.float32, workers=-1)
        out[f"{prefix}_{name}"] = np.where(both, scores / scale, 0.0)
    return out


def token_overlaps(left: list, right: list, prefix: str) -> dict:
    """
    Token-set overlap features for aligned lists of token sets.

    Args:
        left: S1-side token sets, one per pair.
        right: Candidate-side token sets, one per pair.
        prefix: Feature name prefix, e.g. "name".

    Returns:
        Dict with Jaccard, overlap count, and overlap as a fraction of each side.
    """
    inter = np.array([len(a & b) for a, b in zip(left, right)], dtype=np.float32)
    union = np.array([len(a | b) for a, b in zip(left, right)], dtype=np.float32)
    len_left = np.array([len(a) for a in left], dtype=np.float32)
    len_right = np.array([len(b) for b in right], dtype=np.float32)
    return {
        f"{prefix}_jaccard": np.where(union > 0, inter / np.maximum(union, 1), 0.0),
        f"{prefix}_token_overlap": inter,
        f"{prefix}_token_overlap_frac_s1": inter / np.maximum(len_left, 1),
        f"{prefix}_token_overlap_frac_cand": inter / np.maximum(len_right, 1),
    }


def tfidf_pair_cosine(left_texts: list, right_texts: list,
                      left_idx: np.ndarray, right_idx: np.ndarray,
                      analyzer: str, ngram_range: tuple) -> np.ndarray:
    """
    TF-IDF cosine for each pair, fitting one vocabulary on both sides.

    Args:
        left_texts: Texts of the unique S1 records.
        right_texts: Texts of the unique candidate records.
        left_idx: Per pair, index into left_texts.
        right_idx: Per pair, index into right_texts.
        analyzer: 'char_wb' or 'word'.
        ngram_range: n-gram range for the vectorizer.

    Returns:
        float32 cosine per pair (0 if either text is empty).
    """
    out = np.zeros(len(left_idx), dtype=np.float32)
    try:
        _, matrix = build_tfidf_index(left_texts + right_texts,
                                      analyzer=analyzer, ngram_range=ngram_range)
    except ValueError:  # empty vocabulary (all texts empty)
        return out
    left_m = matrix[: len(left_texts)]
    right_m = matrix[len(left_texts):]

    # Rows are L2-normalized, so the row-wise dot product is the cosine
    step = config.FEATURE_CHUNK_SIZE
    for start in range(0, len(left_idx), step):
        end = start + step
        prod = left_m[left_idx[start:end]].multiply(right_m[right_idx[start:end]])
        out[start:end] = np.asarray(prod.sum(axis=1)).ravel()
    return out


def number_overlaps(left: list, right: list) -> dict:
    """
    Overlap of all numbers found in the two addresses.

    Args:
        left: S1-side address number sets, one per pair.
        right: Candidate-side address number sets, one per pair.

    Returns:
        Dict with number Jaccard, overlap count, both-present and
        "both have numbers but none shared" flags.
    """
    base = token_overlaps(left, right, "addr_num")
    both = np.array([bool(a) and bool(b) for a, b in zip(left, right)], dtype=np.float32)
    return {
        "addr_num_jaccard": base["addr_num_jaccard"],
        "addr_num_overlap": base["addr_num_token_overlap"],
        "addr_num_both_present": both,
        "addr_num_conflict": both * (base["addr_num_token_overlap"] == 0),
    }


def structure_features(left: list, right: list) -> dict:
    """
    Postal code, house number, country, length, missing-field and source features.

    Args:
        left: S1 records, one per pair.
        right: Candidate records, one per pair.

    Returns:
        Dict of feature name → array.
    """
    def flags(fn) -> np.ndarray:
        """Apply fn(s1_rec, cand_rec) → bool to every pair."""
        return np.array([fn(a, b) for a, b in zip(left, right)], dtype=np.float32)

    def lengths(recs: list, field: str) -> np.ndarray:
        """Length of a text field for every record."""
        return np.array([len(r.get(field, "")) for r in recs], dtype=np.float32)

    s1_name_len, cand_name_len = lengths(left, "name_norm"), lengths(right, "name_norm")
    s1_addr_len, cand_addr_len = lengths(left, "address_norm"), lengths(right, "address_norm")
    is_s2 = np.array([r["entity_id"].startswith("S2-") for r in right], dtype=np.float32)
    is_s3 = np.array([r["entity_id"].startswith("S3-") for r in right], dtype=np.float32)

    return {
        "postal_code_match": flags(
            lambda a, b: bool(set(a["postal_codes"]) & set(b["postal_codes"]))),
        "postal_code_both_present": flags(
            lambda a, b: bool(a["postal_codes"]) and bool(b["postal_codes"])),
        "postal_code_either_present": flags(
            lambda a, b: bool(a["postal_codes"]) or bool(b["postal_codes"])),
        "house_number_match": flags(
            lambda a, b: bool(a["house_number"]) and a["house_number"] == b["house_number"]),
        "house_number_both_present": flags(
            lambda a, b: bool(a["house_number"]) and bool(b["house_number"])),
        "country_match": flags(
            lambda a, b: bool(a["country_norm"]) and a["country_norm"] == b["country_norm"]),
        "country_both_present": flags(
            lambda a, b: bool(a["country_norm"]) and bool(b["country_norm"])),
        "s1_name_len": s1_name_len,
        "cand_name_len": cand_name_len,
        "name_len_diff": np.abs(s1_name_len - cand_name_len),
        "name_len_ratio": np.minimum(s1_name_len, cand_name_len)
                          / np.maximum(np.maximum(s1_name_len, cand_name_len), 1),
        "s1_addr_len": s1_addr_len,
        "cand_addr_len": cand_addr_len,
        "addr_len_diff": np.abs(s1_addr_len - cand_addr_len),
        "s1_name_missing": (s1_name_len == 0).astype(np.float32),
        "cand_name_missing": (cand_name_len == 0).astype(np.float32),
        "s1_addr_missing": (s1_addr_len == 0).astype(np.float32),
        "cand_addr_missing": (cand_addr_len == 0).astype(np.float32),
        "is_s2": is_s2,
        "is_s3": is_s3,
    }


def rank_features(s1_ids: list, cand_ids: list, sims: dict) -> dict:
    """
    Compare each pair with its competitors.

    For every similarity in `sims`:
    - rank / gap to best among all candidates of the same S1
    - rank / gap to best among all S1s that shortlisted the same candidate
    - mutual best: best candidate for its S1 AND best S1 for its candidate
    Plus how many candidates the S1 has and how many S1s the candidate has.

    Args:
        s1_ids: S1 entity ID per pair.
        cand_ids: Candidate entity ID per pair.
        sims: Dict short name → similarity array (higher = more similar).

    Returns:
        Dict of feature name → array.
    """
    df = pd.DataFrame({"s1": s1_ids, "cand": cand_ids, **sims})
    by_s1, by_cand = df.groupby("s1", sort=False), df.groupby("cand", sort=False)
    out = {}
    for key in sims:
        rank_s1 = by_s1[key].rank(ascending=False, method="min").to_numpy()
        rank_cand = by_cand[key].rank(ascending=False, method="min").to_numpy()
        out[f"rank_{key}_in_s1"] = rank_s1
        out[f"gap_{key}_to_s1_best"] = (by_s1[key].transform("max") - df[key]).to_numpy()
        out[f"rank_{key}_in_cand"] = rank_cand
        out[f"gap_{key}_to_cand_best"] = (by_cand[key].transform("max") - df[key]).to_numpy()
        out[f"mutual_best_{key}"] = ((rank_s1 == 1) & (rank_cand == 1)).astype(np.float32)
    out["n_cands_for_s1"] = by_s1["cand"].transform("size").to_numpy()
    out["n_s1_for_cand"] = by_cand["s1"].transform("size").to_numpy()
    return out


def blocker_features(pairs: list, sources: dict) -> dict:
    """
    Which blockers produced each pair (from blocking.generate_candidates).

    Args:
        pairs: List of (s1_id, cand_id).
        sources: Dict S1 entity_id → {candidate ID: BLOCKER_BITS bitmask}.

    Returns:
        Dict with one 0/1 flag per blocker and the number of blockers.
    """
    masks = np.array([sources[s].get(c, 0) for s, c in pairs], dtype=np.int64)
    out = {}
    for name, bit in BLOCKER_BITS.items():
        if name == "embedding" and not config.USE_EMBEDDINGS:
            continue
        out[f"found_by_{name}"] = ((masks & bit) > 0).astype(np.float32)
    out["n_blockers"] = np.sum(list(out.values()), axis=0).astype(np.float32)
    return out


# ──────────────────────────────────────────────────────────────────────
# Feature matrix
# ──────────────────────────────────────────────────────────────────────

def build_feature_matrix(s1_records_map: dict, candidate_records_map: dict,
                         candidates: dict, sources: dict = None,
                         embeddings: dict = None, verbose: bool = True) -> tuple:
    """
    Build a feature matrix for all candidate pairs.

    Pass all S1 records of a split in one call: the rank features compare
    each pair with the other pairs of the same S1 and the same candidate.

    Args:
        s1_records_map: Dict mapping S1 entity_id → normalized record dict.
        candidate_records_map: Dict mapping S2/S3 entity_id → normalized record dict.
        candidates: Dict mapping S1 entity_id → list of candidate entity_ids.
        sources: Optional blocker bitmasks from blocking.generate_candidates.
        embeddings: Optional output of embeddings.embed_records for this split.
        verbose: Whether to print progress.

    Returns:
        Tuple of:
        - numpy array of shape (n_pairs, n_features), float32
        - list of (s1_id, cand_id) tuples (same order as rows)
        - list of feature names
    """
    groups = config.FEATURE_GROUPS
    pairs = [
        (s1_id, cand_id)
        for s1_id in sorted(candidates)
        for cand_id in candidates[s1_id]
        if cand_id in candidate_records_map
    ]
    s1_ids = [p[0] for p in pairs]
    cand_ids = [p[1] for p in pairs]
    left = [s1_records_map[s] for s in s1_ids]
    right = [candidate_records_map[c] for c in cand_ids]

    cols = {}
    if groups.get("string", True):
        for field, prefix in (("name_norm", "name"), ("name_core", "name_core"),
                              ("address_norm", "addr")):
            cols.update(string_similarities(
                [r[field] for r in left], [r[field] for r in right], prefix))

    if groups.get("tokens", True):
        cols.update(token_overlaps([r["name_tokens"] for r in left],
                                   [r["name_tokens"] for r in right], "name"))
        cols.update(token_overlaps([r["address_tokens"] for r in left],
                                   [r["address_tokens"] for r in right], "addr"))

    tfidf = {}
    if groups.get("tfidf", True) or groups.get("rank", True):
        # Unique records on each side, so each text is vectorized once
        u_s1 = list(dict.fromkeys(s1_ids))
        u_cand = list(dict.fromkeys(cand_ids))
        li = pd.Index(u_s1).get_indexer(s1_ids)
        ri = pd.Index(u_cand).get_indexer(cand_ids)
        s1_recs = [s1_records_map[s] for s in u_s1]
        cand_recs = [candidate_records_map[c] for c in u_cand]
        for name, text_fn, analyzer, ngrams in (
            ("tfidf_name_char", lambda r: r["name_core"], "char_wb", config.TFIDF_NGRAM_RANGE),
            ("tfidf_addr_char", lambda r: r["address_norm"], "char_wb", config.TFIDF_NGRAM_RANGE),
            ("tfidf_name_addr_word", lambda r: r["name_norm"] + " " + r["address_norm"],
             "word", (1, 2)),
        ):
            tfidf[name] = tfidf_pair_cosine(
                [text_fn(r) for r in s1_recs], [text_fn(r) for r in cand_recs],
                li, ri, analyzer, ngrams)
        if groups.get("tfidf", True):
            cols.update(tfidf)

    if groups.get("numbers", True):
        cols.update(number_overlaps([r["address_numbers"] for r in left],
                                    [r["address_numbers"] for r in right]))

    if groups.get("structure", True):
        cols.update(structure_features(left, right))

    if groups.get("rank", True):
        cols.update(rank_features(s1_ids, cand_ids, {
            "name": tfidf["tfidf_name_char"],
            "combined": tfidf["tfidf_name_addr_word"],
        }))

    if groups.get("blockers", True) and sources is not None:
        cols.update(blocker_features(pairs, sources))

    if groups.get("embedding", True) and embeddings is not None:
        from embeddings import pair_cosine
        cols["emb_name_cos"] = pair_cosine(embeddings, "name", s1_ids, cand_ids)
        cols["emb_full_cos"] = pair_cosine(embeddings, "full", s1_ids, cand_ids)

    feature_names = list(cols)
    if pairs:
        X = np.column_stack([np.asarray(cols[n], dtype=np.float32) for n in feature_names])
    else:
        X = np.empty((0, len(feature_names)), dtype=np.float32)

    if verbose:
        print(f"  Built feature matrix: {len(pairs):,} pairs × {len(feature_names)} features")

    return X, pairs, feature_names
