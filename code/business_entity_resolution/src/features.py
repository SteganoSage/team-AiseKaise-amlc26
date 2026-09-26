"""
Stage 3 — pair features for the final matching model.

Computed only for the pruned candidate pairs (candidate_pairs.tsv), in
chunks, straight from the compact normalized frames. Groups can be switched
on/off in config.FEATURE_GROUPS for ablations:

- string:    rapidfuzz ratio / partial_ratio / token_sort / token_set /
             Jaro-Winkler on name_norm, name_core and address_norm.
- tokens:    token Jaccard / overlap on name_core and address tokens.
- tfidf:     blocking TF-IDF cosines (name, name+address) and their ranks,
             reused from stage 1 (no second vectorization of millions of rows).
- numbers:   overlap of all numbers in the address.
- structure: postal code / house number / country match flags, lengths,
             missing-field flags, source (S2 vs S3).
- rank:      how a pair compares with its competitors in the candidate set —
             rank and gap to the best among the S1's candidates and among the
             candidate's S1s, mutual best, candidate counts.
- blockers:  which blockers proposed the pair.
- pruner:    stage-2 pruner probability.
- embedding: embedding cosines (only when USE_EMBEDDINGS).
- frequency: how many S1 / S2+S3 records share the S1's and the candidate's
             name_core and address (Fellegi-Sunter style: agreeing on a name
             90 S1 records share is weak evidence), counted over the whole
             split before any S1 sampling (add_frequency_columns).
- cross:     name × address interactions (product, min, max), exact name /
             address flags, first / last name word equal.
- char_ngram: character 3-gram cosine of name_core (typo tolerant).

All features are country-agnostic (no one-hot country) so France generalizes.
"""

import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import HashingVectorizer

import config
from blocking import BLOCKER_BITS, _pool, group_rank, rowwise_cosine
from prune import take_strings


# rapidfuzz scorers used on every text field; fuzz scores are 0-100, JW is 0-1
STRING_SCORERS = {
    "ratio": (fuzz.ratio, 100.0),
    "partial_ratio": (fuzz.partial_ratio, 100.0),
    "token_sort_ratio": (fuzz.token_sort_ratio, 100.0),
    "token_set_ratio": (fuzz.token_set_ratio, 100.0),
    "jaro_winkler": (JaroWinkler.similarity, 1.0),
}


# ──────────────────────────────────────────────────────────────────────
# Feature groups on one chunk of pairs (lists of strings in, arrays out)
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
        scores = process.cpdist(left, right, scorer=scorer, dtype=np.float32, workers=-1)
        out[f"{prefix}_{name}"] = np.where(both, scores / scale, 0.0)
    return out


def token_overlaps(left: list, right: list, prefix: str) -> dict:
    """
    Token-set overlap features for aligned lists of space-separated strings.

    Args:
        left: S1-side strings, one per pair.
        right: Candidate-side strings, one per pair.
        prefix: Feature name prefix, e.g. "name".

    Returns:
        Dict with Jaccard, overlap count, and overlap as a fraction of each side.
    """
    left_sets = [set(s.split()) for s in left]
    right_sets = [set(s.split()) for s in right]
    inter = np.array([len(a & b) for a, b in zip(left_sets, right_sets)], dtype=np.float32)
    len_left = np.array([len(a) for a in left_sets], dtype=np.float32)
    len_right = np.array([len(b) for b in right_sets], dtype=np.float32)
    union = len_left + len_right - inter
    return {
        f"{prefix}_jaccard": np.where(union > 0, inter / np.maximum(union, 1), 0.0),
        f"{prefix}_token_overlap": inter,
        f"{prefix}_token_overlap_frac_s1": inter / np.maximum(len_left, 1),
        f"{prefix}_token_overlap_frac_cand": inter / np.maximum(len_right, 1),
    }


def number_overlaps(left: list, right: list) -> dict:
    """
    Overlap of all numbers found in the two addresses.

    Args:
        left: S1-side address_numbers strings (space-separated numbers).
        right: Candidate-side address_numbers strings.

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


def edge_word_matches(left: list, right: list) -> dict:
    """
    Whether the first / last words of two names are equal.

    Args:
        left: S1-side name_core strings, one per pair.
        right: Candidate-side name_core strings.

    Returns:
        Dict with name_first_word_equal and name_last_word_equal (0/1).
    """
    first = np.fromiter((bool(a) and bool(b) and a.split(" ", 1)[0] == b.split(" ", 1)[0]
                         for a, b in zip(left, right)), dtype=np.float32, count=len(left))
    last = np.fromiter((bool(a) and bool(b) and a.rsplit(" ", 1)[-1] == b.rsplit(" ", 1)[-1]
                        for a, b in zip(left, right)), dtype=np.float32, count=len(left))
    return {"name_first_word_equal": first, "name_last_word_equal": last}


# Character 3-grams inside word boundaries, hashed (no vocabulary to hold)
CHAR_NGRAM_VECTORIZER = HashingVectorizer(
    analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 20,
    alternate_sign=False, norm="l2", lowercase=False, dtype=np.float32)


def _char_cosine_task(texts: tuple) -> np.ndarray:
    """
    Character 3-gram cosine of aligned string lists (worker task).

    Args:
        texts: Tuple (left strings, right strings).

    Returns:
        Cosine per pair in [0, 1] (0 when either string is empty).
    """
    left, right = texts
    a = CHAR_NGRAM_VECTORIZER.transform(left)
    b = CHAR_NGRAM_VECTORIZER.transform(right)
    return np.asarray(a.multiply(b).sum(axis=1), dtype=np.float32).ravel()


def char_ngram_cosine(left: list, right: list, task_size: int = 100_000) -> np.ndarray:
    """
    Character 3-gram cosine for many pairs, split over worker processes.

    Catches typos that break whole words ("cornerstone trasit" vs
    "cornerstone transit"), which word-level scores miss.

    Args:
        left: S1-side strings, one per pair.
        right: Candidate-side strings.
        task_size: Pairs per worker task.

    Returns:
        Cosine per pair.
    """
    tasks = [(left[i:i + task_size], right[i:i + task_size])
             for i in range(0, len(left), task_size)]
    if config.N_JOBS > 1 and len(tasks) > 1:
        with _pool(min(config.N_JOBS, len(tasks))) as pool:
            parts = pool.map(_char_cosine_task, tasks)
    else:
        parts = [_char_cosine_task(t) for t in tasks]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def structure_features(get) -> dict:
    """
    Postal code, house number, country, length, missing-field features.

    Args:
        get: Function (side, column) → list of strings for this chunk,
            side in {"s1", "cand"}.

    Returns:
        Dict of feature name → array.
    """
    def flags(values) -> np.ndarray:
        """Bool iterable → float32 array."""
        return np.fromiter(values, dtype=np.float32)

    pc_l = [set(s.split()) for s in get("s1", "postal_codes")]
    pc_r = [set(s.split()) for s in get("cand", "postal_codes")]
    hn_l, hn_r = get("s1", "house_number"), get("cand", "house_number")
    co_l, co_r = get("s1", "country_norm"), get("cand", "country_norm")
    lengths = {
        (side, col): np.fromiter((len(s) for s in get(side, col)), dtype=np.float32)
        for side in ("s1", "cand") for col in ("name_norm", "address_norm")
    }
    s1_name, cand_name = lengths[("s1", "name_norm")], lengths[("cand", "name_norm")]
    s1_addr, cand_addr = lengths[("s1", "address_norm")], lengths[("cand", "address_norm")]

    return {
        "postal_code_match": flags(bool(a & b) for a, b in zip(pc_l, pc_r)),
        "postal_code_both_present": flags(bool(a) and bool(b) for a, b in zip(pc_l, pc_r)),
        "postal_code_either_present": flags(bool(a) or bool(b) for a, b in zip(pc_l, pc_r)),
        "house_number_match": flags(bool(a) and a == b for a, b in zip(hn_l, hn_r)),
        "house_number_both_present": flags(bool(a) and bool(b) for a, b in zip(hn_l, hn_r)),
        "country_match": flags(bool(a) and a == b for a, b in zip(co_l, co_r)),
        "country_both_present": flags(bool(a) and bool(b) for a, b in zip(co_l, co_r)),
        "s1_name_len": s1_name,
        "cand_name_len": cand_name,
        "name_len_diff": np.abs(s1_name - cand_name),
        "name_len_ratio": np.minimum(s1_name, cand_name)
                          / np.maximum(np.maximum(s1_name, cand_name), 1),
        "s1_addr_len": s1_addr,
        "cand_addr_len": cand_addr,
        "addr_len_diff": np.abs(s1_addr - cand_addr),
        "s1_name_missing": (s1_name == 0).astype(np.float32),
        "cand_name_missing": (cand_name == 0).astype(np.float32),
        "s1_addr_missing": (s1_addr == 0).astype(np.float32),
        "cand_addr_missing": (cand_addr == 0).astype(np.float32),
    }


def chunk_features(s1: pd.DataFrame, tgt: pd.DataFrame, s1_rows: np.ndarray,
                   tgt_rows: np.ndarray) -> dict:
    """
    All string-based feature groups for one chunk of pairs.

    Args:
        s1: Normalized S1 frame.
        tgt: Normalized S2 + S3 frame.
        s1_rows: S1 row per pair.
        tgt_rows: Target row per pair.

    Returns:
        Ordered dict of feature name → array.
    """
    cache = {}

    def get(side: str, column: str) -> list:
        """Strings of one column for this chunk, fetched once."""
        if (side, column) not in cache:
            frame, rows = (s1, s1_rows) if side == "s1" else (tgt, tgt_rows)
            cache[(side, column)] = take_strings(frame, column, rows)
        return cache[(side, column)]

    groups = config.FEATURE_GROUPS
    cols = {}
    if groups.get("string", True):
        for field, prefix in (("name_norm", "name"), ("name_core", "name_core"),
                              ("address_norm", "addr")):
            cols.update(string_similarities(get("s1", field), get("cand", field), prefix))
    if groups.get("tokens", True):
        cols.update(token_overlaps(get("s1", "name_core"), get("cand", "name_core"), "name"))
        cols.update(token_overlaps(get("s1", "address_norm"), get("cand", "address_norm"), "addr"))
    if groups.get("numbers", True):
        cols.update(number_overlaps(get("s1", "address_numbers"), get("cand", "address_numbers")))
    if groups.get("structure", True):
        cols.update(structure_features(get))
    if groups.get("cross", True):
        cols.update(edge_word_matches(get("s1", "name_core"), get("cand", "name_core")))
    if groups.get("char_ngram", True):
        cols["name_char3_cosine"] = char_ngram_cosine(get("s1", "name_core"),
                                                      get("cand", "name_core"))
    return cols


# ──────────────────────────────────────────────────────────────────────
# Whole-candidate-set features
# ──────────────────────────────────────────────────────────────────────

# Fields whose frequency is counted: feature key → normalized column
FREQUENCY_FIELDS = {"name": "name_core", "addr": "address_norm"}


def add_frequency_columns(s1: pd.DataFrame, tgt: pd.DataFrame, verbose: bool = True) -> None:
    """
    Count how common each record's name and address are in the whole split.

    Fellegi-Sunter record linkage weights agreement by frequency: sharing a
    name that 90 S1 records have ("fresh hair studio") says little, sharing a
    unique one says a lot; the same for an office-building address that
    hosts dozens of businesses. Call this before sampling S1, so a record gets
    the same counts in training (sampled S1) and at test time (all S1).

    Adds int32 columns <key>_n_s1 (S1 records with the same value) and
    <key>_n_tgt (S2+S3 records with the same value) to both frames, in place,
    for every key of FREQUENCY_FIELDS. Empty values get 0.

    Args:
        s1: Normalized S1 frame (all records of the split).
        tgt: Normalized S2 + S3 frame.
        verbose: Whether to print timing.
    """
    t0 = time.time()
    n_s1 = len(s1)
    for key, column in FREQUENCY_FIELDS.items():
        codes, uniques = pd.factorize(pd.concat([s1[column], tgt[column]], ignore_index=True))
        codes = np.maximum(codes, 0)
        n_codes = max(len(uniques), 1)
        in_s1 = np.bincount(codes[:n_s1], minlength=n_codes).astype(np.int32)
        in_tgt = np.bincount(codes[n_s1:], minlength=n_codes).astype(np.int32)
        empty = pd.Index(uniques).get_indexer([""])[0]
        if empty >= 0:
            in_s1[empty] = in_tgt[empty] = 0
        s1[f"{key}_n_s1"], s1[f"{key}_n_tgt"] = in_s1[codes[:n_s1]], in_tgt[codes[:n_s1]]
        tgt[f"{key}_n_s1"], tgt[f"{key}_n_tgt"] = in_s1[codes[n_s1:]], in_tgt[codes[n_s1:]]
        del codes, uniques
    if verbose:
        print(f"  Name/address frequencies counted over {n_s1:,} S1 + {len(tgt):,} "
              f"S2/S3 records ({time.time() - t0:.0f}s)")


def frequency_features(s1: pd.DataFrame, tgt: pd.DataFrame, s1_idx: np.ndarray,
                       tgt_idx: np.ndarray) -> dict:
    """
    Frequency counts of both sides of every pair (see add_frequency_columns).

    Args:
        s1: S1 frame with the frequency columns.
        tgt: S2 + S3 frame with the frequency columns.
        s1_idx: S1 row per pair.
        tgt_idx: Target row per pair.

    Returns:
        Dict of feature name → array (empty if the columns are missing).
    """
    out = {}
    for key in FREQUENCY_FIELDS:
        for count in ("n_s1", "n_tgt"):
            column = f"{key}_{count}"
            if column in s1 and column in tgt:
                out[f"s1_{column}"] = s1[column].to_numpy()[s1_idx].astype(np.float32)
                out[f"cand_{column}"] = tgt[column].to_numpy()[tgt_idx].astype(np.float32)
    return out


def cross_features(cols: dict) -> dict:
    """
    Name × address interactions from the string similarities.

    Args:
        cols: Feature columns built so far (needs the "string" group).

    Returns:
        Dict of feature name → array (empty if the similarities are missing).
    """
    needed = ("name_core_token_set_ratio", "addr_token_set_ratio", "name_core_ratio", "addr_ratio")
    if not all(c in cols for c in needed):
        return {}
    name_sim, addr_sim = cols["name_core_token_set_ratio"], cols["addr_token_set_ratio"]
    name_exact = (cols["name_core_ratio"] >= 1.0).astype(np.float32)
    addr_exact = (cols["addr_ratio"] >= 1.0).astype(np.float32)
    return {
        "name_x_addr": name_sim * addr_sim,
        "name_addr_min": np.minimum(name_sim, addr_sim),
        "name_addr_max": np.maximum(name_sim, addr_sim),
        "name_exact": name_exact,
        "addr_exact": addr_exact,
        "name_exact_x_addr": name_exact * addr_sim,
        "addr_exact_x_name": addr_exact * name_sim,
    }


def group_max(groups: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Maximum of `values` within each group, broadcast back to every element.

    Args:
        groups: Non-negative integer group id per element.
        values: Values.

    Returns:
        Array of group maxima, aligned with values.
    """
    uniq, inverse = np.unique(groups, return_inverse=True)
    best = np.full(len(uniq), -np.inf, dtype=np.float64)
    np.maximum.at(best, inverse, values)
    return best[inverse].astype(np.float32)


def rank_features(s1_idx: np.ndarray, tgt_idx: np.ndarray, sims: dict) -> dict:
    """
    Compare each pair with its competitors in the candidate set.

    For every similarity in `sims`:
    - rank / gap to best among all candidates of the same S1
    - rank / gap to best among all S1s that have the same candidate
    - mutual best: best candidate for its S1 AND best S1 for its candidate
    Plus how many candidates the S1 has and how many S1s the candidate has.

    The candidate-side features (second and third bullet, S1s per candidate)
    depend on how many S1 records are loaded, so they are only built when
    config.FEATURE_GROUPS["cand_context"] is on (see config.py).

    Args:
        s1_idx: S1 row per pair.
        tgt_idx: Target row per pair.
        sims: Dict short name → similarity array (higher = more similar).

    Returns:
        Dict of feature name → array.
    """
    cand_context = config.FEATURE_GROUPS.get("cand_context", True)
    out = {}
    for key, values in sims.items():
        rank_s1 = group_rank(s1_idx, values)
        out[f"rank_{key}_in_s1"] = rank_s1.astype(np.float32)
        out[f"gap_{key}_to_s1_best"] = group_max(s1_idx, values) - values
        if cand_context:
            rank_cand = group_rank(tgt_idx, values)
            out[f"rank_{key}_in_cand"] = rank_cand.astype(np.float32)
            out[f"gap_{key}_to_cand_best"] = group_max(tgt_idx, values) - values
            out[f"mutual_best_{key}"] = ((rank_s1 == 1) & (rank_cand == 1)).astype(np.float32)
    out["n_cands_for_s1"] = np.bincount(s1_idx)[s1_idx].astype(np.float32)
    if cand_context:
        _, inverse, counts = np.unique(tgt_idx, return_inverse=True, return_counts=True)
        out["n_s1_for_cand"] = counts[inverse].astype(np.float32)
    return out


def build_feature_matrix(pairs: pd.DataFrame, s1: pd.DataFrame, tgt: pd.DataFrame,
                         embeddings: dict = None, verbose: bool = True) -> tuple:
    """
    Build the stage-3 feature matrix for the pruned candidate pairs.

    Pass the complete candidate set of a split in one call: the rank features
    compare each pair with the other pairs of the same S1 and candidate.

    Args:
        pairs: Pruned pairs: s1_idx, tgt_idx, blockers, score_*/rank_*,
            is_s2 and (optionally) pruner_prob.
        s1: Normalized S1 frame.
        tgt: Normalized S2 + S3 frame.
        embeddings: Optional {"s1": {"name", "full"}, "tgt": {"name", "full"}}
            embedding arrays aligned with the frames.
        verbose: Whether to print progress.

    Returns:
        Tuple (X float32 matrix, feature names).
    """
    groups = config.FEATURE_GROUPS
    s1_idx = pairs["s1_idx"].to_numpy()
    tgt_idx = pairs["tgt_idx"].to_numpy()

    # String-based groups, chunk by chunk
    chunks = []
    step = config.FEATURE_CHUNK_SIZE
    for start in range(0, len(pairs), step):
        end = start + step
        chunks.append(chunk_features(s1, tgt, s1_idx[start:end], tgt_idx[start:end]))
    cols = {}
    for name in list(chunks[0]) if chunks else []:
        cols[name] = np.concatenate([c.pop(name) for c in chunks]).astype(np.float32, copy=False)
    del chunks

    if groups.get("cross", True):
        cols.update(cross_features(cols))

    if groups.get("frequency", True):
        cols.update(frequency_features(s1, tgt, s1_idx, tgt_idx))

    if groups.get("tfidf", True):
        for c in pairs.columns:
            if c.startswith(("score_", "rank_")) and not c.endswith("embedding"):
                cols[f"blk_{c}"] = pairs[c].to_numpy(dtype=np.float32)

    if groups.get("structure", True):
        cols["is_s2"] = pairs["is_s2"].to_numpy(dtype=np.float32)

    if groups.get("blockers", True):
        mask = pairs["blockers"].to_numpy()
        for name, bit in BLOCKER_BITS.items():
            if config.BLOCKING_TOP_K.get(name) or (name == "embedding" and config.USE_EMBEDDINGS):
                cols[f"found_by_{name}"] = ((mask & bit) > 0).astype(np.float32)

    if groups.get("pruner", True) and "pruner_prob" in pairs:
        cols["pruner_prob"] = pairs["pruner_prob"].to_numpy(dtype=np.float32)

    if groups.get("rank", True):
        sims = {"name": pairs["score_name"].to_numpy(dtype=np.float32),
                "combined": pairs["score_combined"].to_numpy(dtype=np.float32)}
        if "pruner_prob" in pairs:
            sims["pruner"] = pairs["pruner_prob"].to_numpy(dtype=np.float32)
        cols.update(rank_features(s1_idx, tgt_idx, sims))

    if groups.get("embedding", True) and embeddings is not None:
        for field in ("name", "full"):
            cols[f"emb_{field}_cos"] = rowwise_cosine(
                embeddings["s1"][field], embeddings["tgt"][field], s1_idx, tgt_idx)
        if "score_embedding" in pairs:
            cols["blk_rank_embedding"] = pairs["rank_embedding"].to_numpy(dtype=np.float32)

    # Fill the matrix one column at a time, freeing each column as it is copied,
    # so the columns and the matrix (~8.6M test pairs x 91) are not both held
    names = list(cols)
    X = np.empty((len(pairs), len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        X[:, j] = cols.pop(name)
    if verbose:
        print(f"  Stage-3 features: {len(pairs):,} pairs × {len(names)}")
    return X, names
