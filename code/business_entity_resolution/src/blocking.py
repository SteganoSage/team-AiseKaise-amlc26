"""
Blocking module for candidate generation.

Determines the recall ceiling of the pipeline. Uses a union of cheap blockers
to generate candidate (S1, S2/S3) pairs:

1. TF-IDF char n-grams (3-4) on name_core → nearest neighbours
2. TF-IDF on name+address combined
3. Exact postal-code / first-token keys
4. (Optional) Multilingual sentence-embedding kNN

Blocks within the same country when country is present; falls back to global
if country is missing.

Target: blocking pair recall ≥ 0.97 with manageable candidate count.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config


def build_tfidf_index(texts: list, analyzer: str = "char_wb",
                      ngram_range: tuple = None) -> tuple:
    """
    Build a TF-IDF index from a list of texts.

    Args:
        texts: List of text strings to index.
        analyzer: 'char_wb' for character n-grams, 'word' for word tokens.
        ngram_range: Tuple (min_n, max_n) for n-gram range. Defaults to config.

    Returns:
        Tuple of (TfidfVectorizer, sparse matrix of TF-IDF vectors).
    """
    if ngram_range is None:
        ngram_range = config.TFIDF_NGRAM_RANGE

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


def tfidf_top_k(query_matrix, index_matrix, k: int) -> list:
    """
    Find top-K nearest neighbours for each query by cosine similarity.

    Works for sparse TF-IDF matrices and dense embedding arrays alike.

    Processes in batches to manage memory.

    Args:
        query_matrix: Sparse TF-IDF matrix for query records (S1).
        index_matrix: Sparse TF-IDF matrix for index records (S2 or S3).
        k: Number of top candidates to return per query.

    Returns:
        List of lists: for each query, a list of (index, score) tuples,
        sorted by descending score.
    """
    batch_size = 500
    n_queries = query_matrix.shape[0]
    results = []

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch_scores = cosine_similarity(query_matrix[start:end], index_matrix)

        for i in range(batch_scores.shape[0]):
            scores = batch_scores[i]
            # Get top-K indices
            if len(scores) <= k:
                top_indices = np.argsort(scores)[::-1]
            else:
                # Partial sort for efficiency
                top_indices = np.argpartition(scores, -k)[-k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            results.append(
                [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
            )

    return results


def exact_key_blocking(s1_records: list, target_records: list,
                       key_fn, max_block: int = None) -> dict:
    """
    Block by exact key match (e.g., postal code, first name token).

    Keys shared by more than `max_block` target records are dropped: a busy
    postal code or a common first word ("sri", "the") would otherwise add
    thousands of weak candidates per S1 record.

    Args:
        s1_records: List of normalized S1 record dicts.
        target_records: List of normalized S2 or S3 record dicts.
        key_fn: Function that takes a record dict and returns a set of keys.
        max_block: Max target records per key. Defaults to
            config.BLOCKING_EXACT_KEY_MAX_BLOCK.

    Returns:
        Dict mapping S1 entity_id → set of target entity_ids sharing a key.
    """
    if max_block is None:
        max_block = config.BLOCKING_EXACT_KEY_MAX_BLOCK

    # Build inverted index on target records
    key_to_targets = {}
    for rec in target_records:
        keys = key_fn(rec)
        for k in keys:
            if k:
                key_to_targets.setdefault(k, set()).add(rec["entity_id"])

    # Drop unspecific keys
    key_to_targets = {
        k: v for k, v in key_to_targets.items() if len(v) <= max_block
    }

    # Look up S1 records
    candidates = {}
    for rec in s1_records:
        keys = key_fn(rec)
        matched = set()
        for k in keys:
            if k and k in key_to_targets:
                matched.update(key_to_targets[k])
        if matched:
            candidates[rec["entity_id"]] = matched

    return candidates


def postal_code_key(record: dict) -> set:
    """
    Extract postal code keys from a record for exact blocking.

    Args:
        record: Normalized record dict.

    Returns:
        Set of postal code strings.
    """
    return set(record.get("postal_codes", []))


def first_token_key(record: dict) -> set:
    """
    Extract the first token of the normalized name as a blocking key.

    Args:
        record: Normalized record dict.

    Returns:
        Set containing the first name token (or empty set).
    """
    tokens = record.get("name_core", "").split()
    if tokens and len(tokens[0]) >= config.BLOCKING_FIRST_TOKEN_MIN_LEN:
        return {tokens[0]}
    return set()


def country_partitions(s1_records: list, target_records: list,
                       target_name: str = "target",
                       verbose: bool = True) -> list:
    """
    Split S1 and target records into same-country blocks.

    Country is treated as an open set of strings (test adds France), so no
    country value is hard-coded. Fallbacks keep recall safe:
    - S1 records with an empty country search all target records.
    - An S1 country string that no target record uses (e.g. "india" vs "in")
      searches all target records.
    - Target records with an empty country are included in every block.

    Args:
        s1_records: List of normalized S1 record dicts.
        target_records: List of normalized S2 or S3 record dicts.
        target_name: Label for log messages ("S2" / "S3").
        verbose: Whether to print the block sizes.

    Returns:
        List of (s1_indices, target_indices) numpy array pairs, indexing into
        s1_records and target_records.
    """
    all_targets = np.arange(len(target_records))
    if not config.BLOCK_BY_COUNTRY:
        return [(np.arange(len(s1_records)), all_targets)]

    target_by_country = {}
    no_country_targets = []
    for j, rec in enumerate(target_records):
        country = rec.get("country_norm", "")
        if country:
            target_by_country.setdefault(country, []).append(j)
        else:
            no_country_targets.append(j)

    s1_by_country = {}
    for i, rec in enumerate(s1_records):
        s1_by_country.setdefault(rec.get("country_norm", ""), []).append(i)

    partitions = []
    for country, s1_idx in sorted(s1_by_country.items()):
        if country and country in target_by_country:
            tgt_idx = np.array(sorted(target_by_country[country] + no_country_targets))
            note = ""
        else:
            tgt_idx = all_targets
            note = "  (no country)" if not country else \
                f"  ⚠ not found in {target_name} → searching all countries"
        partitions.append((np.array(s1_idx), tgt_idx))
        if verbose:
            print(f"      country '{country}': {len(s1_idx)} S1 vs "
                  f"{len(tgt_idx)} {target_name}{note}")

    return partitions


# Bit flags recording which blocker(s) produced a candidate pair
BLOCKER_BITS = {
    "tfidf_name": 1,
    "tfidf_combined": 2,
    "postal_code": 4,
    "first_token": 8,
    "embedding": 16,
}


def generate_candidates(s1_records: list, s2_records: list, s3_records: list,
                        embeddings: dict = None, verbose: bool = True) -> tuple:
    """
    Generate candidate pairs for all S1 entities against S2 and S3.

    Uses a union of multiple blocking strategies:
    1. TF-IDF char n-grams on name_core
    2. TF-IDF word tokens on name+address combined
    3. Exact postal code matching
    4. First name token matching
    5. Embedding kNN on name+address (only if `embeddings` is given)

    Blocks within the same country when possible (see country_partitions).
    TF-IDF vocabularies are fitted once per source on all records, then
    nearest neighbours are searched inside each country block.

    Args:
        s1_records: List of normalized S1 record dicts.
        s2_records: List of normalized S2 record dicts.
        s3_records: List of normalized S3 record dicts.
        embeddings: Output of embeddings.embed_records for this split, or None.
        verbose: Whether to print progress information.

    Returns:
        Tuple of:
        - candidates: dict S1 entity_id → sorted list of candidate S2/S3 IDs
        - sources: dict S1 entity_id → {candidate ID: BLOCKER_BITS bitmask}
    """
    sources = {rec["entity_id"]: {} for rec in s1_records}

    def add(s1_id: str, cand_id: str, blocker: str) -> None:
        """Record that `blocker` proposed (s1_id, cand_id)."""
        found = sources[s1_id]
        found[cand_id] = found.get(cand_id, 0) | BLOCKER_BITS[blocker]

    if embeddings is not None:
        from embeddings import rows_for
        s1_emb = embeddings["full"][rows_for(embeddings, [r["entity_id"] for r in s1_records])]

    # Process S2 and S3 separately
    for source_name, target_records in [("S2", s2_records), ("S3", s3_records)]:
        if not target_records:
            continue

        if verbose:
            print(f"\n  Blocking S1 vs {source_name} ({len(target_records)} records)...")

        target_id_list = [rec["entity_id"] for rec in target_records]

        # TF-IDF on name_core (char n-grams) and name+address (word 1-2 grams)
        s1_names = [rec["name_core"] for rec in s1_records]
        target_names = [rec["name_core"] for rec in target_records]
        _, name_tfidf = build_tfidf_index(
            s1_names + target_names, analyzer="char_wb"
        )
        s1_name_matrix = name_tfidf[: len(s1_names)]
        target_name_matrix = name_tfidf[len(s1_names):]

        s1_combined = [
            rec["name_norm"] + " " + rec["address_norm"] for rec in s1_records
        ]
        target_combined = [
            rec["name_norm"] + " " + rec["address_norm"] for rec in target_records
        ]
        _, combined_tfidf = build_tfidf_index(
            s1_combined + target_combined, analyzer="word", ngram_range=(1, 2)
        )
        s1_comb_matrix = combined_tfidf[: len(s1_combined)]
        target_comb_matrix = combined_tfidf[len(s1_combined):]

        if embeddings is not None:
            target_emb = embeddings["full"][rows_for(embeddings, target_id_list)]

        partitions = country_partitions(
            s1_records, target_records, target_name=source_name, verbose=verbose
        )

        for s1_idx, tgt_idx in partitions:
            block_s1 = [s1_records[i] for i in s1_idx]
            block_targets = [target_records[j] for j in tgt_idx]

            # --- Blockers 1, 2 (and 5): nearest neighbours ---
            knn_blockers = [
                ("tfidf_name", s1_name_matrix, target_name_matrix,
                 config.BLOCKING_TOP_K_TFIDF_NAME),
                ("tfidf_combined", s1_comb_matrix, target_comb_matrix,
                 config.BLOCKING_TOP_K_TFIDF_COMBINED),
            ]
            if embeddings is not None:
                knn_blockers.append(
                    ("embedding", s1_emb, target_emb, config.BLOCKING_TOP_K_EMBEDDING)
                )
            for blocker, s1_matrix, target_matrix, k in knn_blockers:
                results = tfidf_top_k(s1_matrix[s1_idx], target_matrix[tgt_idx], k)
                for local_i, rec in enumerate(block_s1):
                    for local_j, _ in results[local_i]:
                        add(rec["entity_id"], target_id_list[tgt_idx[local_j]], blocker)

            # --- Blockers 3, 4: exact keys ---
            for blocker, key_fn in (("postal_code", postal_code_key),
                                    ("first_token", first_token_key)):
                key_candidates = exact_key_blocking(block_s1, block_targets, key_fn)
                for s1_id, cands in key_candidates.items():
                    for cand_id in cands:
                        add(s1_id, cand_id, blocker)

    candidates = {s1_id: sorted(found) for s1_id, found in sources.items()}

    if verbose:
        n_targets = len(s2_records) + len(s3_records)
        print_blocking_stats(candidates, n_targets)

    return candidates, sources


def print_blocking_stats(candidates: dict, n_targets: int) -> float:
    """
    Print candidate-set size statistics and the reduction ratio.

    Reduction ratio = 1 - (candidate pairs / all possible S1 x (S2+S3) pairs).
    Candidate_pairs.tsv is audited on this, so it goes in the documentation.

    Args:
        candidates: Dict mapping S1 entity_id → list of candidate IDs.
        n_targets: Total number of S2 + S3 records.

    Returns:
        Reduction ratio in [0, 1].
    """
    sizes = np.array([len(v) for v in candidates.values()]) if candidates else np.zeros(1)
    total_pairs = int(sizes.sum())
    all_pairs = max(len(candidates) * n_targets, 1)
    reduction_ratio = 1.0 - total_pairs / all_pairs

    print(f"\n  Total candidate pairs: {total_pairs:,}")
    print(f"  Candidates per S1: mean {sizes.mean():.1f}, "
          f"median {np.median(sizes):.0f}, max {sizes.max()}")
    print(f"  Reduction ratio: {reduction_ratio:.4f}")

    return reduction_ratio


def evaluate_blocking_recall(candidates: dict, ground_truth: dict,
                             sources: dict = None,
                             verbose: bool = True) -> float:
    """
    Compute blocking recall: fraction of true pairs retained in candidates.

    With `sources`, also prints each blocker's own recall and how many true
    pairs only that blocker found (a blocker with ~0 "only" pairs adds
    candidates without adding recall).

    Args:
        candidates: Dict mapping S1 entity_id → list of candidate IDs.
        ground_truth: Dict mapping S1 entity_id → list of true match IDs.
        sources: Optional dict S1 entity_id → {candidate ID: bitmask}.
        verbose: Whether to print results.

    Returns:
        Blocking recall as a float in [0, 1].
    """
    total_true_pairs = 0
    retained_pairs = 0
    found_by = {name: 0 for name in BLOCKER_BITS}
    only_by = {name: 0 for name in BLOCKER_BITS}

    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        cand_set = set(candidates.get(s1_id, []))
        for match_id in true_matches:
            total_true_pairs += 1
            if match_id in cand_set:
                retained_pairs += 1
            if sources is not None:
                mask = sources.get(s1_id, {}).get(match_id, 0)
                for name, bit in BLOCKER_BITS.items():
                    if mask & bit:
                        found_by[name] += 1
                        if mask == bit:
                            only_by[name] += 1

    recall = retained_pairs / max(total_true_pairs, 1)

    if verbose:
        print(f"\n  Blocking recall: {recall:.4f} "
              f"({retained_pairs}/{total_true_pairs} true pairs retained)")
        if sources is not None and total_true_pairs:
            print(f"  {'blocker':>16s}  {'recall':>7s}  {'only this':>9s}")
            for name in BLOCKER_BITS:
                if found_by[name] or name != "embedding":
                    print(f"  {name:>16s}  {found_by[name] / total_true_pairs:>7.4f}"
                          f"  {only_by[name]:>9d}")

    return recall
