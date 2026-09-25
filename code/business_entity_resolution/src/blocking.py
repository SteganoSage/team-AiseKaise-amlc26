"""
Stage 1 — blocking (candidate generation), built for ~2M S1 x ~5M S2/S3.

Determines the recall ceiling of the pipeline. For each S1 record and each
source (S2, S3) separately, inside the same country:

1. "name" blocker:     TF-IDF on word 1-2 grams of name_core → top-K
2. "combined" blocker: TF-IDF on word 1-2 grams of name + address → top-K
3. (optional) "embedding" blocker: multilingual sentence-embedding kNN (GPU)

How it scales:
- HashingVectorizer: no vocabulary to hold in memory (word pairs over
  millions of records would be tens of millions of entries).
- Terms that appear in more than config.BLOCKING_MAX_DF target records are
  dropped. They carry almost no identity ("limited", "road", "delhi") and are
  what makes an all-vs-all product explode. Word pairs ("newton road",
  "prime realty") are rare, so records made of common words still match.
- The S1 x target cosine is a sparse matrix product computed in row chunks
  across worker processes, keeping only the top-K per S1 row.

Every union pair also gets the cosine of every blocker (not only the one
that found it), so the scores double as features for stages 2 and 3.

Blocks within the same country string when present; falls back to all
records when the country is missing or unknown in the target source.
"""

import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize

import config

# Bit flags recording which blocker(s) produced a candidate pair
BLOCKER_BITS = {"name": 1, "combined": 2, "embedding": 4}

# Worker-process globals for the sparse top-K (inherited through fork)
_SHARED = {}


# ──────────────────────────────────────────────────────────────────────
# Text → TF-IDF
# ──────────────────────────────────────────────────────────────────────

def blocker_texts(frame: pd.DataFrame, blocker: str) -> list:
    """
    Build the text each blocker indexes, for every record of a frame.

    Args:
        frame: Normalized records (normalize.normalize_frame output).
        blocker: "name" or "combined".

    Returns:
        List of strings, one per record.
    """
    if blocker == "name":
        return frame["name_core"].tolist()
    if blocker == "combined":
        return (frame["name_norm"] + " " + frame["address_norm"]).tolist()
    raise ValueError(f"Unknown blocker {blocker}")


def _hash_chunk(texts: list) -> sp.csr_matrix:
    """
    Hash word 1-2 grams of a chunk of texts into term counts (worker task).

    Args:
        texts: List of normalized strings.

    Returns:
        CSR count matrix of shape (len(texts), config.HASH_FEATURES).
    """
    vectorizer = HashingVectorizer(
        analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
        lowercase=False, n_features=config.HASH_FEATURES,
        alternate_sign=False, norm=None, dtype=np.float32,
    )
    return vectorizer.transform(texts)


def _pool(n_workers: int):
    """
    Create a worker pool (fork where available, so large globals are shared).

    Args:
        n_workers: Number of processes.

    Returns:
        multiprocessing Pool.
    """
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    return mp.get_context(method).Pool(n_workers)


def hash_texts(texts: list, n_jobs: int = 1, chunk_size: int = 200_000) -> sp.csr_matrix:
    """
    Hash many texts in parallel.

    Args:
        texts: List of strings.
        n_jobs: Worker processes.
        chunk_size: Texts per task.

    Returns:
        CSR count matrix.
    """
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    if not chunks:
        return sp.csr_matrix((0, config.HASH_FEATURES), dtype=np.float32)
    if n_jobs > 1 and len(chunks) > 1:
        with _pool(min(n_jobs, len(chunks))) as pool:
            parts = pool.map(_hash_chunk, chunks)
    else:
        parts = [_hash_chunk(c) for c in chunks]
    return sp.vstack(parts).tocsr()


def tfidf_pair(query_texts: list, target_texts: list, n_jobs: int = 1) -> tuple:
    """
    TF-IDF matrices for queries (S1) and targets (one source), L2-normalized.

    IDF comes from the targets. Terms absent from the targets or present in
    more than config.BLOCKING_MAX_DF of them get weight 0 (dropped).

    Args:
        query_texts: S1 texts.
        target_texts: S2 or S3 texts.
        n_jobs: Worker processes.

    Returns:
        Tuple (Q, T) of CSR matrices whose row dot products are cosines.
    """
    Q = hash_texts(query_texts, n_jobs)
    T = hash_texts(target_texts, n_jobs)
    doc_freq = np.bincount(T.indices, minlength=T.shape[1])
    idf = np.log((1.0 + T.shape[0]) / (1.0 + doc_freq)) + 1.0
    weight = np.where((doc_freq > 0) & (doc_freq <= config.BLOCKING_MAX_DF), idf, 0.0)
    weight = weight.astype(np.float32)
    for M in (Q, T):
        M.data = (1.0 + np.log(M.data)) * weight[M.indices]  # sublinear tf x idf
        M.eliminate_zeros()
    return l2_normalize(Q, copy=False), l2_normalize(T, copy=False)


def rowwise_cosine(Q, T, q_idx: np.ndarray, t_idx: np.ndarray,
                   chunk: int = 1_000_000) -> np.ndarray:
    """
    Cosine of row q_idx[i] of Q with row t_idx[i] of T, for every i.

    Works for sparse (TF-IDF) and dense (embedding) matrices.

    Args:
        Q: Query matrix (rows L2-normalized).
        T: Target matrix (rows L2-normalized).
        q_idx: Query row per pair.
        t_idx: Target row per pair.
        chunk: Pairs per chunk.

    Returns:
        float32 array of cosines.
    """
    out = np.empty(len(q_idx), dtype=np.float32)
    for start in range(0, len(q_idx), chunk):
        end = start + chunk
        if sp.issparse(Q):
            prod = Q[q_idx[start:end]].multiply(T[t_idx[start:end]])
            out[start:end] = np.asarray(prod.sum(axis=1)).ravel()
        else:  # dense embeddings (stored as float16) — accumulate in float32
            out[start:end] = np.einsum("ij,ij->i",
                                       Q[q_idx[start:end]].astype(np.float32),
                                       T[t_idx[start:end]].astype(np.float32))
    return out


# ──────────────────────────────────────────────────────────────────────
# Top-K search
# ──────────────────────────────────────────────────────────────────────

def _empty_topk() -> tuple:
    """Empty (query_row, target_row, score, rank) result."""
    return (np.empty(0, np.int32), np.empty(0, np.int32),
            np.empty(0, np.float32), np.empty(0, np.int16))


def _topk_rows(start: int, end: int) -> tuple:
    """
    Top-K targets for query rows [start, end) of the shared matrices (worker task).

    Args:
        start: First query row.
        end: One past the last query row.

    Returns:
        Tuple of arrays (query_row, target_col, score, rank), rank starting at 1.
    """
    Q, Tt, k = _SHARED["Q"], _SHARED["Tt"], _SHARED["k"]
    C = (Q[start:end] @ Tt).tocsr()
    rows, cols, vals, ranks = [], [], [], []
    indptr, indices, data = C.indptr, C.indices, C.data
    for i in range(end - start):
        s, e = indptr[i], indptr[i + 1]
        if s == e:
            continue
        seg = data[s:e]
        part = np.argpartition(seg, -k)[-k:] if e - s > k else np.arange(e - s)
        part = part[np.argsort(-seg[part], kind="stable")]
        rows.append(np.full(len(part), start + i, dtype=np.int32))
        cols.append(indices[s:e][part].astype(np.int32))
        vals.append(seg[part].astype(np.float32))
        ranks.append(np.arange(1, len(part) + 1, dtype=np.int16))
    if not rows:
        return _empty_topk()
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), np.concatenate(ranks)


def sparse_topk(Q, T, k: int, n_jobs: int = 1) -> tuple:
    """
    For every row of Q, the k rows of T with the highest cosine (> 0).

    Args:
        Q: (n_queries, F) CSR, rows L2-normalized.
        T: (n_targets, F) CSR, rows L2-normalized.
        k: Neighbours per query.
        n_jobs: Worker processes.

    Returns:
        Tuple of arrays (query_row, target_row, score, rank).
    """
    _SHARED.update(Q=Q, Tt=T.T.tocsr(), k=k)
    step = config.BLOCKING_CHUNK_ROWS
    bounds = [(s, min(s + step, Q.shape[0])) for s in range(0, Q.shape[0], step)]
    try:
        if n_jobs > 1 and len(bounds) > 1:
            with _pool(min(n_jobs, len(bounds))) as pool:
                parts = pool.starmap(_topk_rows, bounds)
        else:
            parts = [_topk_rows(s, e) for s, e in bounds]
    finally:
        _SHARED.clear()
    if not parts:
        return _empty_topk()
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def dense_topk(Qe: np.ndarray, Te: np.ndarray, k: int) -> tuple:
    """
    Top-K by embedding cosine, on GPU when available (torch).

    Args:
        Qe: (n_queries, d) L2-normalized embeddings.
        Te: (n_targets, d) L2-normalized embeddings.
        k: Neighbours per query.

    Returns:
        Tuple of arrays (query_row, target_row, score, rank).
    """
    import torch
    if len(Qe) == 0 or len(Te) == 0:
        return _empty_topk()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    T = torch.from_numpy(Te).to(device=device, dtype=dtype)
    k = min(k, Te.shape[0])
    rows, cols, vals = [], [], []
    step = 256 if device == "cuda" else 64
    for start in range(0, Qe.shape[0], step):
        q = torch.from_numpy(Qe[start:start + step]).to(device=device, dtype=dtype)
        scores, idx = torch.topk(q @ T.T, k, dim=1)
        rows.append(np.repeat(np.arange(start, start + q.shape[0], dtype=np.int32), k))
        cols.append(idx.cpu().numpy().astype(np.int32).ravel())
        vals.append(scores.float().cpu().numpy().ravel())
    ranks = np.tile(np.arange(1, k + 1, dtype=np.int16), Qe.shape[0])
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), ranks


# ──────────────────────────────────────────────────────────────────────
# Candidate generation
# ──────────────────────────────────────────────────────────────────────

def country_partitions(s1_country: np.ndarray, tgt_country: np.ndarray,
                       target_name: str = "target", verbose: bool = True) -> list:
    """
    Split S1 and target rows into same-country blocks.

    Country is treated as an open set of strings (test adds France), so no
    country value is hard-coded. Fallbacks keep recall safe:
    - S1 rows with an empty country search all target rows.
    - An S1 country string that no target row uses (e.g. "india" vs "in")
      searches all target rows.
    - Target rows with an empty country are included in every block.

    Args:
        s1_country: Normalized country per S1 row.
        tgt_country: Normalized country per target row.
        target_name: Label for log messages ("S2" / "S3").
        verbose: Whether to print the block sizes.

    Returns:
        List of (s1_rows, target_rows) integer array pairs.
    """
    all_targets = np.arange(len(tgt_country))
    if not config.BLOCK_BY_COUNTRY:
        return [(np.arange(len(s1_country)), all_targets)]

    tgt_groups = pd.Series(tgt_country).groupby(tgt_country, sort=True).indices
    no_country = tgt_groups.get("", np.empty(0, dtype=np.int64))
    s1_groups = pd.Series(s1_country).groupby(s1_country, sort=True).indices

    partitions = []
    for country, s1_rows in s1_groups.items():
        if country and country in tgt_groups:
            tgt_rows = np.sort(np.concatenate([tgt_groups[country], no_country]))
            note = ""
        else:
            tgt_rows = all_targets
            note = "  (no country)" if not country else \
                f"  ⚠ not found in {target_name} → searching all countries"
        partitions.append((np.asarray(s1_rows), tgt_rows))
        if verbose:
            print(f"      country '{country}': {len(s1_rows):,} S1 vs "
                  f"{len(tgt_rows):,} {target_name}{note}")
    return partitions


def _block_one_source(s1: pd.DataFrame, tgt: pd.DataFrame, source: str,
                      embeddings: dict, verbose: bool) -> pd.DataFrame:
    """
    All blockers for S1 vs one target source, merged into one row per pair.

    Args:
        s1: Normalized S1 frame.
        tgt: Normalized rows of one source (S2 or S3), with a "tgt_row"
            column holding their row number in the combined target frame.
        source: "S2" or "S3".
        embeddings: Optional dict with "s1" and "tgt" embedding arrays
            (rows aligned with s1 and the combined target frame).
        verbose: Whether to print progress.

    Returns:
        DataFrame with s1_idx, tgt_idx, blockers bitmask, and score_<b> /
        rank_<b> for every blocker b (rank = K + 1 when b did not propose it).
    """
    n_jobs = config.N_JOBS
    s1_country = s1["country_norm"].to_numpy(dtype=object)
    tgt_country = tgt["country_norm"].to_numpy(dtype=object)
    partitions = country_partitions(s1_country, tgt_country, source, verbose)

    matrices, found = {}, []
    for blocker, k in config.BLOCKING_TOP_K.items():
        t0 = time.time()
        Q, T = tfidf_pair(blocker_texts(s1, blocker), blocker_texts(tgt, blocker), n_jobs)
        t1 = time.time()
        matrices[blocker] = (Q, T, k)
        for s1_rows, tgt_rows in partitions:
            q, t, _, _ = sparse_topk(Q[s1_rows], T[tgt_rows], k, n_jobs)
            found.append((blocker, s1_rows[q], tgt_rows[t]))
        if verbose:
            # Terms kept per record = what survives BLOCKING_MAX_DF (drives search cost and recall)
            print(f"      {blocker:>9s}: tf-idf {t1 - t0:5.0f}s, search {time.time() - t1:5.0f}s | "
                  f"terms kept per S1 {Q.nnz / max(Q.shape[0], 1):.1f}, "
                  f"per {source} {T.nnz / max(T.shape[0], 1):.1f}")

    if embeddings is not None:
        t0 = time.time()
        e_s1 = embeddings["s1"]
        e_tgt = embeddings["tgt"][tgt["tgt_row"].to_numpy()]
        k = config.BLOCKING_TOP_K_EMBEDDING
        matrices["embedding"] = (e_s1, e_tgt, k)
        for s1_rows, tgt_rows in partitions:
            q, t, _, _ = dense_topk(e_s1[s1_rows], e_tgt[tgt_rows], k)
            found.append(("embedding", s1_rows[q], tgt_rows[t]))
        if verbose:
            print(f"      embedding: search {time.time() - t0:5.0f}s")
    t_union = time.time()

    # Union of all blockers: one row per (S1 row, local target row)
    n_local = len(tgt)
    if found:
        keys = np.concatenate([s.astype(np.int64) * n_local + t for _, s, t in found])
        bits = np.concatenate([np.full(len(s), BLOCKER_BITS[b], np.int8) for b, s, _ in found])
    else:
        keys, bits = np.empty(0, np.int64), np.empty(0, np.int8)
    uniq, inverse = np.unique(keys, return_inverse=True)
    mask = np.zeros(len(uniq), dtype=np.int8)
    np.bitwise_or.at(mask, inverse, bits)
    s1_idx = (uniq // n_local).astype(np.int32)
    local_t = (uniq % n_local).astype(np.int32)

    out = pd.DataFrame({
        "s1_idx": s1_idx,
        "tgt_idx": tgt["tgt_row"].to_numpy()[local_t].astype(np.int32),
        "blockers": mask,
    })
    # Every blocker's cosine and within-(S1, source) rank for every union pair
    for blocker, (Q, T, k) in matrices.items():
        score = rowwise_cosine(Q, T, s1_idx, local_t)
        out[f"score_{blocker}"] = score
        out[f"rank_{blocker}"] = group_rank(s1_idx, score, cap=k + 1)
    if verbose:
        print(f"      union + scores for {len(out):,} pairs: {time.time() - t_union:.0f}s")
    return out


def group_rank(groups: np.ndarray, values: np.ndarray, cap: int = None) -> np.ndarray:
    """
    Rank of each value within its group, highest value = 1.

    Args:
        groups: Group id per element (e.g. s1_idx).
        values: Values to rank (higher is better).
        cap: Optional maximum rank (larger ranks are clipped to it).

    Returns:
        int16/int32 rank per element.
    """
    if len(values) == 0:
        return np.empty(0, dtype=np.int16)
    order = np.lexsort((-values, groups))
    g = groups[order]
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    group_start = np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    rank = np.empty(len(order), dtype=np.int32)
    rank[order] = np.arange(len(order)) - group_start + 1
    if cap is not None:
        rank = np.minimum(rank, cap).astype(np.int16)
    return rank


def generate_candidates(s1: pd.DataFrame, tgt: pd.DataFrame,
                        embeddings: dict = None, verbose: bool = True) -> pd.DataFrame:
    """
    Stage 1: candidate pairs for every S1 record against S2 and S3.

    Args:
        s1: Normalized S1 frame (row i = S1 index i).
        tgt: Normalized S2 + S3 frame (row j = target index j) with a
            "source" column ("S2"/"S3").
        embeddings: Optional {"s1": array, "tgt": array} of embeddings.
        verbose: Whether to print progress.

    Returns:
        DataFrame, one row per candidate pair: s1_idx, tgt_idx, blockers,
        score_<blocker>, rank_<blocker>, is_s2. Sorted by (s1_idx, tgt_idx).
    """
    parts = []
    for source in ("S2", "S3"):
        rows = np.flatnonzero(tgt["source"].to_numpy() == source)
        if len(rows) == 0:
            continue
        if verbose:
            print(f"\n  Blocking S1 vs {source} ({len(rows):,} records)...")
        part_tgt = tgt.iloc[rows].assign(tgt_row=rows)
        part = _block_one_source(s1, part_tgt, source, embeddings, verbose)
        part["is_s2"] = np.int8(source == "S2")
        parts.append(part)

    pairs = pd.concat(parts, ignore_index=True)
    return pairs.sort_values(["s1_idx", "tgt_idx"], kind="stable").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────

def print_candidate_stats(pairs: pd.DataFrame, n_s1: int, n_targets: int,
                          stage: str) -> dict:
    """
    Print candidate-set size statistics and the reduction ratio.

    Reduction ratio = 1 - (candidate pairs / all possible S1 x (S2+S3) pairs).
    candidate_pairs.tsv is audited on this, and smaller candidate sets per
    S1 rank higher, so these numbers go in every experiment report.

    Args:
        pairs: Candidate pairs with an s1_idx column.
        n_s1: Number of S1 records considered.
        n_targets: Total number of S2 + S3 records.
        stage: Label for the printout.

    Returns:
        Dict with total, mean, median, max candidates per S1 and reduction ratio.
    """
    sizes = np.bincount(pairs["s1_idx"].to_numpy(), minlength=n_s1) if len(pairs) \
        else np.zeros(max(n_s1, 1), dtype=int)
    stats = {
        "total_pairs": int(len(pairs)),
        "mean_per_s1": float(sizes.mean()),
        "median_per_s1": float(np.median(sizes)),
        "max_per_s1": int(sizes.max()),
        "s1_without_candidates": int((sizes == 0).sum()),
        "reduction_ratio": 1.0 - len(pairs) / max(n_s1 * n_targets, 1),
    }
    print(f"\n  {stage}: {stats['total_pairs']:,} pairs | per S1: mean "
          f"{stats['mean_per_s1']:.2f}, median {stats['median_per_s1']:.0f}, "
          f"max {stats['max_per_s1']} | S1 with none: {stats['s1_without_candidates']:,} "
          f"| reduction ratio {stats['reduction_ratio']:.6f}")
    return stats


def report_recall(pairs: pd.DataFrame, labels: np.ndarray, n_true_links: int,
                  stage: str, verbose: bool = True) -> float:
    """
    Pair recall of a candidate set: true pairs kept / all true pairs.

    Also prints each blocker's own recall and how many true pairs only that
    blocker found (a blocker with ~0 "only" pairs adds candidates, not recall).

    Args:
        pairs: Candidate pairs with a blockers bitmask column.
        labels: 1 for true pairs, aligned with pairs.
        n_true_links: All true links of the S1 entities considered.
        stage: Label for the printout.
        verbose: Whether to print.

    Returns:
        Recall in [0, 1].
    """
    kept = int(labels.sum())
    recall = kept / max(n_true_links, 1)
    if verbose:
        print(f"\n  {stage} recall: {recall:.4f} ({kept:,}/{n_true_links:,} true pairs)")
        if "blockers" in pairs and n_true_links:
            mask = pairs["blockers"].to_numpy()[labels == 1]
            print(f"  {'blocker':>12s}  {'recall':>7s}  {'only this':>9s}")
            for name, bit in BLOCKER_BITS.items():
                hit = (mask & bit) > 0
                if hit.any() or name != "embedding":
                    print(f"  {name:>12s}  {hit.sum() / n_true_links:>7.4f}  "
                          f"{int((mask == bit).sum()):>9d}")
    return recall
