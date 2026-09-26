"""Stage 2: candidate generation (blocking).

Per country (training matches never cross countries), several TF-IDF views
retrieve, for every Source 1 entity, its top-K most similar Source 2/3 records
by cosine similarity, using a multithreaded sparse top-k product. The union of
all views is the candidate set; each view's cosine and rank are kept because
they are also strong matching features.

Views:
  name3   char 3-grams of the core name with spaces removed (typos, domains)
  skel3   char 3-grams of the consonant skeleton (transliterated names)
  words   word tokens of core name + address + state (the main view)
  addr    word tokens of the address only (records renamed to a random brand)

Very frequent features are dropped (max_df) to bound the cost of the product;
they carry little identifying information anyway. The cost of the product is
roughly sum over query features of their posting-list length, so max_df is
the main speed knob (50k -> 5k was ~25x faster on India).
"""
import time

import numpy as np
import polars as pl
from sparse_dot_topn import sp_matmul_topn

from .config import THREADS
from .tfidf import tfidf_pair

VIEWS = {
    "name3": dict(analyzer="char", ngram_range=(3, 3), max_df=5_000),
    "name4": dict(analyzer="char", ngram_range=(4, 4), max_df=5_000),
    "skel3": dict(analyzer="char", ngram_range=(3, 3), max_df=5_000),
    "words": dict(analyzer="word", token_pattern=r"\S+", max_df=5_000),
    "addr": dict(analyzer="word", token_pattern=r"\S+", max_df=5_000),
    # Word pairs: "173 orchard", "dental preferred" stay rare even when each
    # word is common, so they survive the frequency cap. Name tokens are
    # sorted first so their pairs do not depend on word order.
    "words2": dict(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), max_df=5_000),
    "addr2": dict(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), max_df=5_000),
    # Character 4-grams over name + address: tolerant of typos inside words
    # ("brrton st", "saint pual st", "asswaoman") that break the word views.
    "comb4": dict(analyzer="char", ngram_range=(4, 4), max_df=3_000),
}


def view_cfg(view):
    """Vectoriser settings of a view, without max_df."""
    return {k: v for k, v in VIEWS[view].items() if k != "max_df"}


def view_text(df, view):
    """The string each view vectorises, as a polars expression result.

    Name views use `n_core_tr` (core name with transliterated words mapped back
    to Latin, see ber.translit) when the frame has it, else `n_core`.
    """
    core = pl.col("n_core_tr" if "n_core_tr" in df.columns else "n_core")
    if view in ("name3", "name4"):
        expr = core.str.replace_all(" ", "")
    elif view == "skel3":
        expr = pl.col("n_skel").str.replace_all(" ", "")
    elif view == "words":
        expr = pl.concat_str(
            [core, pl.col("a_norm"),
             pl.when(pl.col("a_state") != "").then("st_" + pl.col("a_state")).otherwise(pl.lit(""))],
            separator=" ")
    elif view == "words2":
        name_sorted = core.str.split(" ").list.sort().list.join(" ")
        expr = pl.concat_str(
            [name_sorted, pl.col("a_norm"),
             pl.when(pl.col("a_state") != "").then("st_" + pl.col("a_state")).otherwise(pl.lit(""))],
            separator=" ")
    elif view == "comb4":
        name_sorted = core.str.split(" ").list.sort().list.join(" ")
        expr = pl.concat_str([name_sorted, pl.col("a_norm")], separator=" ")
    elif view in ("addr", "addr2"):
        expr = pl.col("a_norm")
    else:
        raise ValueError(view)
    return df.select(expr.alias("t"))["t"].to_list()


def topk(query, index_t, k, n_threads, chunk=50_000, threshold=0.0):
    """Row-wise top-k of query @ index_t. Returns (q_row, i_col, score, rank)."""
    rows, cols, vals, ranks = [], [], [], []
    for start in range(0, query.shape[0], chunk):
        c = sp_matmul_topn(query[start:start + chunk], index_t, top_n=k,
                           threshold=threshold, sort=True, n_threads=n_threads)
        counts = np.diff(c.indptr)
        rows.append(np.repeat(np.arange(start, start + c.shape[0], dtype=np.int32), counts))
        cols.append(c.indices.astype(np.int32))
        vals.append(c.data.astype(np.float32))
        ranks.append((np.arange(c.nnz) - np.repeat(c.indptr[:-1], counts)).astype(np.int16))
    cat = lambda xs: np.concatenate(xs) if xs else np.array([])
    return cat(rows), cat(cols), cat(vals), cat(ranks)


def block_country(s1, rec, views, k, n_threads=THREADS, max_df=None, log=print, reverse=None):
    """Candidates for one country's frames. Returns row indices into s1 / rec.

    k: top-K per view, an int or {view: K}.
    max_df: optional {view: cap} overriding the defaults in VIEWS.
    reverse: optional {view: (K, min_cos)}; also retrieve each RECORD's top-K
      S1 entities with cosine >= min_cos in that view (columns <view>_rcos /
      <view>_rrank). This finds records whose entity's own top-K list is
      crowded by look-alike records.
    """
    parts = []
    for view in views:
        t = time.time()
        cap = (max_df or {}).get(view, VIEWS[view]["max_df"])
        kv = k[view] if isinstance(k, dict) else k
        q, i = tfidf_pair(view_text(s1, view), view_text(rec, view), view_cfg(view), max_df=cap)
        t_vec = time.time() - t
        r, c, v, rk = topk(q, i.T.tocsr(), kv, n_threads)
        parts.append(pl.DataFrame({
            "s1_row": r, "rec_row": c, f"{view}_cos": v, f"{view}_rank": rk}))
        log(f"    {view:6s} max_df={cap:>7,}  pairs={len(r):>11,}  "
            f"tfidf {t_vec:4.0f}s  topk {time.time() - t - t_vec:5.0f}s")
        if reverse and view in reverse:
            t = time.time()
            k_rev, min_cos = reverse[view]
            r, c, v, rk = topk(i, q.T.tocsr(), k_rev, n_threads, threshold=min_cos)
            parts.append(pl.DataFrame({
                "s1_row": c, "rec_row": r, f"{view}_rcos": v, f"{view}_rrank": rk}))
            log(f"    {view:6s} reverse top-{k_rev} (cos>={min_cos})  pairs={len(r):>11,}  "
                f"topk {time.time() - t:5.0f}s")
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, on=["s1_row", "rec_row"], how="full", coalesce=True)
    return out


def generate(s1, rec, views=tuple(VIEWS), k=20, n_threads=THREADS, max_df=None, log=print):
    """Candidate pairs (s1_id, rec_id, per-view cos/rank) over all countries.

    s1 / rec are normalised frames from ber.prepare. Countries are taken from
    the data itself, so unseen countries (France in test) are handled the same.
    """
    out = []
    for country in sorted(s1["country"].unique().to_list()):
        s1c = s1.filter(pl.col("country") == country)
        recc = rec.filter(pl.col("country") == country)
        log(f"  {country}: {len(s1c):,} S1 x {len(recc):,} records")
        cand = block_country(s1c, recc, views, k, n_threads, max_df, log)
        out.append(
            cand.with_columns(
                s1_id=s1c["entity_id"].gather(cand["s1_row"]),
                rec_id=recc["entity_id"].gather(cand["rec_row"]),
            ).drop("s1_row", "rec_row")
        )
    return pl.concat(out, how="diagonal")
