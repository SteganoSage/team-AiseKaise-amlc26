"""Pairwise features for candidate (S1 entity, S2/S3 record) pairs.

Feature groups
  tfidf    exact cosine per blocking view (IDF-weighted overlap), computed for
           every candidate pair, plus the blocking rank where retrieved
  context  how a pair compares with the other candidates of the same S1
           entity and of the same record (rank, gap to best, group size).
           Records belong to at most one S1, so competition is informative.
           Record-side context is computed over the FULL candidate table so it
           means the same thing in training and test.
  name     fuzzy ratios on full / core / compact / skeleton forms, token
           Jaccard and containment, legal-form agreement, flags
  address  fuzzy ratios, token Jaccard, house-number and state agreement

Country is never a feature, so the model applies unchanged to unseen countries.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from .blocking import VIEWS, view_cfg, view_text
from .tfidf import tfidf_pair

NAME_COLS = ["n_full", "n_core", "n_compact", "n_skel", "n_legal", "a_norm", "a_state", "a_nums"]
S1_COLS = NAME_COLS + ["core_dup"]
REC_COLS = NAME_COLS + ["n_is_compact", "n_has_dba", "n_is_indic", "a_is_indic",
                        "a_is_empty", "source", "n_core_tr", "unseen_frac",
                        "core_s1_count", "skel_s1_count"]
# Roman numerals / digits inside names distinguish planted near-duplicates
# ("frontier project ii" vs "iii").
_MARK = r"^(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|\d+)$"
_FILLER_ADDED = ["group", "groupe", "holdings", "holding", "partners", "services",
                 "service", "center", "centre", "enterprises", "international"]
FUZZ = {
    "ratio": fuzz.ratio,
    "tset": fuzz.token_set_ratio,
    "tsort": fuzz.token_sort_ratio,
    "partial": fuzz.partial_ratio,
    "jw": JaroWinkler.normalized_similarity,
}
FUZZ_PLAN = {
    "n_full": ["ratio", "tset", "tsort", "partial", "jw"],
    "n_core": ["ratio", "tset", "partial", "jw"],
    "n_compact": ["ratio", "partial"],
    "n_skel": ["ratio", "tset"],
    "n_core_tr": ["ratio", "tset", "jw"],
    "a_norm": ["ratio", "tset", "tsort", "partial"],
}
COS_VIEWS = ("name3", "name4", "skel3", "words", "words2", "addr")
COS_COLS = [f"{v}_cosx" for v in COS_VIEWS]


# ------------------------------------------------------------ candidate table
def add_exact_cosines(cand, s1, rec, chunk=2_000_000, log=print):
    """Add `<view>_cosx` (exact TF-IDF cosine, no frequency cap) for every pair."""
    out = []
    for country in sorted(s1["country"].unique().to_list()):
        s1c = s1.filter(pl.col("country") == country).with_row_index("i")
        recc = rec.filter(pl.col("country") == country).with_row_index("j")
        p = (cand.join(s1c.select(pl.col("entity_id").alias("s1_id"), "i"), on="s1_id")
                 .join(recc.select(pl.col("entity_id").alias("rec_id"), "j"), on="rec_id"))
        ii, jj = p["i"].to_numpy(), p["j"].to_numpy()
        cols = {}
        for view in COS_VIEWS:
            q, m = tfidf_pair(view_text(s1c, view), view_text(recc, view), view_cfg(view),
                              max_df=None, min_df=1)
            vals = np.empty(len(ii), dtype=np.float32)
            for s in range(0, len(ii), chunk):
                a, b = q[ii[s:s + chunk]], m[jj[s:s + chunk]]
                vals[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
            cols[f"{view}_cosx"] = vals
        out.append(p.drop("i", "j").with_columns(**cols))
        log(f"    exact cosines {country}: {len(p):,} pairs")
    return pl.concat(out)


def add_context(cand, score_cols, group):
    """Rank / gap-to-best / group size of each score within `group`."""
    tag = "s1" if group == "s1_id" else "rec"
    feats = [pl.len().over(group).alias(f"n_cand_{tag}")]
    for c in score_cols:
        feats += [
            pl.col(c).rank("ordinal", descending=True).over(group).cast(pl.Int16).alias(f"{c}_rk_{tag}"),
            (pl.col(c).max().over(group) - pl.col(c)).alias(f"{c}_gap_{tag}"),
        ]
    return cand.with_columns(feats)


# ------------------------------------------------------------ pair features
def add_record_extras(rec, s1_all, table):
    """Record-level columns: translated core name, share of unseen name words.

    unseen_frac: share of the record's (translated) core-name words that occur
    in no Source 1 name of the split; records renamed to an invented brand
    ("quodelta") score 1.0.
    """
    from .translit import translate_expr
    rec = rec.with_columns(n_core_tr=translate_expr("n_core", table))
    vocab = (s1_all.select(pl.col("n_core").str.split(" ").alias("w")).explode("w")
                   .unique().with_columns(seen=pl.lit(1, pl.Int8)))
    toks = (rec.select("entity_id", pl.col("n_core_tr").str.split(" ").alias("w")).explode("w")
               .filter(pl.col("w") != "")
               .join(vocab, on="w", how="left")
               .group_by("entity_id").agg(unseen_frac=(1 - pl.col("seen").fill_null(0).mean())
                                          .cast(pl.Float32)))
    # How many Source 1 entities carry exactly this (core / skeleton) name.
    # A name unique among Source 1 is safe to match on name alone, which
    # matters for records without an address; "tech foods" is not.
    by_core = (s1_all.group_by("country", "n_core").len()
                     .rename({"n_core": "n_core_tr", "len": "core_s1_count"}))
    by_skel = s1_all.group_by("country", "n_skel").len().rename({"len": "skel_s1_count"})
    return (rec.join(toks, on="entity_id", how="left")
               .join(by_core, on=["country", "n_core_tr"], how="left")
               .join(by_skel, on=["country", "n_skel"], how="left")
               .with_columns(pl.col("core_s1_count", "skel_s1_count").fill_null(0).cast(pl.Int32)))


def add_s1_extras(s1):
    """Source 1 columns: number of Source 1 entities sharing the core name."""
    return s1.with_columns(core_dup=pl.len().over("country", "n_core").cast(pl.Int32))


def attach(pairs, s1, rec):
    """Join normalised columns of both sides onto (s1_id, rec_id) pairs."""
    left = s1.select(pl.col("entity_id").alias("s1_id"),
                     *[pl.col(c).alias(f"s1_{c}") for c in S1_COLS],
                     pl.col("n_core").alias("s1_n_core_tr"))
    right = rec.select(pl.col("entity_id").alias("rec_id"),
                       *[pl.col(c).alias(f"r_{c}") for c in REC_COLS])
    return pairs.join(left, on="s1_id").join(right, on="rec_id")


def _fuzzy(df, workers=-1):
    """rapidfuzz similarities for every (column, scorer) of FUZZ_PLAN, computed in parallel.

    Args:
        df: Pairs with s1_<col> and r_<col> text columns (see attach).
        workers: Threads for rapidfuzz (-1 = all cores).

    Returns:
        DataFrame with one float32 column per similarity, aligned with df.
    """
    out = {}
    for col, scorers in FUZZ_PLAN.items():
        a, b = df[f"s1_{col}"].to_list(), df[f"r_{col}"].to_list()
        for name in scorers:
            out[f"{col}_{name}"] = cpdist(a, b, scorer=FUZZ[name], workers=workers,
                                          dtype=np.float32)
    return pl.DataFrame(out)


def _tok(col):
    """Polars expression: the column split into non-empty space-separated tokens."""
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _set_features(df):
    """Token-set, number, state, legal-form, marker and flag features of each pair.

    Args:
        df: Pairs with both sides' normalised columns (see attach).

    Returns:
        DataFrame of features aligned with df.
    """
    feats = []
    for col in ("n_core", "n_core_tr", "a_norm", "a_nums", "n_skel"):
        a, b = _tok(f"s1_{col}"), _tok(f"r_{col}")
        inter = a.list.set_intersection(b).list.len()
        la, lb = a.list.len(), b.list.len()
        feats += [
            (inter / (la + lb - inter)).cast(pl.Float32).alias(f"{col}_jacc"),
            (inter / la).cast(pl.Float32).alias(f"{col}_cont_s1"),
            (inter / lb).cast(pl.Float32).alias(f"{col}_cont_r"),
            la.cast(pl.Int16).alias(f"{col}_len_s1"),
            lb.cast(pl.Int16).alias(f"{col}_len_r"),
        ]
    first = lambda c: _tok(c).list.first()
    both = lambda c: (pl.col(f"s1_{c}") != "") & (pl.col(f"r_{c}") != "")
    num = lambda c: first(c).cast(pl.Int64, strict=False)
    marks = lambda c: _tok(c).list.eval(pl.element().filter(pl.element().str.contains(_MARK)))
    m1, m2 = marks("s1_n_core"), marks("r_n_core_tr")
    full1, full2 = _tok("s1_n_full"), _tok("r_n_full")
    # Records reduced to initials or a handle ("jcqdro", "rp", "@pamehpad")
    core1 = _tok("s1_n_core")
    initials = core1.list.eval(pl.element().str.slice(0, 1)).list.join("")
    multi = core1.list.len() >= 2
    feats += [
        (multi & (pl.col("r_n_compact") == initials)).alias("acro_eq"),
        (multi & pl.col("r_n_compact").str.starts_with(initials)).alias("acro_prefix"),
    ]
    added = full2.list.set_difference(full1).list.eval(
        pl.element().filter(pl.element().is_in(_FILLER_ADDED)))
    feats += [
        (num("s1_a_nums") - num("r_a_nums")).abs().log1p().cast(pl.Float32).alias("num_first_logdiff"),
        ((m1.list.len() > 0) & (m2.list.len() > 0)
         & (m1.list.set_symmetric_difference(m2).list.len() > 0)).alias("name_mark_conflict"),
        ((m1.list.len() > 0) & (m2.list.len() == 0)).alias("name_mark_missing"),
        added.list.len().cast(pl.Int8).alias("r_filler_added"),
        (full2.list.last().is_in(_FILLER_ADDED)
         & ~full1.list.contains(full2.list.last())).alias("r_filler_last"),
        pl.col("r_unseen_frac").alias("r_unseen_frac"),
        pl.col("s1_core_dup").alias("s1_core_dup"),
        pl.col("r_core_s1_count").alias("r_core_s1_count"),
        pl.col("r_skel_s1_count").alias("r_skel_s1_count"),
        (first("s1_a_nums") == first("r_a_nums")).alias("num_first_eq"),
        (both("a_state") & (pl.col("s1_a_state") == pl.col("r_a_state"))).alias("state_eq"),
        (both("a_state") & (pl.col("s1_a_state") != pl.col("r_a_state"))).alias("state_conflict"),
        (both("n_legal") & (pl.col("s1_n_legal") == pl.col("r_n_legal"))).alias("legal_eq"),
        (both("n_legal") & (pl.col("s1_n_legal") != pl.col("r_n_legal"))).alias("legal_conflict"),
        (first("s1_n_core") == first("r_n_core")).alias("first_tok_eq"),
        pl.col("r_n_is_compact").alias("r_compact"),
        pl.col("r_n_has_dba").alias("r_dba"),
        pl.col("r_n_is_indic").alias("r_name_indic"),
        pl.col("r_a_is_indic").alias("r_addr_indic"),
        pl.col("r_a_is_empty").alias("r_addr_empty"),
        (pl.col("r_source") == 3).alias("r_is_s3"),
        pl.col("s1_n_compact").str.len_chars().cast(pl.Int16).alias("s1_name_chars"),
        pl.col("r_n_compact").str.len_chars().cast(pl.Int16).alias("r_name_chars"),
    ]
    return df.select(feats)


def word_idf(s1_all):
    """IDF of each core-name word over the split's Source 1 names."""
    words = s1_all.select(pl.col("n_core").str.split(" ").list.unique().alias("w")).explode("w")
    n = len(s1_all)
    return (words.filter(pl.col("w") != "").group_by("w").len()
                 .select("w", (n / pl.col("len")).log().cast(pl.Float32).alias("idf")))


def _idf_diff(df, idf):
    """How rare are the words that differ between the two core names?

    The noise swaps a name word for another common one ("office" -> "service",
    "sport" -> "loisirs"); a different business differs in a rare word. Words
    unknown to Source 1 (invented or garbled) get the maximum IDF.
    """
    max_idf = float(idf["idf"].max() or 0.0)
    a, b = _tok("s1_n_core"), _tok("r_n_core_tr")
    d = df.select("s1_id", "rec_id", a.list.set_difference(b).alias("s1_only"),
                  b.list.set_difference(a).alias("r_only"))
    out = d.select("s1_id", "rec_id",
                   pl.col("s1_only").list.len().cast(pl.Int16).alias("n_s1_only"),
                   pl.col("r_only").list.len().cast(pl.Int16).alias("n_r_only"))
    for side in ("s1_only", "r_only"):
        e = (d.select("s1_id", "rec_id", pl.col(side).alias("w")).explode("w")
              .filter(pl.col("w").is_not_null() & (pl.col("w") != ""))
              .join(idf, on="w", how="left").with_columns(pl.col("idf").fill_null(max_idf)))
        agg = e.group_by("s1_id", "rec_id").agg(
            pl.col("idf").max().alias(f"{side}_idf_max"), pl.col("idf").sum().alias(f"{side}_idf_sum"))
        out = out.join(agg, on=["s1_id", "rec_id"], how="left")
    return out.with_columns(pl.col("^.*_idf_(max|sum)$").fill_null(0.0).cast(pl.Float32))


def pair_features(cand, s1, rec, workers=-1, idf=None):
    """String/set features joined onto candidate rows (keeps all cand columns).

    Adds S1-side context for the fuzzy scores; record-side context must
    already be in `cand` (see add_context on the full table). `idf` (from
    word_idf) enables the word-substitution features.
    """
    df = attach(cand.select("s1_id", "rec_id"), s1, rec)
    feats = pl.concat([df.select("s1_id", "rec_id"), _fuzzy(df, workers), _set_features(df)],
                      how="horizontal")
    if idf is not None:
        feats = feats.join(_idf_diff(df, idf), on=["s1_id", "rec_id"], how="left")
    feats = cand.join(feats, on=["s1_id", "rec_id"])
    return add_context(feats, ["n_core_tset", "n_full_jw", "a_norm_tset", "n_skel_tset",
                               "n_core_tr_tset"], "s1_id")


def feature_columns(df):
    """Model input columns: everything except ids and the label."""
    return [c for c in df.columns if c not in ("s1_id", "rec_id", "label", "country")]
