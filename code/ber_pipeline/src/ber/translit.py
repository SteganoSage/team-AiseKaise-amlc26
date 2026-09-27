"""Token translation table for transliterated business names.

About 18% of Indian Source 2/3 names are written in Indic scripts. After
transliteration to ASCII they read like "phuds" (foods), "tredimg" (trading),
"praibhet limitet" (private limited). Business names reuse a small vocabulary,
so a table learned from training matches undoes most of it: for every token of
a transliterated record name, the Source 1 name token it co-occurs with most.
"""
import json

import polars as pl

from .config import work
from .normalize import LEGAL


def learn(s1, rec, gt, min_count=20, min_purity=0.5):
    """{transliterated token -> Latin token} from matched training pairs."""
    r = (rec.filter(pl.col("n_is_indic"))
            .select(pl.col("entity_id").alias("rec_id"),
                    pl.col("n_full").str.split(" ").alias("t")))
    s = s1.select(pl.col("entity_id").alias("s1_id"), pl.col("n_full").str.split(" ").alias("s"))
    pairs = (gt.join(r, on="rec_id").join(s, on="s1_id")
               .explode("t").explode("s")
               .filter((pl.col("t") != "") & (pl.col("s") != "")))
    n_t = gt.join(r, on="rec_id").explode("t").group_by("t").len().rename({"len": "n"})
    n_s = pairs.select("rec_id", "s").unique().group_by("s").len().rename({"len": "n_s"})
    # Cosine-style score so that ubiquitous targets ("ltd", "pvt") only win
    # for tokens that really are legal forms ("limitet", "praibhet").
    best = (pairs.group_by("t", "s").len()
                 .join(n_t, on="t").join(n_s, on="s")
                 .with_columns(score=pl.col("len") / (pl.col("n") * pl.col("n_s")).sqrt())
                 .sort("score", descending=True)
                 .group_by("t").first()
                 .filter((pl.col("n") >= min_count) & (pl.col("len") / pl.col("n") >= min_purity)
                         & (pl.col("t") != pl.col("s"))))
    return dict(best.select("t", "s").iter_rows())


def load_or_learn(log=print):
    """The learned transliteration table (cached in WORK_DIR/norm/translit_map.json)."""
    path = work("norm", "translit_map.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from .io import load_gt_pairs
    s1 = pl.read_parquet(work("norm", "train_s1.parquet"), columns=["entity_id", "n_full"])
    rec = pl.read_parquet(work("norm", "train_rec.parquet"),
                          columns=["entity_id", "n_full", "n_is_indic"])
    table = learn(s1, rec, load_gt_pairs())
    path.write_text(json.dumps(table, ensure_ascii=False, indent=0), encoding="utf-8")
    log(f"learned {len(table):,} transliteration token mappings")
    return table


def extend_with_skeletons(table, s1, rec, min_share=0.5, min_count=3):
    """Add mappings for transliterated words the learned table does not know.

    Each such word is mapped to the Source 1 word with the same consonant
    skeleton ("kubr" and "kuber" -> "kbr"), when one word clearly dominates that
    skeleton among Source 1 names of the split. Covers rarer words (personal
    and place names) that occur too seldom in training matches to be learned.
    """
    from .normalize import skeleton
    sk = lambda w: skeleton([w])
    words = (s1.select(pl.col("n_core").str.split(" ").alias("w")).explode("w")
               .filter(pl.col("w").str.len_chars() >= 3).group_by("w").len())
    words = words.with_columns(sk=pl.col("w").map_elements(sk, return_dtype=pl.String))
    by_sk = (words.with_columns(total=pl.col("len").sum().over("sk"))
                  .sort("len", descending=True).group_by("sk").first()
                  .filter((pl.col("len") >= min_count) & (pl.col("len") / pl.col("total") >= min_share)
                          & (pl.col("sk").str.len_chars() >= 2)))
    unknown = (rec.filter(pl.col("n_is_indic"))
                  .select(pl.col("n_core").str.split(" ").alias("t")).explode("t")
                  .filter(pl.col("t").str.len_chars() >= 2).unique()
                  .filter(~pl.col("t").is_in(list(table))))
    unknown = unknown.with_columns(sk=pl.col("t").map_elements(sk, return_dtype=pl.String))
    new = unknown.join(by_sk.select("sk", "w"), on="sk").filter(pl.col("t") != pl.col("w"))
    return {**table, **dict(new.select("t", "w").iter_rows())}


def table_for(split, log=print):
    """Learned table + skeleton fallback for one split (cached)."""
    path = work("norm", f"translit_{split}.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from .io import load_norm_s1
    base = load_or_learn(log)
    rec = pl.read_parquet(work("norm", f"{split}_rec.parquet"), columns=["n_core", "n_is_indic"])
    table = extend_with_skeletons(base, load_norm_s1(split), rec)
    path.write_text(json.dumps(table, ensure_ascii=False, indent=0), encoding="utf-8")
    log(f"transliteration table for {split}: {len(base):,} learned + "
        f"{len(table) - len(base):,} skeleton mappings")
    return table


def translate_expr(col, table):
    """Polars expression: map each token of `col`; drop legal forms from the result."""
    legal = set(LEGAL.values()) | set(LEGAL)
    mapping = pl.DataFrame({"k": list(table), "v": list(table.values())})
    return (pl.col(col).str.split(" ")
              .list.eval(pl.element().replace(mapping["k"], mapping["v"]))
              .list.eval(pl.element().filter(~pl.element().is_in(list(legal))))
              .list.join(" "))
