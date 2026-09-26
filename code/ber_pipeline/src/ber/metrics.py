"""Challenge metric (macro F0.5 per Source 1 entity) and blocking diagnostics."""
import polars as pl


def _per_entity(pred, truth, s1_ids):
    """Per-S1 counts: n_pred, n_true, tp. pred/truth are (s1_id, rec_id) frames."""
    base = pl.DataFrame({"s1_id": s1_ids})
    npred = pred.group_by("s1_id").agg(n_pred=pl.len())
    ntrue = truth.group_by("s1_id").agg(n_true=pl.len())
    tp = pred.join(truth, on=["s1_id", "rec_id"]).group_by("s1_id").agg(tp=pl.len())
    return (
        base.join(npred, on="s1_id", how="left")
        .join(ntrue, on="s1_id", how="left")
        .join(tp, on="s1_id", how="left")
        .fill_null(0)
    )


def f05_macro(pred, truth, s1_ids):
    """Macro-averaged F0.5 exactly as the challenge defines it.

    Empty prediction on an entity with no true matches scores 1.0; any other
    case with tp == 0 scores 0.
    """
    e = _per_entity(pred, truth, s1_ids)
    f = pl.when((pl.col("n_pred") == 0) & (pl.col("n_true") == 0)).then(1.0).otherwise(
        1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")))
    return e.select(f.fill_nan(0.0).mean()).item()


def blocking_report(cands, truth, s1_ids):
    """Recall of true pairs and the F0.5 ceiling a perfect matcher could reach."""
    e = _per_entity(cands, truth, s1_ids)
    found = e["tp"].sum()
    total = e["n_true"].sum()
    # Perfect matcher on these candidates: predicts exactly the true pairs found.
    ceil = pl.when(e["n_true"] == 0).then(1.0).otherwise(
        1.25 * e["tp"] / (0.25 * e["n_true"] + e["tp"])).fill_nan(0.0)
    return {
        "pairs": len(cands),
        "pairs_per_s1": len(cands) / max(len(s1_ids), 1),
        "pair_recall": found / max(total, 1),
        "entities_fully_covered": ((e["tp"] == e["n_true"])).mean(),
        "f05_ceiling": e.select(ceil).to_series().mean(),
    }
