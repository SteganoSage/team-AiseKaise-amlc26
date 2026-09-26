"""Cross-encoder score for uncertain pairs (fine-tuned multilingual transformer).

A pretrained multilingual encoder (intfloat/multilingual-e5-base, MIT licence,
278M parameters) is fine-tuned as a pair classifier on the RAW texts
"name | address" of both sides, so it sees the original scripts (Devanagari,
Telugu, ...) and learns typo, transliteration, abbreviation and decoy
patterns (ii/iii, PC/LLC) directly from text.

Leakage control: it is trained only on the candidate pairs of a reserved pool
of training entities (CE_POOL of the kept entities, never used to fit the
LightGBM models), so its scores on the LightGBM training entities and on the
test set are out-of-sample.

Only pairs whose first-stage probability is in [LO, HI] are scored (the rest
are near-certain); the score is a stage-2 feature and missing elsewhere, for
training and test alike.

Run from src/ (after `ber.model stage1`):
  python -m ber.crossenc train
  python -m ber.crossenc score
"""
import argparse
import math
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from .config import stage_dir, work
from .io import in_ce_pool, load_gt_pairs

MODEL = "intfloat/multilingual-e5-base"  # MIT licence, 278M parameters
LO, HI = 0.0005, 0.995  # p1 band that gets a cross-encoder score
MAXLEN = 96
EPOCHS = 1
BATCH = 256
LR = 5e-5
EASY_SAMPLE = 0.03     # share of out-of-band pairs added to the training set
TAG = ""


def model_dir():
    return work(stage_dir("models", TAG), "ce", "x").parent


def _texts(split):
    """entity_id -> "name | address" from the raw (un-normalised) records."""
    frames = []
    for n in (1, 2, 3):
        df = pl.read_parquet(work("raw", f"{split}_source{n}.parquet"),
                             columns=["entity_id", "business_name", "business_address"])
        frames.append(df.select("entity_id",
                                pl.concat_str(["business_name", pl.lit(" | "), "business_address"])
                                .alias("text")))
    return pl.concat(frames)


def _attach_text(pairs, texts):
    return (pairs.join(texts.rename({"entity_id": "s1_id", "text": "ta"}), on="s1_id")
                 .join(texts.rename({"entity_id": "rec_id", "text": "tb"}), on="rec_id"))


def _batches(n, size, order):
    for i in range(0, n, size):
        yield order[i:i + size]


def train(log):
    p1 = pl.read_parquet(work(stage_dir("models", TAG), "p1_train.parquet"))
    pool = p1.filter(in_ce_pool(pl.col("s1_id")))
    band = pl.col("p1").is_between(LO * 0.2, 1 - (1 - HI) * 0.2)
    easy = ~band & (pl.col("rec_id").hash(seed=7) % 10_000 < int(EASY_SAMPLE * 10_000))
    data = pool.filter(band | easy)
    gt = load_gt_pairs().with_columns(label=pl.lit(1.0, pl.Float32))
    data = (data.join(gt, on=["s1_id", "rec_id"], how="left")
                .with_columns(pl.col("label").fill_null(0.0)))
    data = _attach_text(data, _texts("train")).sample(fraction=1.0, shuffle=True, seed=0)
    val_mask = (data["s1_id"].hash(seed=5) % 20 == 0).to_numpy()
    ta, tb, y = data["ta"].to_list(), data["tb"].to_list(), data["label"].to_numpy()
    tr_idx, va_idx = np.where(~val_mask)[0], np.where(val_mask)[0]
    log(f"cross-encoder training pairs {len(tr_idx):,} (pos {y[tr_idx].mean():.3f}), "
        f"validation {len(va_idx):,}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * math.ceil(len(tr_idx) / BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    rng = np.random.default_rng(0)
    step = 0
    for epoch in range(EPOCHS):
        model.train()
        for idx in _batches(len(tr_idx), BATCH, rng.permutation(tr_idx)):
            enc = tok([ta[i] for i in idx], [tb[i] for i in idx], truncation=True,
                      max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits.float(),
                                                      torch.tensor(y[idx], device="cuda"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 2000 == 0:
                log(f"  epoch {epoch} step {step}/{steps} loss {loss.item():.4f}")
        pv = _predict(model, tok, [ta[i] for i in va_idx], [tb[i] for i in va_idx])
        yv = y[va_idx]
        ll = -np.mean(yv * np.log(np.clip(pv, 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - pv, 1e-6, 1)))
        acc = np.mean((pv > 0.5) == (yv > 0.5))
        log(f"epoch {epoch}: validation logloss {ll:.4f}  accuracy {acc:.4f}")
    model.save_pretrained(model_dir())
    tok.save_pretrained(model_dir())


@torch.no_grad()
def _predict(model, tok, ta, tb, batch=2048):
    model.eval()
    order = np.argsort([len(a) + len(b) for a, b in zip(ta, tb)])
    out = np.empty(len(ta), dtype=np.float32)
    for idx in _batches(len(order), batch, order):
        enc = tok([ta[i] for i in idx], [tb[i] for i in idx], truncation=True,
                  max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    return out


def score(log):
    tok = AutoTokenizer.from_pretrained(model_dir())
    model = AutoModelForSequenceClassification.from_pretrained(model_dir()).cuda()
    for split in ("train", "test"):
        path = work(stage_dir("models", TAG), f"p1_{split}.parquet")
        if not path.exists():
            continue
        p1 = pl.read_parquet(path).filter(pl.col("p1").is_between(LO, HI))
        if split == "train":  # pool pairs are in-sample for the cross-encoder
            p1 = p1.filter(~in_ce_pool(pl.col("s1_id")))
        data = _attach_text(p1.select("s1_id", "rec_id"), _texts(split))
        t = time.time()
        ce = _predict(model, tok, data["ta"].to_list(), data["tb"].to_list())
        data.select("s1_id", "rec_id").with_columns(ce=pl.Series(ce)).write_parquet(
            work(stage_dir("models", TAG), f"ce_{split}.parquet"))
        log(f"scored {len(data):,} {split} pairs in {time.time() - t:.0f}s")


def main():
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "score"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    TAG = args.tag
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    train(log) if args.stage == "train" else score(log)


if __name__ == "__main__":
    main()
