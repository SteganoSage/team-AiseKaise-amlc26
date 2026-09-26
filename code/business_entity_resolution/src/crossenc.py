"""
Optional transformer cross-encoder score for the candidate pairs.

A small multilingual transformer (config.CROSSENC_MODEL_NAME, MIT, listed in
MODELS.md) is fine-tuned to read the two records' raw "name | address" texts
side by side and output a match probability. Reading raw text keeps what the
hand-made string features lose: typos inside words, Indic scripts, and a
single different word between two otherwise identical business names. The
probability becomes one more stage-3 feature, so LightGBM decides how much to
trust it.

No leakage into LightGBM: the cross-encoder is fine-tuned on the candidate
pairs of a reserved share of the training S1 entities (CROSSENC_TRAIN_FRAC),
and those entities are then left out of LightGBM training. Every score that
LightGBM trains on or predicts from comes from pairs the cross-encoder never
saw.

Needs a CUDA GPU (Kaggle: Settings → Accelerator → GPU). Without one it is
switched off for the whole run, so training and test always have the same
feature columns.
"""

import math
import time

import numpy as np

import config
from prune import take_strings


def enabled() -> bool:
    """
    Whether the cross-encoder runs in this process.

    Returns:
        True when config.USE_CROSSENC is on and a CUDA GPU is available
        (or config.CROSSENC_ALLOW_CPU is set, for tests on tiny data).
    """
    if not config.USE_CROSSENC:
        return False
    try:
        import torch
    except ImportError:
        print("  ⚠ Cross-encoder off: torch is not installed")
        return False
    if torch.cuda.is_available() or config.CROSSENC_ALLOW_CPU:
        return True
    print("  ⚠ Cross-encoder off: no CUDA GPU (Kaggle: Settings → Accelerator → GPU)")
    return False


def split_rows(rows: np.ndarray, seed: int = None) -> tuple:
    """
    Reserve a share of training S1 rows for fine-tuning the cross-encoder.

    Args:
        rows: Training S1 rows.
        seed: Random seed (default config.RANDOM_SEED).

    Returns:
        Tuple (cross-encoder rows, remaining rows for LightGBM), both sorted.
    """
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    shuffled = rng.permutation(rows)
    n_ce = int(round(len(rows) * config.CROSSENC_TRAIN_FRAC))
    return np.sort(shuffled[:n_ce]), np.sort(shuffled[n_ce:])


def pair_texts(data: dict, s1_idx: np.ndarray, tgt_idx: np.ndarray) -> tuple:
    """
    Raw "name | address" text of both sides of some pairs.

    Raw (not normalized) text is used: the tokenizer handles case, accents and
    Indic scripts itself.

    Args:
        data: Output of load_split (s1 and tgt frames).
        s1_idx: S1 row per pair.
        tgt_idx: Target row per pair.

    Returns:
        Tuple (left texts, right texts), lists of strings.
    """
    def texts(frame, rows):
        names = take_strings(frame, "business_name", rows)
        addrs = take_strings(frame, "business_address", rows)
        return [f"{n} | {a}" for n, a in zip(names, addrs)]
    return texts(data["s1"], s1_idx), texts(data["tgt"], tgt_idx)


def _device():
    """CUDA device when available, else CPU."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit(data: dict, pairs, labels: np.ndarray, ce_rows: np.ndarray,
        verbose: bool = True) -> dict:
    """
    Fine-tune the cross-encoder on the candidate pairs of the reserved S1 rows.

    Binary cross-entropy on one output logit, AdamW with linear warm-up/decay,
    fp16 autocast on GPU.

    Args:
        data: Output of load_split (train).
        pairs: Pruned candidate pairs of the split.
        labels: Their labels.
        ce_rows: Reserved S1 rows (split_rows).
        verbose: Whether to print progress.

    Returns:
        Dict with the tokenizer and the fine-tuned model.
    """
    import torch
    import torch.nn.functional as F
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    t0 = time.time()
    torch.manual_seed(config.RANDOM_SEED)
    sel = np.flatnonzero(np.isin(pairs["s1_idx"].to_numpy(), ce_rows))
    if len(sel) > config.CROSSENC_MAX_TRAIN_PAIRS:
        sel = np.sort(np.random.default_rng(config.RANDOM_SEED)
                      .choice(sel, config.CROSSENC_MAX_TRAIN_PAIRS, replace=False))
    left, right = pair_texts(data, pairs["s1_idx"].to_numpy()[sel], pairs["tgt_idx"].to_numpy()[sel])
    y = labels[sel].astype(np.float32)
    if verbose:
        print(f"\n[cross-encoder] Fine-tuning {config.CROSSENC_MODEL_NAME} on {len(sel):,} pairs "
              f"of {len(ce_rows):,} reserved S1 ({y.mean():.1%} matches), "
              f"{config.CROSSENC_EPOCHS} epoch(s)...")

    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(config.CROSSENC_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.CROSSENC_MODEL_NAME, num_labels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.CROSSENC_LR, weight_decay=0.01)
    batch = config.CROSSENC_TRAIN_BATCH
    steps = math.ceil(len(sel) / batch) * config.CROSSENC_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    model.train()
    rng = np.random.default_rng(config.RANDOM_SEED)
    step = 0
    for epoch in range(config.CROSSENC_EPOCHS):
        order = rng.permutation(len(sel))
        running = 0.0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            enc = tokenizer([left[i] for i in idx], [right[i] for i in idx], truncation=True,
                            max_length=config.CROSSENC_MAX_LENGTH, padding=True,
                            return_tensors="pt").to(device)
            target = torch.from_numpy(y[idx]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            running += loss.item()
            step += 1
            if verbose and step % 500 == 0:
                print(f"    step {step:,}/{steps:,}  loss {running / 500:.4f}")
                running = 0.0
    model.eval()
    if verbose:
        print(f"  cross-encoder fine-tuned in {time.time() - t0:.0f}s")
    return {"tokenizer": tokenizer, "model": model}


def score(bundle: dict, data: dict, pairs, verbose: bool = True) -> np.ndarray:
    """
    Match probability of every candidate pair.

    Pairs are scored in order of text length so each batch pads to a similar
    length (much faster), then put back in the original order.

    Args:
        bundle: Output of fit.
        data: Output of load_split for the pairs' split.
        pairs: Candidate pairs (s1_idx, tgt_idx).
        verbose: Whether to print progress.

    Returns:
        float32 array of probabilities, aligned with pairs.
    """
    import torch

    t0 = time.time()
    tokenizer, model = bundle["tokenizer"], bundle["model"]
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    out = np.empty(len(pairs), dtype=np.float32)
    s1_idx, tgt_idx = pairs["s1_idx"].to_numpy(), pairs["tgt_idx"].to_numpy()
    batch = config.CROSSENC_SCORE_BATCH
    chunk = config.CROSSENC_SCORE_CHUNK   # pairs whose texts are held at once
    for c_start in range(0, len(pairs), chunk):
        c_end = min(c_start + chunk, len(pairs))
        left, right = pair_texts(data, s1_idx[c_start:c_end], tgt_idx[c_start:c_end])
        order = np.argsort([len(a) + len(b) for a, b in zip(left, right)], kind="stable")
        probs = np.empty(len(order), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(order), batch):
                idx = order[start:start + batch]
                enc = tokenizer([left[i] for i in idx], [right[i] for i in idx], truncation=True,
                                max_length=config.CROSSENC_MAX_LENGTH, padding=True,
                                return_tensors="pt").to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(**enc).logits.squeeze(-1)
                probs[idx] = torch.sigmoid(logits.float()).cpu().numpy()
        out[c_start:c_end] = probs
        if verbose:
            done = c_end / max(len(pairs), 1)
            print(f"    scored {c_end:,}/{len(pairs):,} pairs ({done:.0%}, {time.time() - t0:.0f}s)")
    return out
