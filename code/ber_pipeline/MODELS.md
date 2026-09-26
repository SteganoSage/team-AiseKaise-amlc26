# Pretrained models used by the ber pipeline

| Model | HuggingFace ID | License | Params | Used for |
|---|---|---|---|---|
| Multilingual E5 Base | `intfloat/multilingual-e5-base` | MIT (per the model card; re-check the card before the final zip) | 278M | Cross-encoder (`ber/crossenc.py`): fine-tuned as a pair classifier on raw "name \| address" texts of uncertain pairs; its score is a stage-2 LightGBM feature |

Everything else is trained from the provided data only (TF-IDF, rapidfuzz, LightGBM).
Libraries (polars, sparse_dot_topn, anyascii, torch, transformers) are not pretrained models.
