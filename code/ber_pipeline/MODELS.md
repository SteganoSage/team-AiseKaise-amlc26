# Pretrained models

| Model | Hugging Face ID | License | Parameters | Used for |
|---|---|---|---|---|
| Multilingual E5 Base | `intfloat/multilingual-e5-base` | MIT | 278M | Cross-encoder (`src/ber/crossenc.py`): fine-tuned as a pair classifier on the raw "name \| address" texts of uncertain pairs; its score is a stage-2 LightGBM feature |

Both limits of the rules are met: MIT licence, 278M ≤ 8B parameters.
Everything else is trained from the provided data only (TF-IDF, rapidfuzz, LightGBM).
Libraries (polars, sparse_dot_topn, anyascii, scikit-learn, LightGBM, torch,
transformers) are not pretrained models.
