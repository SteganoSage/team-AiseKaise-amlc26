# Pretrained Models Used

Every pretrained model in the final pipeline is listed here with license and parameter count.
Only MIT or Apache 2.0 licensed models with ≤ 8B parameters are allowed.

| Model | HuggingFace ID | License | Params | Used For |
|---|---|---|---|---|
| Multilingual E5 Small | `intfloat/multilingual-e5-small` | MIT (checked on the HF card, 24 Sep 2026) | 118M | Optional (`--embeddings`, off by default): embedding blocker + cosine features |

## Notes

- The default pipeline uses no pretrained model at all (TF-IDF + rapidfuzz + LightGBM trained on the provided data).
- Libraries (LightGBM, scikit-learn, rapidfuzz, pyarrow, torch) are not pretrained models and don't require listing.
- Before adding any model, verify the license on its HuggingFace model card.
- Avoid: CC-BY-NC, Llama-license, Gemma-license, or unknown license.
