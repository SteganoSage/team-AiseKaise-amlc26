# Pretrained Models Used

Every pretrained model in the final pipeline is listed here with license and parameter count.
Only MIT or Apache 2.0 licensed models with ≤ 8B parameters are allowed.

| Model | HuggingFace ID | License | Params | Used For |
|---|---|---|---|---|
| Multilingual E5 Small | `intfloat/multilingual-e5-small` | MIT | 118M | Embedding-based blocking & cosine features |

## Notes

- Libraries (LightGBM, scikit-learn, rapidfuzz, faiss) are not pretrained models and don't require listing.
- Before adding any model, verify the license on its HuggingFace model card.
- Avoid: CC-BY-NC, Llama-license, Gemma-license, or unknown license.
