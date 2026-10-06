# Model files

The scoring engine needs two ONNX model files in the project root. They are **not in git**
(`*.onnx` is in `.gitignore`, ~180 MB together), so they must be obtained separately.

| File in this folder | Source (Hugging Face repo / path) | Size | SHA-256 |
|---|---|---|---|
| `model.onnx` | `sentence-transformers/all-MiniLM-L6-v2` / `onnx/model.onnx` | 90.4 MB | `6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452` |
| `cross_encoder_model.onnx` | `cross-encoder/ms-marco-MiniLM-L6-v2` / `onnx/model.onnx` | 91.0 MB | `5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a` |

Both checksums were verified against the SHA-256 that Hugging Face publishes for each file
(October 2026).

`tokenizer.json` (tracked in git) is shared by both models: BERT-uncased WordPiece,
30,522 tokens. SHA-256 `da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0`.
It is not byte-identical to either repo's own `tokenizer.json`; the code sets truncation
and padding itself.

## Where the copies are kept

- Google Drive: `SkillsDashboard-models/` (move to an SCSP shared drive when possible)

## If the files are missing

1. Copy them from the location above into the project root, keeping the exact file names.
2. Check them: `shasum -a 256 *.onnx` must match the table.
3. If Hugging Face is blocked on your network, download them in Google Colab with
   `huggingface_hub.hf_hub_download` and transfer them through Google Drive.
4. To run without them in the meantime, set `SKILLS_NO_MODELS=1` in `.env` (read-only
   mode: stored scores can be viewed and reviewed; scoring new text is disabled).
