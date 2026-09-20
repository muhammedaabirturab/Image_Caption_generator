# Models Directory

The trained model checkpoints **are committed** to this repository so
that `streamlit run app.py` works immediately after cloning, with no
local training required.

## Files produced by `python -m src.train`

| File | Description |
|---|---|
| `caption_model.keras` | Final model after training completes. |
| `caption_model_best.keras` | Checkpoint with the lowest validation loss (`ModelCheckpoint`) — this is what the app and `src/evaluate.py` load if present. |
| `model_config.json` | Small JSON with `vocab_size`, `max_length`, `embedding_dim`, `lstm_units` — needed to reconstruct inference exactly. |

The tokenizer (`tokenizer.pkl`) lives under `data/processed/` (also
committed) since it is a text-preprocessing artifact, not a model
weight file.

## Retraining

If you change the code, data, or hyperparameters and want to regenerate
these files yourself:

```
python -m src.train
```

This overwrites the files in this folder with a freshly trained model.
