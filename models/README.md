# Models Directory

Trained model files are **not** committed to this repository (they are
listed in `.gitignore`) to keep the repository small. Running the
training script regenerates everything here.

## Files produced by `python -m src.train`

| File | Description |
|---|---|
| `caption_model.keras` | Final model after training completes. |
| `caption_model_best.keras` | Checkpoint with the lowest validation loss (used by the app and evaluation if present). |
| `model_config.json` | Small JSON with `vocab_size`, `max_length`, `embedding_dim`, `lstm_units` — needed to reconstruct inference exactly. |

The tokenizer (`tokenizer.pkl`) lives under `data/processed/` since it is
a text-preprocessing artifact, not a model weight file.

## Getting a trained model

Either:

1. Run the training pipeline yourself (see the main `README.md`):

   ```
   python -m src.train
   ```

2. Or, if you were given a pre-trained `caption_model_best.keras` and
   `model_config.json` separately (e.g. shared via a cloud drive because
   of size), place them directly in this folder and the app will pick
   them up.
