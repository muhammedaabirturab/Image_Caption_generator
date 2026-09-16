"""
End-to-end training script.

Usage:
    python -m src.train
    python -m src.train --epochs 20 --batch-size 64
    python -m src.train --limit-images 200 --epochs 3   # quick sanity run

Steps:
    1. Clean captions (cached to data/processed/captions_clean.json).
    2. Build a tokenizer from the training split only (cached).
    3. Extract InceptionV3 features for every image that is used
       (cached to data/processed/image_features.pkl so the CNN never
       has to run twice).
    4. Build the encoder-decoder model.
    5. Train with teacher forcing: at every step the decoder is given
       the *true* previous words (from the ground-truth caption) and
       learns to predict the next one. This is what lets the LSTM be
       trained with efficient, parallel next-word supervision instead
       of waiting on its own (possibly wrong) earlier predictions.
    6. Save the best checkpoint (lowest validation loss), the final
       model, and a small config file the app/evaluation code needs
       for inference (vocab size, max caption length, ...).
    7. Plot training/validation loss so progress can be shown.
"""

import argparse
import json
import os
import pickle
import random

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src import config
from src.data_preprocessing import (
    build_tokenizer,
    clean_captions_mapping,
    get_splits,
    load_clean_captions,
    load_raw_captions,
    max_caption_length,
    save_clean_captions,
)
from src.feature_extraction import extract_features, load_features, save_features
from src.model import build_captioning_model

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


# ---------------------------------------------------------------------------
# Data preparation (each step is cached so re-runs are fast)
# ---------------------------------------------------------------------------
def prepare_captions():
    if os.path.exists(config.CLEAN_CAPTIONS_FILE):
        print(f"[data] Using cached clean captions: {config.CLEAN_CAPTIONS_FILE}")
        return load_clean_captions()
    print("[data] Cleaning raw captions...")
    raw = load_raw_captions()
    cleaned = clean_captions_mapping(raw)
    save_clean_captions(cleaned)
    return cleaned


def prepare_tokenizer(cleaned, train_ids):
    if os.path.exists(config.TOKENIZER_FILE):
        print(f"[data] Using cached tokenizer: {config.TOKENIZER_FILE}")
        with open(config.TOKENIZER_FILE, "rb") as f:
            return pickle.load(f)
    print("[data] Building tokenizer from training captions...")
    tokenizer = build_tokenizer(cleaned, train_ids)
    os.makedirs(os.path.dirname(config.TOKENIZER_FILE), exist_ok=True)
    with open(config.TOKENIZER_FILE, "wb") as f:
        pickle.dump(tokenizer, f)
    return tokenizer


def prepare_features(image_ids):
    cached = {}
    if os.path.exists(config.FEATURES_FILE):
        cached = load_features()
    missing = [i for i in image_ids if i not in cached]
    if missing:
        print(f"[data] Extracting CNN features for {len(missing)} image(s)...")
        new_features = extract_features(missing)
        cached.update(new_features)
        save_features(cached)
    else:
        print("[data] All required image features already cached.")
    return cached


# ---------------------------------------------------------------------------
# Training-sequence generator (teacher forcing)
# ---------------------------------------------------------------------------
def count_sequences(image_ids, cleaned, features):
    total = 0
    for image_id in image_ids:
        if image_id not in features:
            continue
        for caption in cleaned.get(image_id, []):
            total += max(len(caption.split()) - 1, 0)
    return total


def sequence_generator(image_ids, cleaned, features, tokenizer, max_length,
                        batch_size, vocab_size):
    usable_ids = [i for i in image_ids if i in features]
    if not usable_ids:
        raise ValueError("No images with cached features found for this split.")

    # Pre-tokenize once so the generator does not re-tokenize every epoch.
    tokenized = {
        image_id: [tokenizer.texts_to_sequences([c])[0] for c in cleaned.get(image_id, [])]
        for image_id in usable_ids
    }

    while True:
        random.shuffle(usable_ids)
        X1_batch, X2_batch, y_batch = [], [], []
        for image_id in usable_ids:
            feature = features[image_id]
            for seq in tokenized[image_id]:
                for i in range(1, len(seq)):
                    in_seq = pad_sequences([seq[:i]], maxlen=max_length)[0]
                    out_word = seq[i]
                    if out_word >= vocab_size:
                        continue
                    X1_batch.append(feature)
                    X2_batch.append(in_seq)
                    y_batch.append(out_word)
                    if len(X1_batch) == batch_size:
                        yield (np.array(X1_batch), np.array(X2_batch)), np.array(y_batch)
                        X1_batch, X2_batch, y_batch = [], [], []
        if X1_batch:
            yield (np.array(X1_batch), np.array(X2_batch)), np.array(y_batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the image caption generator.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--limit-images", type=int, default=None,
                         help="Use only the first N training images (quick sanity run).")
    args = parser.parse_args()

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    cleaned = prepare_captions()
    train_ids, val_ids, test_ids = get_splits()

    if args.limit_images:
        train_ids = train_ids[: args.limit_images]
        val_ids = val_ids[: max(1, args.limit_images // 5)]
        print(f"[train] Quick run: limited to {len(train_ids)} train / {len(val_ids)} val images.")

    tokenizer = prepare_tokenizer(cleaned, train_ids)
    vocab_size = min(len(tokenizer.word_index) + 1, config.MAX_VOCAB_SIZE)
    max_length = max_caption_length(cleaned, train_ids)
    print(f"[train] Vocabulary size: {vocab_size} | Max caption length: {max_length}")

    all_needed_ids = train_ids + val_ids
    features = prepare_features(all_needed_ids)

    train_steps = max(1, count_sequences(train_ids, cleaned, features) // args.batch_size)
    val_steps = max(1, count_sequences(val_ids, cleaned, features) // args.batch_size)
    print(f"[train] Steps/epoch: train={train_steps}, val={val_steps}")

    train_gen = sequence_generator(train_ids, cleaned, features, tokenizer,
                                    max_length, args.batch_size, vocab_size)
    val_gen = sequence_generator(val_ids, cleaned, features, tokenizer,
                                  max_length, args.batch_size, vocab_size)

    model = build_captioning_model(vocab_size, max_length, learning_rate=args.learning_rate)
    model.summary()

    callbacks = [
        ModelCheckpoint(config.BEST_MODEL_FILE, monitor="val_loss",
                         save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(config.MODEL_FILE)
    print(f"[train] Final model saved to {config.MODEL_FILE}")
    if os.path.exists(config.BEST_MODEL_FILE):
        print(f"[train] Best checkpoint saved to {config.BEST_MODEL_FILE}")

    model_config = {
        "vocab_size": vocab_size,
        "max_length": max_length,
        "embedding_dim": config.EMBEDDING_DIM,
        "lstm_units": config.LSTM_UNITS,
        "epochs_trained": len(history.history["loss"]),
    }
    with open(config.CONFIG_ARTIFACT_FILE, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    print(f"[train] Model config saved to {config.CONFIG_ARTIFACT_FILE}")

    history_path = os.path.join(config.EVALUATION_DIR, "training_history.json")
    os.makedirs(config.EVALUATION_DIR, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    plot_training_history(history.history)


def plot_training_history(history_dict):
    plt.figure(figsize=(7, 5))
    plt.plot(history_dict["loss"], label="Training loss")
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (sparse categorical cross-entropy)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(config.FIGURES_DIR, "training_history.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[train] Loss curve saved to {out_path}")


if __name__ == "__main__":
    main()
