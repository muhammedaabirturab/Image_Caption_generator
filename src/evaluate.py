"""
Evaluate the trained captioning model on the held-out test split.

Computes corpus-level BLEU-1..4 (each generated caption is scored
against all of its image's reference captions) and saves a handful of
qualitative image/actual/generated examples.

BLEU (Bilingual Evaluation Understudy) measures how many n-gram
word sequences a generated caption shares with one or more reference
captions -- BLEU-1 counts single words, BLEU-4 counts 4-word
sequences, rewarding captions that are not just individually correct
words but correctly *ordered* phrases. It is not a perfect proxy for
"is this a good caption", but it is standard, cheap to compute, and
gives an objective number to track across experiments.

Usage:
    python -m src.evaluate
    python -m src.evaluate --num-examples 8 --limit 200
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from tensorflow.keras.models import load_model

from src import config
from src.caption_generator import generate_caption_tokens
from src.data_preprocessing import build_dog_cat_splits, classify_dog_cat, load_clean_captions, load_raw_captions
from src.feature_extraction import extract_features, load_features, save_features


def strip_tokens(tokens):
    return [t for t in tokens if t not in (config.START_TOKEN, config.END_TOKEN)]


def evaluate(model_path=None, num_examples=6, limit=None):
    model_path = model_path or (
        config.BEST_MODEL_FILE if os.path.exists(config.BEST_MODEL_FILE) else config.MODEL_FILE
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `python -m src.train` first."
        )
    if not os.path.exists(config.CONFIG_ARTIFACT_FILE):
        raise FileNotFoundError(
            f"Missing {config.CONFIG_ARTIFACT_FILE}. Run `python -m src.train` first."
        )

    with open(config.CONFIG_ARTIFACT_FILE, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    max_length = model_config["max_length"]

    import pickle
    with open(config.TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)

    print(f"[evaluate] Loading model from {model_path}")
    model = load_model(model_path)

    cleaned = load_clean_captions()
    raw_captions = load_raw_captions()
    _, _, test_ids, _ = build_dog_cat_splits(raw_captions)
    if limit:
        test_ids = test_ids[:limit]

    features = load_features() if os.path.exists(config.FEATURES_FILE) else {}
    missing = [i for i in test_ids if i not in features]
    if missing:
        print(f"[evaluate] Extracting CNN features for {len(missing)} test image(s)...")
        features.update(extract_features(missing))
        save_features(features)

    test_ids = [i for i in test_ids if i in features]
    if not test_ids:
        raise ValueError(
            "No test images with available features/images found. "
            "Check that data/raw/Flicker8k_Dataset contains the test split images."
        )
    print(f"[evaluate] Evaluating on {len(test_ids)} test images.")

    # Cats are scarce in this dataset (a handful of test images) -- make sure
    # they're guaranteed to show up in the qualitative figure instead of
    # being left to chance among mostly-dog examples.
    cat_test_ids = {
        image_id for image_id in test_ids
        if classify_dog_cat(cleaned.get(image_id, []))[1]
    }

    references, hypotheses = [], []
    per_image_results = {}

    for image_id in test_ids:
        feature = features[image_id]
        predicted_tokens = strip_tokens(
            generate_caption_tokens(model, tokenizer, max_length, feature)
        )
        actual_captions = [strip_tokens(c.split()) for c in cleaned.get(image_id, [])]

        references.append(actual_captions)
        hypotheses.append(predicted_tokens)
        per_image_results[image_id] = {
            "image_id": image_id,
            "actual_captions": [" ".join(c) for c in actual_captions],
            "generated_caption": " ".join(predicted_tokens),
        }

    example_ids = list(cat_test_ids)[:num_examples]
    for image_id in test_ids:
        if len(example_ids) >= num_examples:
            break
        if image_id not in example_ids:
            example_ids.append(image_id)
    qualitative_examples = [per_image_results[i] for i in example_ids]

    smoothing = SmoothingFunction().method1
    bleu_scores = {}
    for n in (1, 2, 3, 4):
        weights = tuple((1.0 / n if i < n else 0.0) for i in range(4))
        bleu_scores[f"BLEU-{n}"] = corpus_bleu(
            references, hypotheses, weights=weights, smoothing_function=smoothing
        )

    print("[evaluate] BLEU scores on test set:")
    for name, score in bleu_scores.items():
        print(f"  {name}: {score:.4f}")

    os.makedirs(config.EVALUATION_DIR, exist_ok=True)
    results = {
        "num_test_images": len(test_ids),
        "bleu_scores": bleu_scores,
        "qualitative_examples": qualitative_examples,
    }
    results_path = os.path.join(config.EVALUATION_DIR, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[evaluate] Full results saved to {results_path}")

    plot_qualitative_examples(qualitative_examples)
    plot_bleu_scores(bleu_scores)
    return results


def plot_bleu_scores(bleu_scores):
    plt.figure(figsize=(6, 4))
    names = list(bleu_scores.keys())
    values = [bleu_scores[n] for n in names]
    plt.bar(names, values, color="#4C72B0")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("BLEU Scores on Dog/Cat Test Set")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    out_path = os.path.join(config.FIGURES_DIR, "bleu_scores.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[evaluate] BLEU chart saved to {out_path}")


def plot_qualitative_examples(examples):
    if not examples:
        return
    n = len(examples)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 5 * rows))
    for idx, example in enumerate(examples):
        image_path = os.path.join(config.IMAGES_DIR, example["image_id"])
        ax = plt.subplot(rows, cols, idx + 1)
        try:
            from PIL import Image
            img = Image.open(image_path)
            ax.imshow(img)
        except Exception:
            pass
        ax.axis("off")
        actual = example["actual_captions"][0] if example["actual_captions"] else ""
        title = f"Actual: {actual}\nGenerated: {example['generated_caption']}"
        ax.set_title(title, fontsize=9, wrap=True)
    plt.tight_layout()
    out_path = os.path.join(config.FIGURES_DIR, "sample_predictions.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[evaluate] Qualitative examples figure saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained captioning model.")
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None,
                         help="Evaluate on only the first N test images (faster).")
    args = parser.parse_args()
    evaluate(num_examples=args.num_examples, limit=args.limit)
