"""
Preprocessing utilities for captions and images.

Caption pipeline:
    raw token file -> per-image list of captions -> lowercase / clean text
    -> add startseq/endseq -> build vocabulary -> Keras Tokenizer

Image pipeline:
    load image -> convert to RGB -> resize to CNN input size -> apply the
    CNN's expected preprocessing (see feature_extraction.py).
"""

import json
import os
import re
import string
from collections import defaultdict

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer

from src import config


# ---------------------------------------------------------------------------
# Caption loading and cleaning
# ---------------------------------------------------------------------------
def load_raw_captions(captions_file=config.CAPTIONS_FILE):
    """
    Parse the Flickr8k token file into {image_id: [caption, caption, ...]}.

    Each line in Flickr8k.token.txt looks like:
        1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing...
    """
    mapping = defaultdict(list)
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_tag, caption = line.split("\t")
            image_id = image_tag.split("#")[0]
            mapping[image_id].append(caption)
    return mapping


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_caption(text):
    """
    Lowercase, strip punctuation/digits, normalize whitespace.

    Example:
        "A dog is running through the grass ." -> "a dog is running through the grass"
    """
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\d+", "", text)
    words = [w for w in text.split() if len(w) > 1 or w == "a"]
    return " ".join(words)


def clean_captions_mapping(raw_mapping):
    """Clean every caption and wrap it with start/end tokens."""
    cleaned = {}
    for image_id, captions in raw_mapping.items():
        cleaned_list = []
        for cap in captions:
            cleaned_text = clean_caption(cap)
            wrapped = f"{config.START_TOKEN} {cleaned_text} {config.END_TOKEN}"
            cleaned_list.append(wrapped)
        cleaned[image_id] = cleaned_list
    return cleaned


def save_clean_captions(cleaned_mapping, path=config.CLEAN_CAPTIONS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned_mapping, f)


def load_clean_captions(path=config.CLEAN_CAPTIONS_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataset splits (train / validation / test, no image leakage)
# ---------------------------------------------------------------------------
def load_split_ids(split_file):
    """Read one of Flickr_8k.{train,dev,test}Images.txt into a list of ids."""
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_splits():
    """
    Return (train_ids, val_ids, test_ids) using the dataset's official
    partition (~6000 / 1000 / 1000 images -> 75/12.5/12.5, close to the
    target 80/10/10 split). Splitting is done by image id, so no single
    image's captions or features ever appear in more than one split.
    """
    train_ids = load_split_ids(config.TRAIN_SPLIT_FILE)
    val_ids = load_split_ids(config.DEV_SPLIT_FILE)
    test_ids = load_split_ids(config.TEST_SPLIT_FILE)
    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Dog/cat-only dataset scope
# ---------------------------------------------------------------------------
# This project intentionally restricts itself to two animal classes: dogs
# and cats. Flickr8k has ~2,012 dog images but only ~9 pure cat images (plus
# ~14 images showing both) out of 8,091 total -- with the full, unrestricted
# dataset, a small mini-project model kept defaulting to "a dog is running
# through the grass" for anything it was unsure about, including real cat
# photos, because "dog" was such an overwhelming prior. Narrowing the whole
# project to just dogs vs. cats keeps every training image relevant to one
# of two classes that can be reasoned about and explained clearly, and
# trains fast enough to iterate on with limited compute.
DOG_KEYWORDS = ["dog", "dogs", "puppy", "puppies"]
CAT_KEYWORDS = ["cat", "cats", "kitten", "kittens"]


def _matches_any(text, keywords):
    return any(re.search(rf"\b{kw}\b", text) for kw in keywords)


def classify_dog_cat(captions_for_image):
    """Return (is_dog, is_cat) found in one image's captions (not exclusive)."""
    text = " ".join(captions_for_image).lower()
    return _matches_any(text, DOG_KEYWORDS), _matches_any(text, CAT_KEYWORDS)


def _split_ids(ids, val_frac=0.1, test_frac=0.1, min_each=2):
    """Split a shuffled id list into train/val/test, guaranteeing at least
    `min_each` ids in val/test when there are enough ids to spare."""
    n = len(ids)
    n_val = max(min_each, round(n * val_frac)) if n > 2 * min_each else min(min_each, n // 3)
    n_test = max(min_each, round(n * test_frac)) if n > 2 * min_each else min(min_each, n // 3)
    val = ids[:n_val]
    test = ids[n_val:n_val + n_test]
    train = ids[n_val + n_test:]
    return train, val, test


def build_dog_cat_splits(raw_captions, dog_sample_size=config.DOG_SAMPLE_SIZE,
                          seed=config.RANDOM_SEED):
    """
    Build a small dog/cat-only dataset from Flickr8k and split it into
    train/val/test. Images mentioning neither a dog nor a cat are dropped
    entirely. Dog images are randomly downsampled to `dog_sample_size`
    (Flickr8k has far more dog images than are needed, and they would
    otherwise still dominate). All cat images are kept (and treated as the
    "cat" class even if a dog also appears in the same photo) since there
    are only a handful in the whole dataset.

    Returns (train_ids, val_ids, test_ids, stats).
    """
    import random as _random
    rng = _random.Random(seed)

    dog_ids, cat_ids = [], []
    for image_id, caps in raw_captions.items():
        is_dog, is_cat = classify_dog_cat(caps)
        if is_cat:
            cat_ids.append(image_id)
        elif is_dog:
            dog_ids.append(image_id)

    rng.shuffle(dog_ids)
    dog_ids = dog_ids[:dog_sample_size]
    rng.shuffle(cat_ids)

    dog_train, dog_val, dog_test = _split_ids(dog_ids)
    cat_train, cat_val, cat_test = _split_ids(cat_ids)

    train_ids = dog_train + cat_train
    val_ids = dog_val + cat_val
    test_ids = dog_test + cat_test
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    stats = {
        "dog_total": len(dog_ids), "cat_total": len(cat_ids),
        "dog_train": len(dog_train), "dog_val": len(dog_val), "dog_test": len(dog_test),
        "cat_train": len(cat_train), "cat_val": len(cat_val), "cat_test": len(cat_test),
    }
    return train_ids, val_ids, test_ids, stats


def oversample_cats_in_training(train_ids, raw_captions,
                                 cap=config.CAT_OVERSAMPLE_CAP, seed=config.RANDOM_SEED):
    """
    Duplicate cat training images so they get comparable per-epoch exposure
    to the (more numerous) dog training images, capped at `cap` repeats.
    Only ever applied to the training split.
    """
    import random as _random
    rng = _random.Random(seed)

    dog_count, cat_ids = 0, []
    for image_id in train_ids:
        is_dog, is_cat = classify_dog_cat(raw_captions.get(image_id, []))
        if is_cat:
            cat_ids.append(image_id)
        elif is_dog:
            dog_count += 1

    if not cat_ids:
        return list(train_ids)

    factor = min(cap, max(1, round(dog_count / len(cat_ids)))) if dog_count else 1
    resampled = list(train_ids) + cat_ids * (factor - 1)
    rng.shuffle(resampled)
    return resampled


# ---------------------------------------------------------------------------
# Vocabulary / tokenizer
# ---------------------------------------------------------------------------
def build_tokenizer(cleaned_mapping, image_ids, max_vocab=config.MAX_VOCAB_SIZE,
                     min_freq=config.MIN_WORD_FREQUENCY):
    """
    Fit a Keras Tokenizer on the captions of the given image ids only
    (normally the training split) so the vocabulary reflects real
    training-time frequencies.
    """
    all_captions = []
    for image_id in image_ids:
        all_captions.extend(cleaned_mapping.get(image_id, []))

    # First pass: raw word frequencies to enforce min_freq.
    freq = defaultdict(int)
    for caption in all_captions:
        for word in caption.split():
            freq[word] += 1
    vocab_words = {w for w, c in freq.items() if c >= min_freq}

    filtered_captions = []
    for caption in all_captions:
        words = [w for w in caption.split() if w in vocab_words]
        filtered_captions.append(" ".join(words))

    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<unk>")
    tokenizer.fit_on_texts(filtered_captions)
    return tokenizer


def max_caption_length(cleaned_mapping, image_ids):
    lengths = [
        len(caption.split())
        for image_id in image_ids
        for caption in cleaned_mapping.get(image_id, [])
    ]
    return max(lengths) if lengths else config.GREEDY_MAX_LENGTH_FALLBACK


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_and_resize_image(image_path, target_size=config.CNN_INPUT_SIZE):
    """Load an image from disk, force RGB, resize for the CNN encoder."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    return np.array(img)
