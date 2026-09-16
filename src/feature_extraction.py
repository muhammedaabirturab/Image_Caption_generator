"""
Image feature extraction using a pretrained CNN (transfer learning).

Why a pretrained CNN?
    Training a CNN from scratch to recognise the objects, scenes and
    actions that show up in captions would need far more data and
    compute than a college mini-project has available. InceptionV3,
    pretrained on the 1000-class ImageNet dataset, has already learned
    a rich set of visual features (edges, textures, object parts, whole
    objects). We reuse those learned filters -- this reuse is exactly
    what "transfer learning" means.

What part of the CNN is used?
    InceptionV3 is loaded with `include_top=False, pooling="avg"`, which
    drops the final 1000-way ImageNet classification layer and instead
    global-average-pools the last convolutional feature map. The network
    itself is not fine-tuned (all layers are frozen) -- it is used purely
    as a fixed feature extractor.

What does the resulting feature represent?
    Each image is reduced to a single 2048-dimensional vector that
    summarises its visual content (objects, scene layout, colours,
    textures) in a form dense enough for the caption decoder to condition
    on, but far smaller than the raw pixel grid.

Because feature extraction is a fixed forward pass, it only needs to be
run once per image; the resulting vectors are cached to disk so that
training/evaluation never re-runs the CNN.
"""

import os
import pickle

import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from src import config
from src.data_preprocessing import load_and_resize_image


def build_encoder():
    """Return the frozen InceptionV3 feature extractor."""
    base_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    base_model.trainable = False
    return base_model


def extract_features(image_ids, images_dir=config.IMAGES_DIR, batch_size=32,
                      progress=True):
    """
    Extract a 2048-d feature vector for every image id.

    Returns: dict {image_id: np.ndarray of shape (2048,)}
    """
    encoder = build_encoder()
    features = {}

    iterator = range(0, len(image_ids), batch_size)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Extracting CNN features")

    for start in iterator:
        batch_ids = image_ids[start:start + batch_size]
        batch_arrays = []
        valid_ids = []
        for image_id in batch_ids:
            path = os.path.join(images_dir, image_id)
            if not os.path.exists(path):
                continue
            arr = load_and_resize_image(path).astype("float32")
            batch_arrays.append(arr)
            valid_ids.append(image_id)

        if not batch_arrays:
            continue

        batch = np.stack(batch_arrays, axis=0)
        batch = preprocess_input(batch)
        batch_features = encoder.predict(batch, verbose=0)

        for image_id, feature_vec in zip(valid_ids, batch_features):
            features[image_id] = feature_vec

    return features


def save_features(features, path=config.FEATURES_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(features, f)


def load_features(path=config.FEATURES_FILE):
    with open(path, "rb") as f:
        return pickle.load(f)
