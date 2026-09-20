"""
Central configuration for the Image Caption Generator project.

All paths and hyperparameters used across preprocessing, training,
evaluation and inference are defined here so they only need to be
changed in one place.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

IMAGES_DIR = os.path.join(RAW_DIR, "Flicker8k_Dataset")
CAPTIONS_FILE = os.path.join(RAW_DIR, "Flickr8k.token.txt")
TRAIN_SPLIT_FILE = os.path.join(RAW_DIR, "Flickr_8k.trainImages.txt")
DEV_SPLIT_FILE = os.path.join(RAW_DIR, "Flickr_8k.devImages.txt")
TEST_SPLIT_FILE = os.path.join(RAW_DIR, "Flickr_8k.testImages.txt")

CLEAN_CAPTIONS_FILE = os.path.join(PROCESSED_DIR, "captions_clean.json")
FEATURES_FILE = os.path.join(PROCESSED_DIR, "image_features.pkl")
TOKENIZER_FILE = os.path.join(PROCESSED_DIR, "tokenizer.pkl")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "caption_model.keras")
BEST_MODEL_FILE = os.path.join(MODELS_DIR, "caption_model_best.keras")
CONFIG_ARTIFACT_FILE = os.path.join(MODELS_DIR, "model_config.json")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
EVALUATION_DIR = os.path.join(OUTPUTS_DIR, "evaluation")

# ---------------------------------------------------------------------------
# Image / CNN encoder settings
# ---------------------------------------------------------------------------
CNN_INPUT_SIZE = (299, 299)   # required input size for InceptionV3
CNN_FEATURE_DIM = 2048        # output dimension of InceptionV3 pooled features

# ---------------------------------------------------------------------------
# Text / vocabulary settings
# ---------------------------------------------------------------------------
START_TOKEN = "startseq"
END_TOKEN = "endseq"
MAX_VOCAB_SIZE = 5000          # cap on number of most-frequent words kept
MIN_WORD_FREQUENCY = 2         # words appearing fewer times are dropped

# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 256
LSTM_UNITS = 256
DROPOUT_RATE = 0.4
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
VALIDATION_SPLIT_SEED = 42

# Decoding
GREEDY_MAX_LENGTH_FALLBACK = 34  # used only if max_length is not available

# Reproducibility
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Dog/cat-only dataset scope (see src/data_preprocessing.py:build_dog_cat_splits
# and the README's Dataset section for why the project is scoped this way).
# ---------------------------------------------------------------------------
DOG_SAMPLE_SIZE = 300     # randomly keep at most this many of Flickr8k's ~2012 dog images
CAT_OVERSAMPLE_CAP = 10   # never repeat a single cat training image more than this
