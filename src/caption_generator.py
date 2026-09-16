"""
Inference: turn a single image into a natural-language caption.

Pipeline:
    image -> load + preprocess -> InceptionV3 feature (2048-d)
          -> greedy decoding loop with the trained LSTM decoder
          -> readable caption (startseq/endseq stripped)

Decoding strategy: greedy search. At every step we take the single
most probable next word given the image feature and the words
generated so far, append it, and feed the extended sequence back in.
Decoding stops when the model predicts `endseq` or when `max_length`
is reached. Greedy decoding is simple to explain and reason about,
which matters more for a college mini-project than the small quality
gain beam search would add.
"""

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src import config
from src.data_preprocessing import load_and_resize_image


def preprocess_image_for_encoder(image_array):
    """Apply InceptionV3's expected preprocessing to a loaded RGB image array."""
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    arr = image_array.astype("float32")
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


def extract_single_feature(image_path, encoder):
    image_array = load_and_resize_image(image_path)
    batch = preprocess_image_for_encoder(image_array)
    feature = encoder.predict(batch, verbose=0)
    return feature  # shape (1, 2048)


def generate_caption_tokens(model, tokenizer, max_length, feature):
    """
    Greedy-decode a caption for a precomputed feature vector.
    Returns the raw token list, including startseq/endseq.
    """
    feature = np.array(feature).reshape(1, config.CNN_FEATURE_DIM)
    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

    caption_tokens = [config.START_TOKEN]
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([" ".join(caption_tokens)])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        predictions = model.predict([feature, sequence], verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_word = index_to_word.get(predicted_index)

        if predicted_word is None:
            break
        caption_tokens.append(predicted_word)
        if predicted_word == config.END_TOKEN:
            break

    return caption_tokens


def generate_caption(model, encoder, tokenizer, max_length, image_path=None,
                      feature=None):
    """
    Generate a readable caption for an image.

    Provide either `image_path` (a file on disk) or a precomputed
    `feature` vector of shape (1, 2048) / (2048,).
    """
    if feature is None:
        if image_path is None:
            raise ValueError("Provide either image_path or feature.")
        feature = extract_single_feature(image_path, encoder)

    caption_tokens = generate_caption_tokens(model, tokenizer, max_length, feature)
    return clean_generated_caption(caption_tokens)


def clean_generated_caption(tokens):
    """Strip startseq/endseq tokens and return a readable sentence."""
    words = [w for w in tokens if w not in (config.START_TOKEN, config.END_TOKEN)]
    sentence = " ".join(words).strip()
    if sentence:
        sentence = sentence[0].upper() + sentence[1:] + "."
    return sentence
