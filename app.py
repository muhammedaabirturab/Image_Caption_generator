"""
Streamlit front-end for the Image Caption Generator.

Loads the already-trained model, tokenizer and CNN encoder once, then
lets the user upload an image and see a generated caption. All
training happens offline via `python -m src.train`; this app never
retrains anything.
"""

import json
import os
import pickle

import streamlit as st
from PIL import Image

from src import config
from src.caption_generator import generate_caption

st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="centered")


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained model, tokenizer, CNN encoder and config once per session."""
    missing = []
    model_path = config.BEST_MODEL_FILE if os.path.exists(config.BEST_MODEL_FILE) else config.MODEL_FILE
    if not os.path.exists(model_path):
        missing.append(f"trained model ({model_path})")
    if not os.path.exists(config.TOKENIZER_FILE):
        missing.append(f"tokenizer ({config.TOKENIZER_FILE})")
    if not os.path.exists(config.CONFIG_ARTIFACT_FILE):
        missing.append(f"model config ({config.CONFIG_ARTIFACT_FILE})")

    if missing:
        return {"error": "Missing required artifact(s): " + ", ".join(missing)}

    from tensorflow.keras.models import load_model
    from src.feature_extraction import build_encoder

    model = load_model(model_path)
    with open(config.TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)
    with open(config.CONFIG_ARTIFACT_FILE, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    encoder = build_encoder()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "encoder": encoder,
        "max_length": model_config["max_length"],
    }


def main():
    st.title("Image Caption Generator")
    st.write("Upload an image and let the deep learning model generate a caption.")

    artifacts = load_artifacts()
    if "error" in artifacts:
        st.error(
            f"{artifacts['error']}.\n\n"
            "Train the model first by running:\n\n"
            "```\npython -m src.train\n```"
        )
        return

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception:
            st.error("Could not read this file as an image. Please upload a valid JPG/PNG.")
            return

        st.image(image, caption="Uploaded image", use_column_width=True)

        with st.spinner("Generating caption..."):
            try:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image_resized = image.resize(config.CNN_INPUT_SIZE)
                import numpy as np
                image_array = np.array(image_resized)

                from src.caption_generator import preprocess_image_for_encoder
                batch = preprocess_image_for_encoder(image_array)
                feature = artifacts["encoder"].predict(batch, verbose=0)

                caption = generate_caption(
                    model=artifacts["model"],
                    encoder=artifacts["encoder"],
                    tokenizer=artifacts["tokenizer"],
                    max_length=artifacts["max_length"],
                    feature=feature,
                )
            except Exception as exc:
                st.error(f"Caption generation failed: {exc}")
                return

        st.subheader("Generated Caption")
        st.success(caption)

    st.divider()
    st.caption(
        "College mini-project: InceptionV3 (CNN encoder) + LSTM decoder, "
        "trained on the Flickr8k dataset."
    )


if __name__ == "__main__":
    main()
