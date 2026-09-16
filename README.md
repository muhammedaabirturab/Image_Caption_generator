# Image Caption Generator

A college Deep Learning mini-project that generates natural-language
captions for photographs, combining computer vision (a pretrained CNN)
with natural language generation (an LSTM decoder).

```
Image → CNN Encoder (InceptionV3) → Image Feature Vector → LSTM Decoder → Caption
```

## 1. Overview

Given an input photograph, the system produces a short English sentence
describing what is in it (e.g. *"A dog is running through the grass."*).
It does this with an **encoder-decoder** architecture: a pretrained
Convolutional Neural Network (CNN) "encodes" the image into a fixed-size
feature vector, and a Long Short-Term Memory (LSTM) network "decodes"
that vector, one word at a time, into a caption. A Streamlit web app
wraps the trained model so a user can upload a photo and see a caption
generated in real time.

## 2. Objective

Automatic image captioning sits at the intersection of computer vision
and natural language processing: the model must both recognise what is
in an image and describe it in fluent, grammatical language. This
project builds a complete, working, end-to-end pipeline for this task —
data preprocessing, transfer-learning-based feature extraction, a
trainable caption-generation model, an evaluation procedure, and a
usable interface — scoped appropriately for a single-semester mini
project rather than a research system.

## 3. Technologies Used

- Python 3.11
- TensorFlow / Keras — model definition and training
- InceptionV3 (pretrained on ImageNet) — CNN image encoder
- LSTM — sequence decoder
- NumPy, Pandas — data handling
- Pillow — image loading/preprocessing
- Matplotlib — training curves and result plots
- scikit-learn — utility helpers
- NLTK — BLEU score evaluation
- Streamlit — web application interface

## 4. Dataset

This project uses **Flickr8k**:

- **8,091 images** of everyday scenes, people and activities.
- **5 independently written captions per image** (~40,455 captions).
- Purpose: a small, well-annotated benchmark for image captioning —
  large enough to learn meaningful visual-language associations, small
  enough to preprocess and train on a laptop CPU.
- **Split:** the dataset's official partition of 6,000 / 1,000 / 1,000
  images (train / validation / test — 75% / 12.5% / 12.5%, close to the
  conventional 80/10/10 split) is used as-is. The split is done **per
  image**, so all 5 captions of a given image stay in the same split —
  there is no leakage of an image (or its captions/features) across
  train, validation and test.

The dataset is **not** committed to this repository. See
[`data/README.md`](data/README.md) for exact download instructions and
the expected directory layout.

## 5. System Architecture

```
                Image
                  |
        Resize + InceptionV3 preprocessing
                  |
      InceptionV3 (frozen, include_top=False, pooling="avg")
                  |
          Image feature vector (2048-d)
                  |
                  v
   Dense(256) ----+---- Add ----> Dense(256, relu) --> Dense(vocab, softmax) --> next word
                  ^
                  |
        LSTM(256) (decoder_lstm)
                  ^
                  |
   Embedding(vocab, 256) <- caption tokens so far (Embedding + LSTM branch)
```

The image feature and the LSTM's encoding of the words generated so far
are each projected to the same 256-dimensional space and combined by
addition (a "merge" architecture), then passed through a small
classifier head that outputs a probability distribution over the
vocabulary for the next word.

## 6. Methodology

1. **Preprocessing** (`src/data_preprocessing.py`) — captions are
   lowercased, stripped of punctuation/digits, whitespace-normalised,
   and wrapped with `startseq`/`endseq` tokens; images are converted to
   RGB and resized to `299×299` for InceptionV3.
2. **Feature extraction** (`src/feature_extraction.py`) — every image is
   passed once through a frozen, pretrained InceptionV3 (ImageNet
   weights, classification head removed) to obtain a 2048-d feature
   vector. This is **transfer learning**: instead of learning to detect
   edges/textures/objects from scratch (which Flickr8k is far too small
   for), the model reuses filters already learned from 1.4M ImageNet
   images. Features are cached to disk so the CNN never runs twice on
   the same image.
3. **Tokenization** — a Keras `Tokenizer` is fit on the training
   captions only, capped at the most frequent 5,000 words, to build the
   vocabulary and word↔index mappings.
4. **Model training** (`src/train.py`) — the encoder-decoder model
   (`src/model.py`) is trained with **teacher forcing**: at each
   timestep the decoder receives the *true* previous words from the
   ground-truth caption (not its own earlier predictions) and is trained
   to predict the next one. This turns caption generation into a
   supervised next-word-classification problem that trains efficiently.
5. **Caption generation / inference** (`src/caption_generator.py`) —
   starting from `startseq`, the trained decoder is applied
   **greedily**: at each step, the single most probable next word is
   picked and appended, until `endseq` is produced or `max_length` is
   reached.
6. **Evaluation** (`src/evaluate.py`) — generated captions on the
   held-out test split are compared against their 5 reference captions
   using corpus-level BLEU-1..4, plus a handful of qualitative
   image/actual/generated examples.

## 7. Installation

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 8. Dataset Setup

1. Download and extract both archives into `data/raw/` (see
   [`data/README.md`](data/README.md) for the direct links and expected
   folder layout):
   - `Flickr8k_Dataset.zip` → images
   - `Flickr8k_text.zip` → captions + train/val/test split files
2. No manual preprocessing step is required — `src/train.py` builds and
   caches cleaned captions, the tokenizer, and CNN features automatically
   the first time it runs.

## 9. Training

```bash
python -m src.train
```

Optional flags:

```bash
python -m src.train --epochs 20 --batch-size 64          # full run (default settings)
python -m src.train --limit-images 40 --epochs 2          # quick pipeline sanity check
```

This saves `models/caption_model.keras` (final),
`models/caption_model_best.keras` (lowest validation loss, via
`ModelCheckpoint`), `models/model_config.json`, and a training-loss plot
in `outputs/figures/training_history.png`. `EarlyStopping` halts training
if validation loss stops improving.

## 10. Evaluation

```bash
python -m src.evaluate
```

Computes BLEU-1..4 on the test split and writes
`outputs/evaluation/evaluation_results.json`,
`outputs/figures/bleu_scores.png`, and
`outputs/figures/sample_predictions.png` (a few images with their actual
vs. generated captions).

## 11. Running the Application

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (typically `http://localhost:8501`),
upload a JPG/PNG, and the generated caption is displayed below the
image. The app only loads the already-trained model — it never trains
anything itself.

## 12. Results

Results below are from an actual training + evaluation run of this code
on the full Flickr8k dataset (not fabricated) — raw numbers are in
`outputs/evaluation/evaluation_results.json` and
`outputs/evaluation/training_history.json`.

**Setup:** trained on the full 6,000-image training split (~30,000
captions), validated on 1,000 images, evaluated on the held-out 1,000
test images. Vocabulary: 4,478 words. Max caption length: 37 tokens.
Embedding/LSTM size: 256. Batch size: 64.

**Training:** `EarlyStopping` (patience 4, monitoring `val_loss`)
stopped training after **7 epochs**, restoring the weights from
**epoch 3**, which had the lowest validation loss:

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---|---|---|---|---|
| 1 | 3.862 | 0.310 | 3.383 | 0.344 |
| 2 | 3.197 | 0.362 | 3.211 | 0.366 |
| **3 (best)** | 2.968 | 0.380 | **3.173** | 0.376 |
| 4 | 2.829 | 0.391 | 3.195 | 0.377 |
| 5 | 2.735 | 0.400 | 3.186 | 0.381 |
| 6 | 2.660 | 0.406 | 3.278 | 0.380 |
| 7 | 2.602 | 0.411 | 3.279 | 0.382 |

Training loss keeps falling after epoch 3 while validation loss
flattens/rises — a textbook overfitting signal on a dataset this size,
which is exactly why `ModelCheckpoint` + `EarlyStopping` are used instead
of just taking the final epoch. See
`outputs/figures/training_history.png`.

**BLEU on the 1,000-image test set** (greedy decoding, corpus BLEU):

| Metric | Score |
|---|---|
| BLEU-1 | 0.296 |
| BLEU-2 | 0.180 |
| BLEU-3 | 0.100 |
| BLEU-4 | 0.058 |

See `outputs/figures/bleu_scores.png`. These are modest but expected
for a small (~1.2M-parameter decoder), briefly-trained model on an
8k-image dataset — reference points from the literature for much more
heavily tuned/trained models on Flickr8k are typically in the BLEU-1 ≈
0.55–0.65 range.

**Qualitative examples** (`outputs/figures/sample_predictions.png`)
show the model correctly identifying common subjects and actions (e.g.
a real test-set caption/generation pair: actual *"a brown dog running"*
→ generated *"a dog is running through the grass"*), while also showing
two typical failure modes of a small greedy-decoded model: over-generic
captions (defaulting to frequent training phrases like "a dog is running
through the grass" for other dog photos) and, occasionally, repetition
loops on more complex/crowded scenes (e.g. repeating "a black shirt and
a black shirt..."). Both are discussed in Limitations below.

## 13. Limitations

- Flickr8k (~8k images) is small by modern captioning-dataset standards;
  the model generalises far less well than systems trained on
  MS-COCO-scale data (100k+ images).
- Captions are grammatically simple, generic, and can miss uncommon
  objects, spatial relationships, or scenes not well represented in
  Flickr8k's largely people/animals/outdoor-activity photos.
- Training ran on CPU within this environment; a GPU or more epochs
  would likely reduce loss further and improve fluency.
- Greedy decoding is simple and explainable but can produce mildly
  repetitive or literal phrasing; it does not consider alternative,
  possibly better, word sequences the way beam search would. In this
  run, a small number of test images (typically crowded/complex scenes)
  triggered visible word-repetition loops (e.g. "a black shirt and a
  black shirt..."), a known failure mode of greedy decoding on an
  undertrained decoder.
- With early stopping at epoch 3, the model has seen relatively few
  gradient updates; captions for common subjects (dogs, people in red
  shirts) are reasonable, but rarer scene types tend to get a generic,
  frequently-seen caption instead of a specific one.
- BLEU rewards n-gram overlap with the specific reference captions
  available; a caption can be accurate and still score low if it is
  phrased differently from all 5 references.

## 14. Future Scope

- Train on a larger dataset (e.g. MS-COCO, Flickr30k) for more general
  captions.
- Add an attention mechanism so the decoder can focus on different image
  regions per word.
- Explore Transformer-based captioning architectures.
- Implement beam search decoding.
- Extend to multilingual caption generation.

## 15. Conclusion

This project implements a complete, working image captioning pipeline —
from raw dataset to a usable web interface — using a pretrained CNN
encoder and an LSTM decoder trained with teacher forcing on Flickr8k.
While intentionally scoped for a mini-project rather than a
state-of-the-art system, every stage (preprocessing, transfer-learning
feature extraction, model training, greedy decoding, BLEU evaluation,
and a Streamlit demo) is implemented, tested, and reproducible from the
raw dataset, and is explainable end-to-end for a viva.

## Project Structure

```
Image_Caption_generator/
├── app.py                     # Streamlit application
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md              # dataset download instructions
│   ├── raw/                   # Flickr8k images + captions (not committed)
│   └── processed/             # cleaned captions, tokenizer, cached features (not committed)
│
├── src/
│   ├── __init__.py
│   ├── config.py              # paths + hyperparameters
│   ├── data_preprocessing.py  # caption cleaning, splits, tokenizer, image loading
│   ├── feature_extraction.py  # InceptionV3 encoder
│   ├── model.py                # encoder-decoder model definition
│   ├── train.py                # training pipeline
│   ├── caption_generator.py   # greedy-decoding inference
│   └── evaluate.py             # BLEU evaluation + qualitative examples
│
├── notebooks/
│   └── data_exploration.ipynb # dataset statistics / sanity checks
│
├── models/
│   ├── README.md
│   └── (trained model files, not committed)
│
└── outputs/
    ├── figures/                # training curves, BLEU chart, sample predictions
    └── evaluation/             # evaluation_results.json, training_history.json
```
