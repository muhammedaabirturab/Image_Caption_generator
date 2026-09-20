# Dog & Cat Image Caption Generator

A college Deep Learning mini-project that generates natural-language
captions for photographs of dogs and cats, combining computer vision (a
pretrained CNN) with natural language generation (an LSTM decoder).

```
Image → CNN Encoder (InceptionV3) → Image Feature Vector → LSTM Decoder → Caption
```

## 1. Overview

Given an input photograph, the system produces a short English sentence
describing what is in it (e.g. *"A dog is running through the grass."*
or *"A cat sits alone in dry grass."*). It does this with an
**encoder-decoder** architecture: a pretrained Convolutional Neural
Network (CNN) "encodes" the image into a fixed-size feature vector, and
a Long Short-Term Memory (LSTM) network "decodes" that vector, one word
at a time, into a caption. A Streamlit web app wraps the trained model
so a user can upload a photo and see a caption generated in real time.

**Scope:** this project intentionally captions **dogs and cats only**
(see Dataset below for why). Uploading a photo of something else will
still produce a caption, but the model has no real training signal for
other subjects and its output for them shouldn't be trusted.

## 2. Objective

Automatic image captioning sits at the intersection of computer vision
and natural language processing: the model must both recognise what is
in an image and describe it in fluent, grammatical language. This
project builds a complete, working, end-to-end pipeline for this task —
data preprocessing, transfer-learning-based feature extraction, a
trainable caption-generation model, an evaluation procedure, and a
usable interface — scoped appropriately for a single-semester mini
project rather than a research system. The domain is deliberately
narrowed to two animal classes (see Dataset) so that the whole pipeline,
including its failure cases, stays small enough to fully understand and
explain.

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

This project uses a **filtered subset of Flickr8k** — dog and cat images
only.

**Why filter it down.** Flickr8k has 8,091 images, but they are not
evenly spread across subjects: ~2,012 images mention a dog, while only
**9 images are unambiguously a cat** (plus 14 more that show a dog and a
cat together). An earlier, unrestricted version of this project trained
on the full dataset and — unsurprisingly given that imbalance — learned
to caption almost anything uncertain as "a dog running through the
grass," including real photos of cats. Rather than paper over that with
a fragile fix, the project's scope was narrowed to just the two classes
Flickr8k can actually support: dogs and cats. Every training image is
now relevant to one of two classes that can be reasoned about, and the
much smaller dataset trains in minutes instead of hours.

**Construction** (`build_dog_cat_splits` in `src/data_preprocessing.py`):

1. Every Flickr8k image is classified as dog, cat, or neither, by
   keyword-matching its 5 captions (`dog(s)`/`puppy(-ies)` vs.
   `cat(s)`/`kitten(s)`). Images mentioning neither are dropped.
2. Dog images are randomly downsampled to at most 300 (out of ~2,012) —
   still far more than the cat class can match, but small enough that
   dogs don't drown out the much rarer cats.
3. All cat images are kept (there are only 23 total).
4. The result is split **80/10/10 by image** (with a minimum of 2
   images in validation/test even for the tiny cat class), so no
   image's captions or features appear in more than one split.

**Resulting dataset:**

| | Total | Train | Val | Test |
|---|---|---|---|---|
| Dog | 300 | 240 | 30 | 30 |
| Cat | 23 | 19 | 2 | 2 |

Within training only, cat images are additionally **oversampled**
(repeated up to 10× per epoch — see `oversample_cats_in_training`) so
the model sees roughly comparable dog/cat signal per epoch despite the
underlying scarcity. This only changes training-time sampling; the
validation and test splits are left untouched at their real, small
sizes so evaluation numbers aren't inflated by duplicated images.

Even with oversampling, 19 training cat images is extreme few-shot
learning by deep-learning standards — see Limitations for what this
does and doesn't fix.

The dataset is **not** committed to this repository. See
[`data/README.md`](data/README.md) for exact download instructions and
the expected directory layout — the full instructions there still
describe downloading all of Flickr8k; the dog/cat filtering happens
automatically in code when training.

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
4. **Dog/cat dataset construction** (`build_dog_cat_splits` +
   `oversample_cats_in_training` in `src/data_preprocessing.py`) — the
   full Flickr8k set is filtered down to dog/cat images and split (see
   Dataset above), then cat training images are oversampled up to 10×
   per epoch so the model gets meaningfully more gradient signal for the
   much rarer class. Disable oversampling with
   `python -m src.train --no-oversample-cats`.
5. **Model training** (`src/train.py`) — the encoder-decoder model
   (`src/model.py`) is trained with **teacher forcing**: at each
   timestep the decoder receives the *true* previous words from the
   ground-truth caption (not its own earlier predictions) and is trained
   to predict the next one. This turns caption generation into a
   supervised next-word-classification problem that trains efficiently.
6. **Caption generation / inference** (`src/caption_generator.py`) —
   starting from `startseq`, the trained decoder is applied
   **greedily**: at each step, the single most probable next word is
   picked and appended, until `endseq` is produced or `max_length` is
   reached.
7. **Evaluation** (`src/evaluate.py`) — generated captions on the
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
python -m src.train --epochs 40 --batch-size 64           # full run (dataset is small, trains in minutes)
python -m src.train --limit-images 40 --epochs 2          # quick pipeline sanity check
python -m src.train --no-oversample-cats                  # train without cat oversampling
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
on the dog/cat-filtered dataset (not fabricated) — raw numbers are in
`outputs/evaluation/evaluation_results.json` and
`outputs/evaluation/training_history.json`.

**Setup:** trained on 259 unique dog/cat training images (240 dog + 19
cat, oversampled to 430 samples/epoch), validated on 32 images,
evaluated on a held-out 32-image test set. Vocabulary: 638 words (down
from 4,478 on the unrestricted dataset — a direct result of the much
narrower domain). Max caption length: 29 tokens. Embedding/LSTM size:
256. Batch size: 64.

**Training:** `EarlyStopping` (patience 5, monitoring `val_loss`)
stopped training after **9 epochs** (about 9 minutes total on this
machine's CPU), restoring the weights from **epoch 4**, which had the
lowest validation loss (see `outputs/figures/training_history.png` —
training loss keeps falling afterwards while validation loss rises, the
same overfitting pattern seen on the full dataset, just resolved much
faster here since each epoch sees far less data).

**BLEU on the 32-image test set** (greedy decoding, corpus BLEU):

| Metric | Unrestricted (8k) run | Dog/cat-only run |
|---|---|---|
| BLEU-1 | 0.296 | **0.714** |
| BLEU-2 | 0.180 | **0.522** |
| BLEU-3 | 0.100 | **0.333** |
| BLEU-4 | 0.058 | **0.215** |

BLEU jumped substantially after narrowing the domain — expected, since
the vocabulary and scene variety the model has to cover shrank a lot.
These two numbers aren't a fully apples-to-apples comparison (different
test sets, different sizes), but the qualitative results below tell the
more honest story.

**Qualitative results — dogs:** on the held-out dog test images, the
model reliably identifies "dog" and a plausible action/setting, e.g. a
real test example: actual *"a beige and dark brown dog plays in the
swimming pool with his mouth open"* → generated *"a black dog is running
in the snow"* — right subject and register, wrong specific action/color
(no image evidence is actually used to distinguish these; the decoder
still leans on frequent phrase patterns rather than describing what's
uniquely different about each photo). See
`outputs/figures/sample_predictions.png`.

**Qualitative results — cats (the actual point of this exercise):**
this is the more important test, since it's the failure mode that
motivated restricting the dataset in the first place. Testing directly
on all 9 unambiguous cat photos in the dataset:

| Image (in training set) | Generated caption |
|---|---|
| `300222673_573fd4044b.jpg` | "A man plays his yellow guitar while staring at his cat." |
| `771048251_602e5e8f45.jpg` | "A orange kitten biting the nose of a child." |
| `3539817989_5353062a39.jpg` | "A man kneeling on the ground surrounded by several cats." |
| `50030244_02cd4de372.jpg` | "A cat sits alone in dry grass." |
| `3229898555_16877f5180.jpg` | "A white and brown cat bats at a frayed string dangling in front of him." |
| `3354075558_3b67eaa502.jpg` | "A blackstriped cat is looking at a cord it has pinned with a paw." |
| `3421480658_b3518b6819.jpg` | "A girl reaches up to kiss a cat which is sitting on the counter." |
| **`2973269132_252bfd0160.jpg` (held-out, val split)** | **"A dog is running on the grass."** |
| `2506892928_7e79bec613.jpg` (train) | "Three children play in the garden." (doesn't mention either animal) |

7 of the 8 training-set cat images now correctly say "cat"/"kitten" —
a real improvement over the unrestricted model, which said "dog" for
every cat photo it was given, including ones from its own training set.
**But the one cat image that was genuinely held out of training
(validation split) still gets captioned "a dog is running on the
grass"** — the exact old failure mode. With only 19 cat training images,
even after 10× oversampling, the model has enough repeated exposure to
recognise *those specific photos* but not enough visual diversity to
reliably generalise the concept "cat" to a new one. This is discussed
further in Limitations.

## 13. Limitations

- **Cat generalisation is memorisation, not true recognition — verified,
  not assumed.** Flickr8k has only 9 unambiguous cat photos and 14 more
  showing a dog and cat together. Even with 10× training-time
  oversampling, the one cat photo held out of training (never used to
  update weights) still gets captioned "a dog is running on the grass" —
  the identical failure mode this project set out to fix. The 7/8
  training-set cat images that *are* now captioned correctly likely
  reflect the model having enough repeated exposure to recognise those
  specific photos, not a generalised visual concept of "cat." This is a
  genuine, demonstrated ceiling of Flickr8k's composition (23 cat images
  total in 8,091), not a bug in the pipeline — no amount of resampling
  the existing images manufactures new visual diversity. A real fix
  needs actual additional cat photographs, which is outside Flickr8k and
  outside this project's stated scope.
- **The model's usable scope is dogs and cats only, by design.** Point
  it at any other subject and it will still emit *some* caption (usually
  defaulting to a frequent dog-scene phrase), but there is no meaningful
  training signal behind that output — it should not be trusted for
  anything outside these two classes.
- Uploading a genuinely different-looking dog or cat than anything in
  Flickr8k's ~300 sampled dog / 19 trained cat images can still get a
  generic or mismatched caption — 300/19 images is a small sample even
  within just two classes.
- Captions are grammatically simple and can attach the wrong color,
  action, or setting to an otherwise correctly-identified animal (see
  the swimming-pool example in Results) — the model often gets *what*
  animal it is right without correctly describing *what it's doing*.
- Training ran on CPU within this environment; a GPU would allow more
  experimentation (larger cat samples, attention, etc.) within the same
  time budget.
- Greedy decoding is simple and explainable but does not consider
  alternative, possibly better, word sequences the way beam search
  would; it also tends to reuse frequent training-set phrases.
- BLEU rewards n-gram overlap with the specific reference captions
  available; with only a 32-image test set here, BLEU is a noisier
  signal than on the full 1,000-image split, and single outliers move it
  more.

## 14. Future Scope

- Add real additional cat (and other animal) photos beyond Flickr8k so
  oversampling has genuine visual diversity to work with, not just
  repeats of the same handful of images.
- Add an attention mechanism so the decoder can focus on different image
  regions per word, which should also help distinguish specific actions
  and colors, not just the animal class.
- Explore Transformer-based captioning architectures.
- Implement beam search decoding.
- Extend to more animal classes once each has reasonable Flickr8k
  representation, or move to a larger/more balanced dataset (MS-COCO,
  Flickr30k) for general-purpose (non-animal-restricted) captioning.

## 15. Conclusion

This project implements a complete, working image captioning pipeline —
from raw dataset to a usable web interface — using a pretrained CNN
encoder and an LSTM decoder trained with teacher forcing on a
dog/cat-filtered subset of Flickr8k. The dataset was deliberately
narrowed to two animal classes after an unrestricted first attempt
demonstrated a genuine class-imbalance failure (defaulting to "dog" for
every uncertain image, cats included); restricting scope made the
problem small enough to solve within Flickr8k's real limits, and testing
directly on a genuinely held-out cat photo shows the fix is partial —
correct for training-set-adjacent cats, but not yet a generalised
concept of "cat." That honest, verified result, not a polished-looking
BLEU number, is the actual takeaway suitable for a viva discussion of
class imbalance, few-shot learning, and the limits of resampling without
new data. Every stage (preprocessing, transfer-learning feature
extraction, model training, greedy decoding, BLEU evaluation, and a
Streamlit demo) is implemented, tested, and reproducible from the raw
dataset.

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
