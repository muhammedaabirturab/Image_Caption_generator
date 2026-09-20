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
2. Dog images are randomly downsampled to at most 1,200 (out of ~2,012).
   An earlier version of this project capped dogs at just 300 for faster
   iteration, but that turned out to be a real mistake, not just a
   speed/quality tradeoff: 300 images wasn't enough *visual diversity*
   for the model to reliably tell grass, snow, sand and dirt apart, so
   it kept defaulting to whichever setting word it had seen the
   strongest (if not even the most frequent) association with — see the
   grass/snow finding under Results. 1,200 fixes most (not all) of that
   while still capping the dominant class well below its true size.
3. All cat images are kept (there are only 23 total).
4. The result is split **80/10/10 by image** (with a minimum of 2
   images in validation/test even for the tiny cat class), so no
   image's captions or features appear in more than one split.

**Resulting dataset:**

| | Total | Train | Val | Test |
|---|---|---|---|---|
| Dog | 1,200 | 960 | 120 | 120 |
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
python -m src.train --epochs 30 --batch-size 64           # full run (~20-30 min on a laptop CPU)
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
`outputs/evaluation/training_history.json`. This is the second full
retrain of the dog/cat model — see the note on dog sample size below for
why.

**Setup:** trained on 979 unique dog/cat training images (960 dog + 19
cat, oversampled to 1,150 samples/epoch), validated on 122 images,
evaluated on a held-out 122-image test set. Vocabulary: 1,180 words. Max
caption length: 34 tokens. Embedding/LSTM size: 256. Batch size: 64.

**Training:** `EarlyStopping` (patience 5, monitoring `val_loss`)
stopped training after **10 epochs** (about 27 minutes total on this
machine's CPU), restoring the weights from **epoch 5**, which had the
lowest validation loss (see `outputs/figures/training_history.png` —
the same overfitting pattern as every other run here: training loss
keeps falling while validation loss flattens then rises).

**BLEU on the 122-image test set** (greedy decoding, corpus BLEU):

| Metric | Unrestricted (8k) run | Dog/cat, 300 dogs | Dog/cat, 1,200 dogs (current) |
|---|---|---|---|
| BLEU-1 | 0.296 | 0.714 | **0.609** |
| BLEU-2 | 0.180 | 0.522 | **0.438** |
| BLEU-3 | 0.100 | 0.333 | **0.311** |
| BLEU-4 | 0.058 | 0.215 | **0.218** |

BLEU-1/2 actually *dropped* from the 300-dog run. That's not a
regression — the 300-dog run was evaluated on only 32 test images with
a 638-word vocabulary; this run is evaluated on 122 test images (a much
more representative sample) with a 1,180-word vocabulary (more ways to
phrase something, and more ways to be scored as "wrong" for not matching
the exact reference wording). BLEU-4 essentially held steady. The real
evidence for whether the underlying problem improved is the targeted
check below, not the aggregate BLEU number.

**Targeted check — does it still say "snow" for grass?** This is what
actually motivated retraining with more dog images: a real user-uploaded
photo of a dog running on visibly green grass got captioned "a dog is
running across the snow." Investigating showed this wasn't a
word-frequency bug (in the training captions, "grass" outnumbers "snow"
3-to-1, even in the exact "running ___ the ___" phrase position) — the
300-dog model just hadn't seen enough visual variety to ground the
distinction reliably. After retraining with 4× more dog images:

| | 35 grass-only test images |
|---|---|
| Incorrectly captioned with "snow" | **5 (14%)** |

Down substantially from the original failure (which was consistent
enough that a random real-world grass photo hit it), but **not solved**
— 1 in 7 grass-only photos in this exact check still gets "snow." More
training images and/or more epochs would likely narrow this further; an
attention mechanism (Future Scope) would address it more fundamentally
by letting the decoder actually look at image regions instead of relying
on one pooled feature vector per image.

**Qualitative results — cats:** re-run on this larger model, the result
is unchanged from the 300-dog run (expected — the cat data didn't
change). All 9 unambiguous cat photos in the dataset:

| Image | Split | Generated caption | Correct? |
|---|---|---|---|
| `3354075558_3b67eaa502.jpg` | train | "A cat licks itself on a tile floor." | ✅ |
| `50030244_02cd4de372.jpg` | train | "A group of cats sit in the grass." | ✅ |
| `300222673_573fd4044b.jpg` | train | "A man plays a song on the guitar for his cat." | ✅ |
| `3229898555_16877f5180.jpg` | train | "A cat standing on carpet is interested in a piece of string..." | ✅ |
| `3539817989_5353062a39.jpg` | train | "A man kneeling on the ground surrounded by several cats." | ✅ |
| `771048251_602e5e8f45.jpg` | train | "A blonde child is being bitten on the nose by a little orange kitten." | ✅ |
| `3421480658_b3518b6819.jpg` | train | "A young girl standing next to a yellow cat on a kitchen countertop." | ✅ |
| `2506892928_7e79bec613.jpg` | train | "Three children pose among wildflowers." (mentions neither animal) | ❌ |
| **`2973269132_252bfd0160.jpg`** | **val (held out)** | **"A dog is running on a grassy field."** | **❌** |

7/9 correct, but the *only* held-out cat image (never used for a weight
update) is still captioned as a dog. This is the same conclusion as
before, now re-confirmed on a differently-trained model: the model
recognises the specific cat photos it was oversampled on, not a
generalised "cat" concept. Adding more dog images didn't and wasn't
expected to change this — cat coverage is still capped at 19 training
images. See Limitations.

## 13. Limitations

- **Setting/action words (e.g. "snow" vs. "grass") are still sometimes
  wrong even when the animal is right — verified, not assumed.** A real
  user-uploaded photo of a dog on green grass was captioned "running
  across the snow." Investigation showed this was a genuine
  visual-grounding weakness, not a word-frequency artifact ("grass"
  outnumbers "snow" 3-to-1 in the training captions, even in the exact
  phrase position generated). Retraining with 4× more dog images (300 →
  1,200) cut the error rate on a targeted check from a consistent,
  reproducible failure down to **5/35 (14%)** of grass-only test images
  still being mislabelled "snow." Meaningfully better, not solved.
  Because the model conditions on a single pooled 2048-d image vector
  with no attention over image regions, fine-grained background details
  compete with a fairly strong "frequent phrase" prior learned from the
  captions; more data narrows this gradually rather than fixing it
  outright.
- **Cat generalisation is memorisation, not true recognition — verified,
  not assumed.** Flickr8k has only 9 unambiguous cat photos and 14 more
  showing a dog and cat together. Even with 10× training-time
  oversampling, the one cat photo held out of training (never used to
  update weights) still gets captioned as a dog — the identical failure
  mode this project set out to fix. The 7/9 training-set cat images that
  *are* captioned correctly likely reflect the model having enough
  repeated exposure to recognise those specific photos, not a
  generalised visual concept of "cat." This is a genuine, demonstrated
  ceiling of Flickr8k's composition (23 cat images total in 8,091), not
  a bug in the pipeline — no amount of resampling the existing images
  manufactures new visual diversity, and unlike the dog/grass-snow case,
  simply sampling more images can't fix this because there are no more
  cat images left in Flickr8k to sample. A real fix needs actual
  additional cat photographs, which is outside Flickr8k and outside this
  project's stated scope.
- **The model's usable scope is dogs and cats only, by design.** Point
  it at any other subject and it will still emit *some* caption (usually
  defaulting to a frequent dog-scene phrase), but there is no meaningful
  training signal behind that output — it should not be trusted for
  anything outside these two classes.
- Captions are grammatically simple and can attach the wrong color or
  action to an otherwise correctly-identified animal in its correct
  setting — the model often gets *what* animal and roughly *where* right
  without correctly describing the specific action.
- Training ran on CPU within this environment; a GPU would allow larger
  dog/cat samples and more architecture experiments within the same time
  budget.
- Greedy decoding is simple and explainable but does not consider
  alternative, possibly better, word sequences the way beam search
  would; it also tends to reuse frequent training-set phrases.
- BLEU rewards n-gram overlap with the specific reference captions
  available; a caption can be reasonable and still score low if it's
  phrased differently from all 5 references.

## 14. Future Scope

- Use the full ~2,012 dog images instead of capping at 1,200, and/or
  train more epochs before early stopping — the grass/snow error rate
  dropped substantially (see Results) when dog images went from 300 to
  1,200, so more still-available data would plausibly help further.
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
concept of "cat." A second iteration, prompted by a real user-uploaded
photo getting an obviously wrong setting ("snow" for a grass field),
found that the initial 300-image dog sample (chosen purely for fast
training) was too visually narrow, and quadrupling it measurably cut —
without fully eliminating — that error. That pattern of finding a
concrete failure, verifying its actual cause with data rather than
guessing, and quantifying the fix rather than declaring victory, is the
real methodological takeaway here, more so than any single BLEU number:
it's a case study in class imbalance, sample-size-driven visual
grounding, few-shot learning, and the limits of resampling without new
data. Every stage (preprocessing, transfer-learning feature
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
