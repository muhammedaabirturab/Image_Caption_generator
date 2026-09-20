# Data Directory

This project uses the **Flickr8k** dataset. The dataset itself is **not**
committed to this repository (it is listed in `.gitignore`) because it is
too large and is not ours to redistribute. Follow the steps below to set
it up locally.

## Directory layout expected by the code

```
data/
├── raw/
│   ├── Flicker8k_Dataset/          # 8,091 JPEG images
│   ├── Flickr8k.token.txt          # 5 raw captions per image
│   ├── Flickr_8k.trainImages.txt   # 6,000 training image ids
│   ├── Flickr_8k.devImages.txt     # 1,000 validation image ids
│   └── Flickr_8k.testImages.txt    # 1,000 test image ids
└── processed/                      # created automatically by src/train.py
    ├── captions_clean.json
    ├── tokenizer.pkl
    └── image_features.pkl
```

## How to obtain the dataset

Download the two archives and extract them both into `data/raw/`:

- Images: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
- Captions/splits: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

After extracting, `data/raw/` should contain the folder `Flicker8k_Dataset/`
(note the original dataset's spelling, without the second "r") plus the
`.txt` files listed above.

## About the dataset

- **Images:** 8,091 photographs of everyday scenes and activities.
- **Captions:** 5 independently written English captions per image
  (~40,455 captions total), describing the main people/objects/actions.
- **Purpose:** a small, clean benchmark for image captioning research —
  large enough to learn generalizable visual-language associations, small
  enough to preprocess and train on a single laptop.
- **Split:** the dataset ships with an official partition of
  6,000 / 1,000 / 1,000 images for train / validation / test
  (75% / 12.5% / 12.5%, close to the commonly used 80/10/10 split). This
  project does **not** use that official split directly — see below.

## Dog/cat filtering (this project's actual scope)

Download and extract the full Flickr8k dataset as described above — that
part doesn't change. But `src/train.py` and `src/evaluate.py` don't use
Flickr8k's full 8,091 images or its official split file. Instead,
`build_dog_cat_splits()` in `src/data_preprocessing.py` filters the
dataset down to dog and cat images only (Flickr8k is extremely
imbalanced: ~2,012 dog images vs. only 23 cat images — see the main
[README](../README.md#4-dataset) for why), caps dogs at 300, keeps all
23 cat images, and builds a fresh 80/10/10 split from just that subset.
This happens automatically in code — no extra manual step is needed
beyond downloading the full dataset above. The split is still done **by
image**, so no image or caption leaks between train, validation and
test.

## Processed artifacts

`python -m src.train` will automatically create everything under
`data/processed/` the first time it runs:

- `captions_clean.json` — lowercased, cleaned captions with `startseq`/`endseq`.
- `tokenizer.pkl` — the Keras `Tokenizer` fit on the training captions.
- `image_features.pkl` — cached 2048-d InceptionV3 feature vectors, so the
  CNN forward pass only ever runs once per image.

`captions_clean.json` and `tokenizer.pkl` **are committed** (they're
small and the app needs the tokenizer to run without a full local
retrain). `image_features.pkl` is excluded from git — it's a large
(~60MB), easily-regenerated cache, not needed for inference.
