# Postcard Colour Classification

Colour analysis pipeline for the BiblioTech Hackathon 2026 (Team 4 — Inked and Stamped).

## Overview

This module classifies 35,930 Belgian historical postcard images into 8 fine-grained colour categories using a combination of feature extraction, supervised classification, unsupervised classification, and post-processing rules.

## Final categories

| Category | Description |
|---|---|
| `bw` | Black-and-white |
| `sepia` | Sepia / warm brown tones |
| `color_photo` | Real colour photograph |
| `color_handcolored` | Hand-coloured image |
| `monotone_blue` | Blue monotone print |
| `monotone_green` | Green monotone print |
| `monotone_purple` | Purple monotone print |
| `monotone_red` | Red monotone print |

## Pipeline

**Step 1 — Feature extraction** (`02_src/01_extract_colour_features.py`)
Extracts 50 colour features per image using OpenCV and NumPy, including saturation distribution, hue diversity, sepia pixel ratio, and opponent colour channels.

**Step 2 — Metadata-guided classification** (`02_src/02_metadata_color_classification.py`)
Trains a Random Forest classifier on postcards with existing metadata colour labels. Predicts labels for unlabelled postcards. Produces three broad categories: black-and-white, sepia, colour.

**Step 3 — Fine-grained classification** (`02_src/03_fine_grained_classification.py`)
Applies GMM sub-clustering within each broad category to produce six sub-categories. Post-processing rules then identify four monotone printing styles from the existing metadata labels (Blue, Green, Purple, Red), consolidating them into a consistent set of eight final categories.

## Notebooks

- `03_notebook/01_post_processing.ipynb` — applies and validates post-processing rules
- `03_notebook/02_colour_analysis_insights.ipynb` — explores colour distribution by decade, city, and publisher

## Data

- `01_data/00_metadata/20230301-Postcards.csv` — original metadata
- `01_data/01_processed/postcard_fine_labels.csv` — final colour labels for all 35,930 postcards

## Key findings

- Colour photographs increase sharply from the 1960s onwards
- Monotone printing styles (blue, green, purple, red) were present in the original metadata but inconsistently labelled — this pipeline consolidates them into structured categories for the first time
