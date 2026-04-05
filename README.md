# Postcard Colour Classification

Colour analysis pipeline developed for the BiblioTech Hackathon 2026 (Team 4 — Inked and Stamped).

The 2026 edition of the KU Leuven BiblioTech Hackathon invited participants to work with two travel-related datasets from KU Leuven Libraries digitized collections, including a dataset of [Belgian historical postcards](https://kuleuven.limo.libis.be/discovery/collectionDiscovery?vid=32KUL_KUL:KULeuven&collectionId=81531489730001488&lang=en).

This repository focuses on the colour analysis pipeline I developed for 35,930 postcard images, with the goal of producing a more consistent and reusable colour labelling system.


## Overview

Historical postcard metadata often contains inconsistent and subjective colour labels. The same image may be labelled differently, and manual annotation is difficult to scale. This makes it hard to search, compare, or analyse images across the collection.

To address this, I built an end-to-end colour analysis pipeline that learns colour directly from the images rather than relying solely on metadata.

The pipeline:

- processes 35,930 postcard images  
- extracts more than 50 visual colour features  
- combines:
  - supervised learning (Random Forest)
  - unsupervised clustering (GMM)
  - rule-based refinement  
- produces a consistent system of 8 colour categories  

Metadata is used as guidance rather than ground truth, clustering is used to discover structure in the data, and post-processing rules ensure that the final categories remain semantically meaningful.


## Final Categories

| Category | Description |
|----------|------------|
| `bw` | Black-and-white |
| `sepia` | Sepia / warm brown tones |
| `monotone_blue` | Blue monotone print |  
| `monotone_red` | Red monotone print |  
| `monotone_green` | Green monotone print |
| `monotone_purple` | Purple monotone print |
| `color_handcolored` | Hand-coloured image |  
| `color_photo` | Real colour photograph |

![Final categories overview](04_result/category.png)


## Pipeline

### Step 1 — Feature Extraction  
`02_src/01_extract_colour_features.py`

Extracts 50 colour features per image using OpenCV and NumPy, including:

- saturation distribution  
- hue diversity  
- sepia pixel ratio  
- opponent colour channels  


### Step 2 — Metadata-Guided Classification  
`02_src/02_metadata_color_classification.py`

Trains a Random Forest classifier on postcards with existing metadata labels and predicts labels for unlabelled data.

Outputs three broad categories:

- black-and-white  
- sepia  
- colour  


### Step 3 — Fine-Grained Classification  
`02_src/03_fine_grained_classification.py`

Applies GMM sub-clustering within each broad category to produce six sub-categories.

Post-processing rules:

- merge GMM clusters for black-and-white and sepia to maintain semantic consistency  
- identify monotone printing styles (blue, green, purple, red) from metadata  
- consolidate all outputs into a unified set of 8 final categories  


## Notebooks

- `03_notebook/01_post_processing.ipynb`  
  Applies and validates post-processing rules  

- `03_notebook/02_colour_analysis_insights.ipynb`  
  Explores colour distribution by decade, city, and publisher  


## Data

- `01_data/00_metadata/20230301-Postcards.csv`  
  Original metadata  

- `01_data/01_processed/postcard_fine_labels.csv`  
  Final colour labels for all 35,930 postcards  


## Key Findings

- Colour photographs increase significantly from the 1960s onwards  
- Monotone printing styles (blue, green, purple, red) were present but inconsistently labelled in the original metadata  
- This pipeline provides a more standardised and consistent colour labelling system, improving the usability of the collection for analysis and exploration  


## Limitations

- Some paintings are classified as photographs  
- Colour features alone cannot fully capture semantic image types  


## Next Step

Use CLIP embeddings to:

- distinguish photographs from paintings  
- combine visual semantics (CLIP) with colour features from the current pipeline  

This would enable richer and more accurate cultural heritage metadata.
