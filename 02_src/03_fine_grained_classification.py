"""
03_fine_grained_classification.py
===================================
Fine-grained colour classification:
  bw    → bw_dark, bw_light
  sepia → sepia_dark, sepia_light
  color → color_handcolored, color_photo

Step 1: Load postcard_color_labels_final.csv (from script 04)
Step 2: For each of the 3 classes, run GMM sub-clustering
Step 3: For the 'color' class, use CLIP embeddings to separate
        hand-colored from real color photos (if available)
Step 4: Save final fine-grained labels
"""

import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


# =============================================================
# SETTINGS
# =============================================================
LABELS_CSV  = "/scratch/leuven/387/vsc38795/postcard_color_project/output/postcard_color_labels_final.csv"
OUTPUT_DIR  = "/scratch/leuven/387/vsc38795/postcard_color_project/output"

# CLIP files — set to None if not available yet
CLIP_EMB_PATH = "/scratch/leuven/387/vsc38795/postcard_color_project/clip_emb_matrix.pt"
IMG_IDS_PATH  = "/scratch/leuven/387/vsc38795/postcard_color_project/img_ids.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
# HELPER: show sample images per label
# =============================================================
def show_samples(df, label_col, path_col="image_path",
                 n=9, random_state=42, title_prefix=""):
    labels = sorted(df[label_col].dropna().unique())
    for lbl in labels:
        subset = df[df[label_col] == lbl]
        sample = subset.sample(min(n, len(subset)), random_state=random_state)
        cols = 3
        rows = math.ceil(len(sample) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")
        for ax, (_, row) in zip(axes, sample.iterrows()):
            try:
                img = Image.open(row[path_col]).convert("RGB")
                ax.imshow(img)
                ax.set_title(os.path.basename(row[path_col]), fontsize=7)
            except Exception:
                ax.text(0.5, 0.5, "load error", ha="center", va="center")
            ax.axis("off")
        plt.suptitle(
            f"{title_prefix}{label_col} = {lbl}  (n={len(subset)})",
            fontsize=13
        )
        plt.tight_layout()
        plt.show()


# =============================================================
# STEP 1 — Load labels from script 04
# =============================================================
print("Loading labels...")
df = pd.read_csv(LABELS_CSV)
print(f"Total images: {len(df)}")
print(df["predicted_label"].value_counts())


# =============================================================
# STEP 2 — GMM sub-clustering for bw and sepia
# =============================================================

def gmm_subcluster(df_subset, features, n_components=2,
                   label_prefix="", label_map=None):
    """
    Run GMM on a subset of images.
    Returns the subset with a new 'sub_label' column.
    label_map: dict mapping cluster int → label string
               e.g. {0: 'bw_light', 1: 'bw_dark'}
               If None, auto-assigns based on brightness.
    """
    df_sub = df_subset.dropna(subset=features).copy()

    # Clip outliers
    for col in features:
        p1  = df_sub[col].quantile(0.01)
        p99 = df_sub[col].quantile(0.99)
        df_sub[col] = df_sub[col].clip(lower=p1, upper=p99)

    X = RobustScaler().fit_transform(df_sub[features].values)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=5,
        max_iter=300,
        random_state=42,
    )
    clusters = gmm.fit_predict(X)
    df_sub["_cluster"] = clusters

    # Auto-assign labels by s_p50
    if label_map is None:
        means = df_sub.groupby("_cluster")["s_p50"].mean()
        sorted_clusters = means.sort_values().index.tolist()
        label_map = {
            sorted_clusters[0]: f"{label_prefix}_dark",
            sorted_clusters[1]: f"{label_prefix}_light",
        }

    df_sub["sub_label"] = df_sub["_cluster"].map(label_map)
    print(f"\n{label_prefix} sub-cluster counts:")
    print(df_sub["sub_label"].value_counts())

    return df_sub[["image_path", "sub_label"]]


# --- Features for bw sub-clustering ---
BW_FEATURES = [
    "s_p50",
    "s_p95",
    "chromatic_pixel_ratio",
    "saturated_pixel_ratio",
    "sepia_pixel_ratio",
]

# --- Features for sepia sub-clustering ---
SEPIA_FEATURES = [
    "sepia_pixel_ratio",
    "yb_mean",
    "s_p50",
    "s_bimodal_gap",
    "rg_mean",
]


# Run bw sub-clustering
print("\n=== BW sub-clustering ===")
df_bw    = df[df["predicted_label"] == "bw"].copy()
bw_sub   = gmm_subcluster(df_bw, BW_FEATURES,
                           n_components=2, label_prefix="bw")

# Run sepia sub-clustering
print("\n=== Sepia sub-clustering ===")
df_sepia  = df[df["predicted_label"] == "sepia"].copy()
sepia_sub = gmm_subcluster(df_sepia, SEPIA_FEATURES,
                            n_components=2, label_prefix="sepia")


# =============================================================
# STEP 3 — Color sub-clustering: hand-colored vs real photo
#
# Strategy A (always runs): use colour features
#   hand-colored: hue_unique_count low, s_bimodal_gap high
#   real photo:   hue_entropy high, hue_unique_count high
#
# Strategy B (runs if CLIP available): combine with CLIP
# =============================================================

COLOR_FEATURES = [
    "hue_entropy",
    "hue_unique_count",
    "s_bimodal_gap",
    "colorfulness_v2",
    "saturated_pixel_ratio",
    "sepia_pixel_ratio",
    "dom1_saturation",
]
COLOR_FEATURES = [f for f in COLOR_FEATURES if f in df.columns]

df_color = df[df["predicted_label"] == "color"].copy()
df_color_clean = df_color.dropna(subset=COLOR_FEATURES).copy()

for col in COLOR_FEATURES:
    p1  = df_color_clean[col].quantile(0.01)
    p99 = df_color_clean[col].quantile(0.99)
    df_color_clean[col] = df_color_clean[col].clip(lower=p1, upper=p99)

X_color = RobustScaler().fit_transform(df_color_clean[COLOR_FEATURES].values)

# --- Try to load CLIP embeddings ---
clip_available = False
if os.path.exists(CLIP_EMB_PATH) and os.path.exists(IMG_IDS_PATH):
    try:
        import torch
        print("\nLoading CLIP embeddings...")
        emb_matrix = torch.load(CLIP_EMB_PATH, map_location="cpu").numpy()
        img_names  = open(IMG_IDS_PATH).read().splitlines()
        clip_df    = pd.DataFrame({
            "image_path_clip": img_names,
            "clip_idx": range(len(img_names))
        })
        clip_df["basename"]           = clip_df["image_path_clip"].apply(os.path.basename)
        df_color_clean["basename"]    = df_color_clean["image_path"].apply(os.path.basename)
        merged = df_color_clean.merge(clip_df, on="basename", how="inner")

        if len(merged) > 100:
            clip_indices = merged["clip_idx"].values
            X_clip_full  = emb_matrix[clip_indices]
            pca_clip     = PCA(n_components=30, random_state=42)
            X_clip_30    = pca_clip.fit_transform(X_clip_full)
            X_clip_30    = RobustScaler().fit_transform(X_clip_30)

            # Re-align X_color to merged rows
            merged_idx  = merged.index
            X_color_sub = RobustScaler().fit_transform(
                merged[COLOR_FEATURES].values
            )
            X_combined  = np.hstack([X_color_sub, X_clip_30])
            df_color_gmm = merged.copy()
            X_for_gmm    = X_combined
            clip_available = True
            print(f"CLIP available, using combined features ({X_combined.shape[1]} dims)")
        else:
            print("Too few CLIP matches, falling back to colour features only.")
    except Exception as e:
        print(f"CLIP load failed ({e}), using colour features only.")

if not clip_available:
    df_color_gmm = df_color_clean.copy()
    X_for_gmm    = X_color
    print("\nUsing colour features only for color sub-clustering.")

# Run GMM on color group
print("\n=== Color sub-clustering ===")
gmm_color = GaussianMixture(
    n_components=2,
    covariance_type="full",
    n_init=5,
    max_iter=300,
    random_state=42,
)
color_clusters = gmm_color.fit_predict(X_for_gmm)
df_color_gmm["_cluster"] = color_clusters

# Assign labels by hue_entropy:
# higher hue_entropy → more diverse hues → real color photo
means_entropy = df_color_gmm.groupby("_cluster")["hue_entropy"].mean()
real_cluster  = means_entropy.idxmax()
hand_cluster  = means_entropy.idxmin()

color_label_map = {
    real_cluster: "color_photo",
    hand_cluster: "color_handcolored",
}
df_color_gmm["sub_label"] = df_color_gmm["_cluster"].map(color_label_map)

print("Color sub-cluster counts:")
print(df_color_gmm["sub_label"].value_counts())
print(f"\nMean hue_entropy per cluster:")
print(df_color_gmm.groupby("sub_label")["hue_entropy"].mean().round(3))

color_sub = df_color_gmm[["image_path", "sub_label"]]


# =============================================================
# STEP 4 — Merge all sub-labels back
# =============================================================
sub_label_df = pd.concat([bw_sub, sepia_sub, color_sub], ignore_index=True)

df = df.merge(sub_label_df, on="image_path", how="left")

# Fill any unmatched rows with the original 3-class label
df["fine_label"] = df["sub_label"].fillna(df["predicted_label"])

print("\n=== Final fine-grained label distribution ===")
print(df["fine_label"].value_counts())


# =============================================================
# STEP 5 — PCA visualisation
# =============================================================
FEAT_VIS = [f for f in COLOR_FEATURES + BW_FEATURES + SEPIA_FEATURES
            if f in df.columns]
FEAT_VIS = list(dict.fromkeys(FEAT_VIS))  # deduplicate

df_vis = df.dropna(subset=FEAT_VIS).copy()
for col in FEAT_VIS:
    p1  = df_vis[col].quantile(0.01)
    p99 = df_vis[col].quantile(0.99)
    df_vis[col] = df_vis[col].clip(lower=p1, upper=p99)

X_vis = RobustScaler().fit_transform(df_vis[FEAT_VIS].values)
pca   = PCA(n_components=2, random_state=42)
X_2d  = pca.fit_transform(X_vis)
ev    = pca.explained_variance_ratio_

color_palette = {
    "bw_dark":           "#333333",
    "bw_light":          "#aaaaaa",
    "sepia_dark":        "#7B3F00",
    "sepia_light":       "#D2A679",
    "color_photo":       "#1f77b4",
    "color_handcolored": "#ff7f0e",
}

plt.figure(figsize=(10, 7))
for lbl, col in color_palette.items():
    mask = df_vis["fine_label"] == lbl
    if mask.sum() == 0:
        continue
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                s=4, alpha=0.4, color=col, label=f"{lbl} (n={mask.sum()})")
plt.legend(fontsize=8, markerscale=2)
plt.title(f"PCA — fine-grained labels  (PC1={ev[0]:.1%}, PC2={ev[1]:.1%})")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.show()


# =============================================================
# STEP 6 — Sample images per fine label
# =============================================================
#show_samples(df, label_col="fine_label", path_col="image_path", n=9)


# =============================================================
# STEP 7 — Save
# =============================================================
out_path = os.path.join(OUTPUT_DIR, "postcard_fine_labels.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
print(f"Total rows: {len(df)}")
