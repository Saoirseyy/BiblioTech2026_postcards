"""
02_metadata_color_classification.py
=====================================
Use metadata colour labels as ground truth to:
  1. Directly label images that have metadata
  2. Train a Random Forest classifier on colour features
  3. Predict labels for images without metadata
  4. Optionally combine with CLIP embeddings if available
 
Colour categories (consolidated):
  bw    — Black-and-white, Grey
  sepia — Sepia, Brown
  color — Colour only
  (Blue, Green, Red, Purple, Pink → NaN, predicted by classifier)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =============================================================
# SETTINGS
# =============================================================
METADATA_CSV = "/scratch/leuven/387/vsc38795/postcard_color_project/20230301-Postcards-csv.csv"
COLOR_CSV = "/scratch/leuven/387/vsc38795/postcard_color_project/output/postcard_color_features_v2.csv"
OUTPUT_DIR    = "/scratch/leuven/387/vsc38795/postcard_color_project/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# HELPER: show sample images per label
# =============================================================
def show_label_samples(df, label_col, path_col="image_path",
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
        plt.suptitle(f"{title_prefix}{label_col} = {lbl}  (n={len(subset)})",
                     fontsize=13)
        plt.tight_layout()
        plt.show()
 
 
# =============================================================
# STEP 1 — Load and clean metadata
# =============================================================
print("Loading metadata...")
meta = pd.read_csv(METADATA_CSV)
print(f"Metadata shape: {meta.shape}")
 
# Extract IE id from Resolver URL
meta["IE_id"] = meta["Resolver URL 856$u"].str.extract(r"/(IE\d+)/")
 
# Consolidate colour labels into 3 categories
COLOUR_MAP = {
    "Black-and-white": "bw",
    "Grey":            "bw",
    "Sepia":           "sepia",
    "Brown":           "sepia",
    "Colour":          "color",
    
}
meta["color_label"] = meta["Colour 340$o_standardized"].map(COLOUR_MAP)
 
print("\nConsolidated colour distribution:")
print(meta["color_label"].value_counts())
print(f"Unmapped: {meta['color_label'].isna().sum()}")
 
 
# =============================================================
# STEP 2 — Load colour features, keep front side only
# =============================================================
print("\nLoading colour features...")
df_color = pd.read_csv(COLOR_CSV)
df_color["side_flag"] = df_color["file_name"].str.extract(
    r"_(R|V)(?=[._])", expand=False
)
df_front = df_color[df_color["side_flag"] == "R"].copy()
print(f"Front-side images: {len(df_front)}")
 
# Extract IE id from image path
df_front["IE_id"] = df_front["image_path"].str.extract(r"/(IE\d+)/")
print(f"IE id extracted: {df_front['IE_id'].notna().sum()} / {len(df_front)}")
 
 
# =============================================================
# STEP 3 — Join metadata colour labels onto image features
# =============================================================
meta_slim = meta[["IE_id", "color_label",
                   "Colour 340$o_standardized",
                   "Date 264$c_estimate",
                   "Date 264$c_estimateDecade"]].copy()
 
df = df_front.merge(meta_slim, on="IE_id", how="left")
print(f"\nAfter join:")
print(f"  Total images      : {len(df)}")
print(f"  With label        : {df['color_label'].notna().sum()}")
print(f"  Without label     : {df['color_label'].isna().sum()}")
print("\nLabel distribution (matched images):")
print(df["color_label"].value_counts())
 
 
# =============================================================
# STEP 4 — Prepare features
# =============================================================
FEATURES = [
    # Core colour signals
    "saturated_pixel_ratio",
    "chromatic_pixel_ratio",
    "colorfulness_v2",
    "s_p50",
    "s_p95",
    # Hue diversity
    "hue_entropy",
    "hue_unique_count",
    "s_bimodal_gap",
    # Sepia / tonal
    "sepia_pixel_ratio",
    "yb_mean",
    "rg_mean",
    # Dominant colour saturation
    "dom1_saturation",
    "dom2_saturation",
]
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"\nFeatures used: {len(FEATURES)}")
 
# Clip outliers
df_clean = df.dropna(subset=FEATURES).copy()
for col in FEATURES:
    p1  = df_clean[col].quantile(0.01)
    p99 = df_clean[col].quantile(0.99)
    df_clean[col] = df_clean[col].clip(lower=p1, upper=p99)
 
scaler   = RobustScaler()
X_all    = scaler.fit_transform(df_clean[FEATURES].values)
 
# Split into labelled and unlabelled
labelled_mask   = df_clean["color_label"].notna()
X_labelled      = X_all[labelled_mask]
y_labelled      = df_clean.loc[labelled_mask, "color_label"].values
X_unlabelled    = X_all[~labelled_mask]
 
print(f"\nLabelled   : {len(X_labelled)}")
print(f"Unlabelled : {len(X_unlabelled)}")
 
 
# =============================================================
# STEP 5 — Train Random Forest on labelled data
# =============================================================
print("\nTraining Random Forest classifier...")
 
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",   # handles class imbalance
    random_state=42,
    n_jobs=-1,
)
 
# Cross-validation to estimate accuracy
cv_scores = cross_val_score(clf, X_labelled, y_labelled,
                             cv=5, scoring="f1_macro", n_jobs=-1)
print(f"5-fold CV F1 (macro): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
 
# Fit on all labelled data
clf.fit(X_labelled, y_labelled)
 
# Feature importance
importance_df = pd.DataFrame({
    "feature":    FEATURES,
    "importance": clf.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nTop feature importances:")
print(importance_df.head(8).to_string(index=False))
 
# Plot feature importance
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance_df["feature"][::-1],
        importance_df["importance"][::-1], color="steelblue")
ax.set_xlabel("Importance")
ax.set_title("Random Forest — Feature Importance")
plt.tight_layout()
plt.show()
 
 
# =============================================================
# STEP 6 — Evaluate on labelled data (in-sample, for sanity check)
# =============================================================
y_pred_labelled = clf.predict(X_labelled)
print("\nClassification report (labelled data):")
print(classification_report(y_labelled, y_pred_labelled,
                             target_names=["bw", "color", "sepia"]))
 
# Confusion matrix
cm = confusion_matrix(y_labelled, y_pred_labelled,
                      labels=["bw", "sepia", "color"])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["bw", "sepia", "color"],
            yticklabels=["bw", "sepia", "color"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion matrix (labelled data)")
plt.tight_layout()
plt.show()
 
 
# =============================================================
# STEP 7 — Predict labels for unlabelled images
# =============================================================
print("\nPredicting labels for unlabelled images...")
 
# Assign labels: use metadata label if available, else predict
df_clean["predicted_label"] = df_clean["color_label"].copy()
df_clean["label_source"]    = "metadata"
 
if len(X_unlabelled) > 0:
    y_pred_unlabelled = clf.predict(X_unlabelled)
    y_prob_unlabelled = clf.predict_proba(X_unlabelled).max(axis=1)
 
    df_clean.loc[~labelled_mask, "predicted_label"] = y_pred_unlabelled
    df_clean.loc[~labelled_mask, "label_source"]    = "predicted"
    df_clean.loc[~labelled_mask, "label_confidence"] = y_prob_unlabelled
    df_clean.loc[labelled_mask,  "label_confidence"] = 1.0
 
print("\nFinal label distribution:")
print(df_clean["predicted_label"].value_counts())
print("\nBy source:")
print(df_clean.groupby(["label_source", "predicted_label"]).size())
 
 
# =============================================================
# STEP 8 — Visualise: PCA coloured by label
# =============================================================
from sklearn.decomposition import PCA
 
pca  = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_all)
ev   = pca.explained_variance_ratio_
 
color_map = {"bw": "gray", "sepia": "peru", "color": "steelblue"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
for ax, source in zip(axes, ["metadata", "all"]):
    if source == "metadata":
        mask = labelled_mask.values
        labels = y_labelled
        title  = "Metadata labels only"
    else:
        mask   = np.ones(len(df_clean), dtype=bool)
        labels = df_clean["predicted_label"].values
        title  = "All images (metadata + predicted)"
 
    for lbl, col in color_map.items():
        m = labels == lbl
        ax.scatter(X_2d[mask][m, 0], X_2d[mask][m, 1],
                   s=4, alpha=0.4, color=col, label=lbl)
    ax.set_title(title)
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
 
plt.suptitle("PCA coloured by colour label", fontsize=12)
plt.tight_layout()
plt.show()
 
 
# =============================================================
# STEP 9 — Sample images per label
# =============================================================
show_label_samples(
    df_clean[df_clean["predicted_label"] == "bw"],
    label_col="predicted_label", path_col="image_path", n=9,
    title_prefix="[Final] "
)
show_label_samples(
    df_clean[df_clean["predicted_label"] == "sepia"],
    label_col="predicted_label", path_col="image_path", n=9,
    title_prefix="[Final] "
)
show_label_samples(
    df_clean[df_clean["predicted_label"] == "color"],
    label_col="predicted_label", path_col="image_path", n=9,
    title_prefix="[Final] "
)
 
 
# =============================================================
# STEP 10 — Save results
# =============================================================
out_cols = (
    ["image_id", "file_name", "image_path", "IE_id",
     "color_label", "predicted_label", "label_source", "label_confidence",
     "Colour 340$o_standardized", "Date 264$c_estimate", "Date 264$c_estimateDecade"]
    + FEATURES
)
out_cols = [c for c in out_cols if c in df_clean.columns]
 
out_path = os.path.join(OUTPUT_DIR, "postcard_color_labels_final.csv")
df_clean[out_cols].to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
print(f"Total rows: {len(df_clean)}")
 
