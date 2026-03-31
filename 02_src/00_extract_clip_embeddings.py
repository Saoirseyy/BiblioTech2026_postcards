"""
00_extract_clip_embeddings.py
==============================
Extract CLIP embeddings for all postcard images.
Saves clip_emb_matrix.pt and img_ids.txt for use in clustering.
 
Requirements:
    pip install sentence-transformers torch pillow tqdm --break-system-packages
"""
 
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
 
# =============================================================
# SETTINGS
# =============================================================
DATA_ROOT  = "/scratch/leuven/387/vsc38793/Dataset_0003_PicturePostcards"
OUTPUT_DIR = "/scratch/leuven/387/vsc38795/postcard_color_project"
 
CLIP_EMB_PATH = os.path.join(OUTPUT_DIR, "clip_emb_matrix.pt")
IMG_IDS_PATH  = os.path.join(OUTPUT_DIR, "img_ids.txt")
 
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
 
# Batch size: lower this if you run out of memory
BATCH_SIZE = 64
 
 
# =============================================================
# STEP 1 — Find all image files
# =============================================================
def find_image_files(root_dir):
    image_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(VALID_EXTENSIONS):
                image_files.append(os.path.join(root, f))
    return sorted(image_files)
 
 
print("Scanning image files...")
all_images = find_image_files(DATA_ROOT)
print(f"Found {len(all_images)} images.")
 
 
# =============================================================
# STEP 2 — Load CLIP model
# =============================================================
print("\nLoading CLIP model (clip-ViT-B-32)...")
model = SentenceTransformer("clip-ViT-B-32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Using device: {device}")
 
 
# =============================================================
# STEP 3 — Extract embeddings in batches
# =============================================================
def load_image_safe(path):
    """Load image as RGB. Return None if loading fails."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None
 
 
print(f"\nExtracting embeddings (batch_size={BATCH_SIZE})...")
 
embeddings  = []
valid_paths = []
failed      = 0
 
for i in tqdm(range(0, len(all_images), BATCH_SIZE), desc="Encoding batches"):
    batch_paths  = all_images[i : i + BATCH_SIZE]
    batch_images = []
    batch_valid  = []
 
    for path in batch_paths:
        img = load_image_safe(path)
        if img is not None:
            batch_images.append(img)
            batch_valid.append(path)
        else:
            failed += 1
 
    if len(batch_images) == 0:
        continue
 
    with torch.no_grad():
        batch_emb = model.encode(
            batch_images,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=device,
        )
 
    embeddings.append(batch_emb.cpu())
    valid_paths.extend(batch_valid)
 
print(f"\nDone. Encoded {len(valid_paths)} images. Failed: {failed}.")
 
 
# =============================================================
# STEP 4 — Save results
# =============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# Stack all embeddings into one matrix
emb_matrix = torch.cat(embeddings, dim=0)
print(f"Embedding matrix shape: {emb_matrix.shape}")  # (N, 512)
 
# Save embedding matrix
torch.save(emb_matrix, CLIP_EMB_PATH)
print(f"Saved embeddings to : {CLIP_EMB_PATH}")
 
# Save image path list
with open(IMG_IDS_PATH, "w") as f:
    for path in valid_paths:
        f.write(path + "\n")
print(f"Saved image IDs to  : {IMG_IDS_PATH}")
 
print("\nAll done. You can now run 03_clip_color_clustering.py.")