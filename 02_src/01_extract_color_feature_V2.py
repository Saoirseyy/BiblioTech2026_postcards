import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy as scipy_entropy

# =========================================================
# SETTINGS
# =========================================================
DATA_ROOT  = "/scratch/leuven/387/vsc38793/Dataset_0003_PicturePostcards"
OUTPUT_CSV = "/scratch/leuven/387/vsc38795/postcard_color_project/output/postcard_color_features_v2.csv"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
RESIZE_TO    = (80, 80)
TOP_K_COLORS = 3
QUANT_LEVEL  = 16

# ---- Thresholds ----
SAT_THRESHOLD       = 25   # s > 25 → pixel is considered saturated (OpenCV range 0–255)
SAT_HIGH_THRESHOLD  = 80   # s > 80 → highly saturated pixel
SAT_LOW_THRESHOLD   = 15   # s < 15 → low saturation / near grayscale pixel
CHROMA_THRESHOLD    = 20   # max(R,G,B)-min(R,G,B) > 20 → pixel has noticeable color difference
HUE_BINS            = 18   # hue histogram bins (each ≈20 degrees)
HUE_BINS_FINE       = 36   # finer hue histogram (each ≈5 degrees, used for diversity)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def find_image_files(root_dir):
    image_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(VALID_EXTENSIONS):
                image_files.append(os.path.join(root, f))
    return sorted(image_files)


def get_image_id(path):
    parent = os.path.basename(os.path.dirname(path))
    if parent.startswith("IE"):
        return parent
    return os.path.splitext(os.path.basename(path))[0]


def quantized_dominant_colors(rgb_img, top_k=3, levels=16):
    pixels = rgb_img.reshape(-1, 3)
    q = (pixels // levels).astype(np.int32)
    unique_bins, counts = np.unique(q, axis=0, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1][:top_k]
    top_bins = unique_bins[sorted_idx]
    colors = []
    for b in top_bins:
        color = (b * levels + levels // 2).clip(0, 255)
        colors.append(color.astype(int))
    while len(colors) < top_k:
        colors.append(np.array([0, 0, 0]))
    return colors


def single_rgb_to_hsv_sat(r_val, g_val, b_val):
    arr = np.array([[[int(r_val), int(g_val), int(b_val)]]], dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    return float(hsv[0, 0, 1]) / 255.0


# =========================================================
# CORE FEATURE COMPUTATION
# =========================================================
def compute_features(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, RESIZE_TO, interpolation=cv2.INTER_AREA)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)

    h = img_hsv[:, :, 0].astype(np.float32)   # OpenCV: 0-179
    s = img_hsv[:, :, 1].astype(np.float32)   # OpenCV: 0-255
    v = img_hsv[:, :, 2].astype(np.float32)   # OpenCV: 0-255

    # =========================================================
    # BLOCK 1 — Basic RGB / HSV statistics
    # =========================================================
    r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
    h_mean, s_mean, v_mean = h.mean(), s.mean(), v.mean()
    h_std,  s_std,  v_std  = h.std(),  s.std(),  v.std()

    brightness_mean = gray.mean()
    brightness_std  = gray.std()

    # similarity: the smaller the channel differences, the closer to black-and-white
    grayness_score = np.mean(
        np.abs(r - g) + np.abs(r - b) + np.abs(g - b)
    ) / 255.0

    # rough sepia indicator: if R is larger than B the color tends to be warm
    sepia_score = np.mean((r - b) / (r + g + b + 1.0))

    # ratio of warm / cool pixels (OpenCV hue range 0–179)
    warm_mask  = (h < 15) | (h > 150) | ((h >= 15) & (h < 35))
    cool_mask  = (h >= 35) & (h < 100)
    warm_ratio = warm_mask.mean()
    cool_ratio = cool_mask.mean()

    dom_colors = quantized_dominant_colors(img_rgb, top_k=TOP_K_COLORS, levels=QUANT_LEVEL)

    # =========================================================
    # BLOCK 2 — Saturation distribution
    #
    # black-and-white: almost all pixels have s ≈ 0
    # sepia: low saturation but not zero, distribution concentrated at low values
    # colorized postcards: low-sat background + some high-sat colored regions → bimodal
    # real color photos: saturation more evenly distributed and generally higher
    # =========================================================
    saturated_pixel_ratio = (s > SAT_THRESHOLD).mean()

    s_p10 = float(np.percentile(s, 10))
    s_p50 = float(np.percentile(s, 50))
    s_p90 = float(np.percentile(s, 90))
    s_p95 = float(np.percentile(s, 95))

    # bimodal gap: colorized postcards tend to have high p90 (colored areas)
    # but low p10 (gray background), so the gap becomes large
    s_bimodal_gap = s_p90 - s_p10

    # ratio of highly saturated / low saturation pixels
    highly_saturated_ratio = (s > SAT_HIGH_THRESHOLD).mean()
    low_sat_ratio          = (s < SAT_LOW_THRESHOLD).mean()

    #  bimodal ratio: usually small for colorized postcards
    # (large gray background + small colored areas)
    bimodal_ratio = highly_saturated_ratio / (low_sat_ratio + 1e-6)

    # =========================================================
    # BLOCK 3 — Pixel color difference (channel range)
    # black-and-white: RGB channels are almost equal → range ≈ 0
    # sepia: R > B, so range slightly larger
    # color photos: range clearly larger
    # =========================================================
    pixel_stack   = np.stack([r, g, b], axis=-1)
    channel_range = pixel_stack.max(axis=-1) - pixel_stack.min(axis=-1)

    chromatic_pixel_ratio = (channel_range > CHROMA_THRESHOLD).mean()
    channel_range_mean    = channel_range.mean()
    channel_range_p95     = float(np.percentile(channel_range, 95))

    # =========================================================
    # BLOCK 4 — Sepia pixel ratio (more precise than sepia_score)
    # condition: warm hue + low saturation + medium brightness
    # =========================================================
    sepia_mask = (
        (h < 25) &
        (s > 15) & (s < 80) &
        (v > 80) & (v < 220)
    )
    sepia_pixel_ratio = sepia_mask.mean()

    # =========================================================
    # BLOCK 5 — Hue diversity (only using saturated pixels)
    #
    # hue_entropy：randomness of hue distribution
    #   high for real color photos, low for BW / sepia / colorized
    #
    # hue_unique_count：number of hue bins with >1% pixels
    #   colorized postcards: usually 3–6 ; real color photos: can reach 15–30
    #
    # hue_dominance：proportion of the most common hue bin
    #   colorized / sepia: colors concentrated → high ; real color photos: colors spread → low
    # =========================================================
    sat_mask = s > SAT_THRESHOLD

    if sat_mask.sum() > 10:
        hue_sat = h[sat_mask]

        # entropy（18 bins）
        hue_hist_18, _ = np.histogram(hue_sat, bins=HUE_BINS, range=(0, 180))
        hue_hist_18    = hue_hist_18.astype(float)
        hue_hist_norm  = hue_hist_18 / (hue_hist_18.sum() + 1e-9)
        hue_entropy    = float(scipy_entropy(hue_hist_norm + 1e-9))

        # dominance
        hue_dominance  = float(hue_hist_18.max()) / (sat_mask.sum() + 1e-6)
    else:
        hue_entropy   = 0.0
        hue_dominance = 1.0

    # hue_unique_count（36 bins, finer resolution）
    fine_sat_mask = s > 40
    if fine_sat_mask.sum() > 20:
        hue_hist_36, _ = np.histogram(
            h[fine_sat_mask], bins=HUE_BINS_FINE, range=(0, 180)
        )
        min_bin = fine_sat_mask.sum() * 0.01
        hue_unique_count = int((hue_hist_36 > min_bin).sum())
    else:
        hue_unique_count = 0

    # =========================================================
    # BLOCK 6 — Opponent color channels (Hasler & Süsstrunk 2003)
    #
    # rg = R - G（red-green opponent）
    # yb = 0.5*(R+G) - B（yellow-blue opponent）
    #
    # black-and-white：rg_mean ≈ 0, yb_mean ≈ 0, std , both std are small
    # sepia：yb_mean > 0（warmer tone），rg_mean slightly positive，low std
    # real color photos: larger std, mean can be positive or negative
    # =========================================================
    rg = r - g
    yb = 0.5 * (r + g) - b

    rg_mean = float(rg.mean())
    yb_mean = float(yb.mean())
    rg_std  = float(rg.std())
    yb_std  = float(yb.std())

    # improved colorfulness metric
    colorfulness_v2 = (
        np.sqrt(rg_std**2 + yb_std**2) +
        0.3 * np.sqrt(rg_mean**2 + yb_mean**2)
    )

    # =========================================================
    # BLOCK 7 — Saturation of dominant colors
    # =========================================================
    dom1_sat = single_rgb_to_hsv_sat(dom_colors[0][0], dom_colors[0][1], dom_colors[0][2])
    dom2_sat = single_rgb_to_hsv_sat(dom_colors[1][0], dom_colors[1][1], dom_colors[1][2])

    # =========================================================
    # RETURN
    # =========================================================
    return {
        # --- Basic identification ---
        "image_id":   get_image_id(image_path),
        "file_name":  os.path.basename(image_path),
        "image_path": image_path,

        # --- BLOCK 1: BLOCK 1: Original basic statistics ---
        "r_mean": r_mean, "g_mean": g_mean, "b_mean": b_mean,
        "h_mean": h_mean, "s_mean": s_mean, "v_mean": v_mean,
        "h_std":  h_std,  "s_std":  s_std,  "v_std":  v_std,
        "brightness_mean": brightness_mean,
        "brightness_std":  brightness_std,
        "grayness_score":  grayness_score,
        "sepia_score":     sepia_score,
        "warm_ratio":      warm_ratio,
        "cool_ratio":      cool_ratio,

        # --- BLOCK 2: Saturation distribution ---
        "saturated_pixel_ratio":   saturated_pixel_ratio,   # ★ black-and-white vs color
        "s_p10":                   s_p10,
        "s_p50":                   s_p50,
        "s_p90":                   s_p90,
        "s_p95":                   s_p95,
        "s_bimodal_gap":           s_bimodal_gap,            # ★ detect colorized postcards
        "highly_saturated_ratio":  highly_saturated_ratio,   # ★ detect colorized postcards
        "low_sat_ratio":           low_sat_ratio,            # ★ colorized postcards / black-and-white
        "bimodal_ratio":           bimodal_ratio,            # ★ detect colorized postcards

        # --- BLOCK 3: Pixel color difference ---
        "chromatic_pixel_ratio": chromatic_pixel_ratio,     # ★ black-and-white vs color
        "channel_range_mean":    channel_range_mean,
        "channel_range_p95":     channel_range_p95,

        # --- BLOCK 4: Sepia pixels ---
        "sepia_pixel_ratio": sepia_pixel_ratio,             # ★ detect sepia postcards

        # --- BLOCK 5: Hue diversity ---
        "hue_entropy":      hue_entropy,                    # ★ color vs others
        "hue_unique_count": hue_unique_count,               # ★ detect colorized postcards
        "hue_dominance":    hue_dominance,                  # ★ colorized postcards / sepia

        # --- BLOCK 6: Opponent color channels ---
        "rg_mean": rg_mean, "yb_mean": yb_mean,
        "rg_std":  rg_std,  "yb_std":  yb_std,
        "colorfulness_v2": colorfulness_v2,                 # ★ overall color strength

        # --- BLOCK 7: Dominant colors ---
        "dom1_r": int(dom_colors[0][0]),
        "dom1_g": int(dom_colors[0][1]),
        "dom1_b": int(dom_colors[0][2]),
        "dom1_saturation": dom1_sat,

        "dom2_r": int(dom_colors[1][0]),
        "dom2_g": int(dom_colors[1][1]),
        "dom2_b": int(dom_colors[1][2]),
        "dom2_saturation": dom2_sat,

        "dom3_r": int(dom_colors[2][0]),
        "dom3_g": int(dom_colors[2][1]),
        "dom3_b": int(dom_colors[2][2]),
    }


# =========================================================
# MAIN
# =========================================================
def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    image_files = find_image_files(DATA_ROOT)
    print(f"Found {len(image_files)} image files.")

    rows, failed = [], 0
    for path in tqdm(image_files, desc="Extracting color features v2"):
        row = compute_features(path)
        if row is None:
            failed += 1
            continue
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone. Saved to: {OUTPUT_CSV}")
    print(f"Rows saved : {len(df)}")
    print(f"Failed     : {failed}")
    print(f"\nColumns ({len(df.columns)}):")
    print(list(df.columns))


if __name__ == "__main__":
    main()
