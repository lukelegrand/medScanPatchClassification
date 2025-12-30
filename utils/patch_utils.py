# utils/patch_utils.py
import os
import cv2
import numpy as np
from totalsegmentator.map_to_binary import class_map
from config import WINDOW_SIZE, STRIDE, PURITY_THRESHOLD, FILL_RATIO_THRESHOLD

# Initialize Class Map
raw_map = class_map['total']
ID_TO_NAME = {}

# Logic to invert map (ID -> Name)
first_key = list(raw_map.keys())[0]
first_val = list(raw_map.values())[0]

if isinstance(first_key, str) and isinstance(first_val, int):
    ID_TO_NAME = {v: k for k, v in raw_map.items()}
elif isinstance(first_key, int) and isinstance(first_val, str):
    ID_TO_NAME = raw_map


def extract_patches(images_dir, masks_dir, output_root):
    """
    Iterates over 2D slices in the temp folder and extracts patches.
    """
    created_folders = set()
    total_patches_extracted = 0
    files = [f for f in os.listdir(images_dir) if f.endswith(".png")]

    for filename in files:
        img_path = os.path.join(images_dir, filename)
        mask_path = os.path.join(masks_dir, filename)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if img is None or mask is None:
            continue

        h, w = img.shape

        # Sliding window
        for y in range(0, h - WINDOW_SIZE + 1, STRIDE):
            for x in range(0, w - WINDOW_SIZE + 1, STRIDE):

                mask_patch = mask[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE]

                # Quick check to skip empty patches
                if np.max(mask_patch) == 0:
                    continue

                unique, counts = np.unique(mask_patch, return_counts=True)

                # Ignore background (0) for dominance check
                if 0 in unique:
                    idx_0 = np.where(unique == 0)[0][0]
                    bg_count = counts[idx_0]
                    unique = np.delete(unique, idx_0)
                    counts = np.delete(counts, idx_0)
                else:
                    bg_count = 0

                if len(unique) == 0:
                    continue

                # Find dominant organ in this patch
                dominant_idx = np.argmax(counts)
                dominant_id = unique[dominant_idx]
                dominant_count = counts[dominant_idx]

                if dominant_id not in ID_TO_NAME:
                    continue

                total_pixels = WINDOW_SIZE * WINDOW_SIZE
                non_bg_pixels = total_pixels - bg_count

                if non_bg_pixels == 0:
                    continue

                purity = dominant_count / non_bg_pixels
                fill_ratio = non_bg_pixels / total_pixels

                if purity >= PURITY_THRESHOLD and fill_ratio > FILL_RATIO_THRESHOLD:
                    organ_name = str(ID_TO_NAME[dominant_id])

                    organ_folder = os.path.join(output_root, organ_name)

                    if organ_name not in created_folders:
                        os.makedirs(organ_folder, exist_ok=True)
                        created_folders.add(organ_name)

                    img_patch = img[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE]

                    save_name = f"{filename.replace('.png', '')}_y{y}_x{x}.png"
                    cv2.imwrite(os.path.join(organ_folder, save_name), img_patch)
                    total_patches_extracted += 1

    return total_patches_extracted