# utils/image_utils.py
import os
import cv2
import numpy as np
from config import WINDOW_CENTER, WINDOW_WIDTH, MAX_VAL_CHECK

def normalize_ct(slice_data):
    """
    Normalizes CT data using Hounsfield Unit (HU) Windowing.
    Formula: Window Center (Level) and Window Width.
    """
    # 1. Calculate min and max HU based on window
    min_hu = WINDOW_CENTER - (WINDOW_WIDTH / 2)
    max_hu = WINDOW_CENTER + (WINDOW_WIDTH / 2)

    # 2. Clip values to this range
    clipped = np.clip(slice_data, min_hu, max_hu)

    # 3. Normalize to 0-1 range
    # (val - min) / (max - min)
    # The denominator is essentially just the WINDOW_WIDTH
    norm = (clipped - min_hu) / (max_hu - min_hu)

    # 4. Scale to 0-255 uint8
    return (norm * 255).astype(np.uint8)


def slice_volume_to_disk(img_vol, mask_vol, output_base_dir, subject_id):
    """
    Slices 3D volumes into 2D PNGs and saves them to a temp folder.
    """
    images_dir = os.path.join(output_base_dir, subject_id, "images")
    masks_dir = os.path.join(output_base_dir, subject_id, "masks")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # NIfTI is usually (Width, Height, Depth)
    width, height, depth = img_vol.shape
    valid_slices_count = 0

    for i in range(depth):
        img_slice = img_vol[:, :, i]
        mask_slice = mask_vol[:, :, i]

        # Skip slices containing only background
        if np.max(mask_slice) == 0:
            continue

        # Rotate to standard view
        img_slice = np.rot90(img_slice)
        mask_slice = np.rot90(mask_slice)

        # Validate data integrity
        if np.max(mask_slice) > MAX_VAL_CHECK:
            pass

        # --- CHANGED: Use CT Normalization ---
        img_png = normalize_ct(img_slice)
        mask_png = mask_slice.astype(np.uint8)

        filename = f"{subject_id}_slice_{i:04d}.png"

        cv2.imwrite(os.path.join(images_dir, filename), img_png)
        cv2.imwrite(os.path.join(masks_dir, filename), mask_png)

        valid_slices_count += 1

    return valid_slices_count, images_dir, masks_dir