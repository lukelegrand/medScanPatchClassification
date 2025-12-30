# example1.py
import os
import shutil
import argparse
from tqdm import tqdm
from config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, TEMP_SLICE_DIR
from utils.nifti_utils import combine_masks, load_volumes
from utils.image_utils import slice_volume_to_disk
from utils.patch_utils import extract_patches


def process_pipeline(input_root, output_root):
    # 1. Setup
    if os.path.exists(TEMP_SLICE_DIR):
        shutil.rmtree(TEMP_SLICE_DIR)
    os.makedirs(TEMP_SLICE_DIR, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    # 2. Get list of subject folders
    # We look for folders containing 'ct.nii.gz'
    subjects = []
    if os.path.exists(input_root):
        for item in os.listdir(input_root):
            subj_path = os.path.join(input_root, item)
            # --- CHANGED: Check for ct.nii.gz ---
            if os.path.isdir(subj_path) and os.path.exists(os.path.join(subj_path, 'ct.nii.gz')):
                subjects.append(item)
    else:
        print(f"Error: Input directory {input_root} does not exist.")
        return

    print(f"Found {len(subjects)} subjects to process.")

    # 3. Iterate with Progress Bar
    global_patch_count = 0

    for subject_id in tqdm(subjects, desc="Processing CT Scans", unit="scan"):
        subject_path = os.path.join(input_root, subject_id)

        # --- CHANGED: Target the CT file ---
        ct_path = os.path.join(subject_path, "ct.nii.gz")

        # Define where the combined mask will live
        combined_mask_path = os.path.join(subject_path, "combined.nii.gz")

        # A. Combine Masks (Only if not already done)
        if not os.path.exists(combined_mask_path):
            success, msg = combine_masks(subject_path, combined_mask_path)
            if not success:
                tqdm.write(f"Skipping {subject_id}: {msg}")
                continue

        # B. Load Volumes (Use CT path)
        img_vol, mask_vol = load_volumes(ct_path, combined_mask_path)
        if img_vol is None:
            continue

        # C. Slice to Temp Disk
        count, img_dir, mask_dir = slice_volume_to_disk(
            img_vol, mask_vol, TEMP_SLICE_DIR, subject_id
        )

        if count == 0:
            tqdm.write(f"No valid slices found for {subject_id}")
            continue

        # D. Extract Patches
        patches_found = extract_patches(img_dir, mask_dir, output_root)
        global_patch_count += patches_found

        # E. Cleanup Temp Files for this subject
        shutil.rmtree(os.path.join(TEMP_SLICE_DIR, subject_id))

    # Final cleanup
    if os.path.exists(TEMP_SLICE_DIR):
        shutil.rmtree(TEMP_SLICE_DIR)

    print(f"\nPipeline Complete!")
    print(f"Total Patches Generated: {global_patch_count}")
    print(f"Data saved to: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT Patch Extraction Pipeline")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_DIR, help="Path to raw data folders")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Path to save patches")

    args = parser.parse_args()

    process_pipeline(args.input, args.output)