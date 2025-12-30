# utils/nifti_utils.py
import os
import nibabel as nib
from totalsegmentator.libs import combine_masks_to_multilabel_file


def combine_masks(subject_path, temp_output_path):
    """
    Combines individual segmentation files into one multi-label file.
    """
    input_folder = os.path.join(subject_path, "segmentations")

    if not os.path.exists(input_folder):
        return False, "Segmentations folder not found"

    # Create directory for the combined file if it doesn't exist
    os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)

    try:
        # TotalSegmentator function to combine masks
        # This works for both CT and MRI masks as they share the same ID structure
        combine_masks_to_multilabel_file(input_folder, temp_output_path)
        return True, "Success"
    except Exception as e:
        return False, str(e)


def load_volumes(img_path, mask_path):
    """
    Loads MRI and Mask volumes, ensuring canonical orientation.
    """
    try:
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)

        # Force canonical orientation (RAS+) so all scans face the same way
        # This is critical for mixed datasets
        img_vol = nib.as_closest_canonical(img_nii).get_fdata()
        mask_vol = nib.as_closest_canonical(mask_nii).get_fdata()

        return img_vol, mask_vol
    except Exception as e:
        print(f"Error loading volumes: {e}")
        return None, None