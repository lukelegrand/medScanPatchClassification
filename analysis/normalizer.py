import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

# --- CONFIGURATION ---
SOURCE_DIR = r"C:\TAC450 final\dataPipeline\combined_datasetMoreClass"
DEST_DIR = r"C:\TAC450 final\dataPipeline\patches_normalizedMC"

TARGET_COUNT = 100
MIN_REQUIRED = 10  # Classes below this count will be IGNORED (not copied)


# ---------------------

def create_normalized_dataset(source_path, dest_path):
    src_root = Path(source_path)
    dst_root = Path(dest_path)

    if not src_root.exists():
        print(f"❌ Source directory '{src_root}' does not exist.")
        return

    # Create destination root if it doesn't exist
    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"📂 Creating normalized dataset in: {dst_root}\n")

    print(f"{'CATEGORY':<25} | {'ACTION':<15} | {'STATUS'}")
    print("-" * 70)

    for folder in sorted(src_root.iterdir()):
        if not folder.is_dir():
            continue

        # Get valid images from source
        files = [f for f in folder.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]
        count = len(files)

        # Define destination folder for this category
        dest_folder = dst_root / folder.name

        # --- CASE 1: SKIP (Too few images) ---
        if count < MIN_REQUIRED:
            print(f"{folder.name:<25} | {'⚠️ SKIPPED':<15} | Count {count} < {MIN_REQUIRED} (Not copied)")
            continue

        # If we are proceeding, create the category folder in destination
        dest_folder.mkdir(exist_ok=True)

        # --- CASE 2: UNDERSAMPLE (Too many images) ---
        if count > TARGET_COUNT:
            # Pick random 500
            selected_files = random.sample(files, TARGET_COUNT)

            for f in selected_files:
                shutil.copy2(f, dest_folder / f.name)

            print(f"{folder.name:<25} | {'Undersampling':<15} | Copied 500 (from {count})")

        # --- CASE 3: AUGMENT (Medium amount of images) ---
        else:
            # 1. Copy ALL original files first
            for f in files:
                shutil.copy2(f, dest_folder / f.name)

            # 2. Generate new images to reach TARGET_COUNT
            needed = TARGET_COUNT - count
            generated = 0

            # Loop until we have enough
            while generated < needed:
                src_file = random.choice(files)  # Pick a random source image
                try:
                    with Image.open(src_file) as img:
                        # --- Augmentation Logic ---
                        # Rotate
                        angle = random.randint(-15, 15)
                        aug_img = img.rotate(angle)

                        # Mirror
                        if random.choice([True, False]):
                            aug_img = ImageOps.mirror(aug_img)

                        # Brightness
                        enhancer = ImageEnhance.Brightness(aug_img)
                        factor = random.uniform(0.8, 1.2)
                        aug_img = enhancer.enhance(factor)

                        # Save to DESTINATION
                        new_filename = f"aug_{generated}_{src_file.name}"
                        # Ensure we save in the same format as original
                        aug_img.save(dest_folder / new_filename)

                        generated += 1
                except Exception as e:
                    print(f"Error augmenting {src_file.name}: {e}")

            print(f"{folder.name:<25} | {'Augmenting':<15} | Copied {count} + Generated {needed}")

    print("-" * 70)
    print("✅ Process Complete.")


if __name__ == "__main__":
    create_normalized_dataset(SOURCE_DIR, DEST_DIR)