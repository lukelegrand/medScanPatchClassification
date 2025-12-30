import os
from pathlib import Path
import pandas as pd

# --- FIX: Set backend to non-interactive 'Agg' before importing pyplot ---
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------

# --- CONFIGURATION ---
DATA_DIR = r"C:\TAC450 final\dataPipeline\combined_datasetMoreClass"


# ---------------------

def generate_overview(root_path):
    root = Path(root_path)

    if not root.exists():
        print(f"❌ Error: The directory '{root}' does not exist.")
        return

    print(f"{'CATEGORY':<25} | {'COUNT':<10}")
    print("-" * 40)

    category_data = []
    total_files = 0
    empty_folders = 0

    # Iterate over every item in the root directory
    for folder in sorted(root.iterdir()):
        if folder.is_dir():
            file_count = len([f for f in folder.glob('*') if f.is_file()])

            category_data.append({
                "Category": folder.name,
                "Count": file_count
            })

            total_files += file_count

            # Print row
            print(f"{folder.name:<25} | {file_count:<10}")

            if file_count == 0:
                empty_folders += 1

    print("-" * 40)
    print(f"{'TOTAL SCANS':<25} | {total_files:<10}")
    print(f"Total Categories: {len(category_data)}")

    if empty_folders > 0:
        print(f"\n⚠️ WARNING: Found {empty_folders} empty folders!")

    # --- Generate DataFrame and Plots ---
    if category_data:
        df = pd.DataFrame(category_data)

        df.to_csv("dataset_overview_combinedMC.csv", index=False)
        print("\nSaved data to 'dataset_overviewMC.csv'")

        plt.figure(figsize=(14, 6))

        # Plot 1: Bar Chart
        plt.subplot(1, 2, 1)
        plt.bar(df['Category'], df['Count'], color='skyblue', edgecolor='black')
        plt.title('File Count per Category')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=90)

        # Plot 2: Histogram
        plt.subplot(1, 2, 2)
        plt.hist(df['Count'], bins=10, color='salmon', edgecolor='black')
        plt.title('Distribution of Class Sizes')
        plt.xlabel('Number of Files')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig("dataset_overview_plot_combinedMC.png")
        print("Saved visualization to 'dataset_overview_plot_normalizedMC.png'")
    else:
        print("\nNo data found to plot.")


if __name__ == "__main__":
    generate_overview(DATA_DIR)