import os
import re
import pandas as pd
from fuzzywuzzy import process

# Define paths
CSV_PATH = "cuisines.csv"            # Your existing CSV file
IMAGE_DIR = "image_for_cuisines/data"  # Directory where your .jpg images are stored

# Read the CSV file
df = pd.read_csv(CSV_PATH)

# Ensure the 'image_file' column exists; if not, create it.
if 'image_file' not in df.columns:
    df['image_file'] = ""

# Get a list of all .jpg files in the image directory
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')]

def clean_name(name):
    """
    Clean a string by lowercasing, removing punctuation, and collapsing whitespace.
    """
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

# Pre-clean the image filenames for better fuzzy matching
cleaned_filenames = {}
for f in image_files:
    base, _ = os.path.splitext(f)
    # Replace underscores and hyphens with spaces and clean punctuation
    base_clean = base.replace("_", " ").replace("-", " ")
    base_clean = re.sub(r"[^\w\s]", " ", base_clean.lower())
    base_clean = re.sub(r"\s+", " ", base_clean).strip()
    cleaned_filenames[f] = base_clean

# For each row in the CSV, attempt to match the recipe name to a filename
for idx, row in df.iterrows():
    if pd.notna(row['name']):
        recipe_name = clean_name(str(row['name']))
        best_match, best_score = None, 0
        for filename, cleanfile in cleaned_filenames.items():
            score = process.extractOne(recipe_name, [cleanfile])[1]
            if score > best_score:
                best_score = score
                best_match = filename
        # Use a threshold score (e.g., 50) to decide if the match is acceptable
        if best_score > 50:
            df.at[idx, 'image_file'] = best_match
        else:
            df.at[idx, 'image_file'] = ""
    else:
        df.at[idx, 'image_file'] = ""

# Write the updated DataFrame back to the same CSV file (overwriting it)
df.to_csv(CSV_PATH, index=False)
print(f"Updated CSV saved to {CSV_PATH}")
