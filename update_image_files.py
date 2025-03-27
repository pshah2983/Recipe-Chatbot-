import pandas as pd
import os
import re
from difflib import SequenceMatcher

def clean_name(name):
    # Convert to lowercase
    name = name.lower()
    # Remove recipe, with, and other common words
    name = re.sub(r'\s*recipe\s*', ' ', name)
    name = re.sub(r'\s*with\s*', ' ', name)
    name = re.sub(r'\s*and\s*', ' ', name)
    name = re.sub(r'\s*style\s*', ' ', name)
    # Remove text in parentheses
    name = re.sub(r'\([^)]*\)', '', name)
    # Remove special characters but keep underscores
    name = re.sub(r'[^a-z0-9\s_]', ' ', name)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name.strip())
    return name

def get_words(name):
    return set(clean_name(name).split())

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Read the CSV file
df = pd.read_csv('cuisines.csv')

# Get list of image files
image_dir = 'image_for_cuisines/data'
image_files = os.listdir(image_dir)

# Create a mapping of image files with their cleaned names and word sets
image_info = {}
for img_file in image_files:
    # Remove the number prefix and file extension
    clean_img_name = re.sub(r'^\d+\.', '', img_file)
    clean_img_name = os.path.splitext(clean_img_name)[0]
    clean_img_name = clean_name(clean_img_name)
    words = get_words(clean_img_name)
    image_info[img_file] = {
        'clean_name': clean_img_name,
        'words': words
    }

def find_matching_image(recipe_name):
    clean_recipe = clean_name(recipe_name)
    recipe_words = get_words(recipe_name)
    
    best_match = None
    best_score = 0
    
    for img_file, info in image_info.items():
        # Calculate word overlap score
        common_words = recipe_words.intersection(info['words'])
        word_score = len(common_words) / max(len(recipe_words), len(info['words']))
        
        # Calculate string similarity score
        sim_score = similar(clean_recipe, info['clean_name'])
        
        # Combined score (weighted average)
        score = 0.7 * word_score + 0.3 * sim_score
        
        if score > best_score:
            best_score = score
            best_match = img_file
    
    # Only return matches above a certain threshold
    return best_match if best_score > 0.15 else None

# Update image_file column
df['image_file'] = df['name'].apply(find_matching_image)

# Save updated CSV
df.to_csv('cuisines.csv', index=False)

# Print statistics
total_recipes = len(df)
matched_recipes = df['image_file'].notna().sum()

print(f"\nTotal recipes: {total_recipes}")
print(f"Matched with images: {matched_recipes}")
print(f"Match rate: {matched_recipes/total_recipes*100:.1f}%\n")

print("Sample matches:")
print("-" * 80)
sample = df.sample(n=5)
for idx, row in sample.iterrows():
    print(f"\nRecipe: {row['name']}")
    print(f"Image:  {row['image_file']}") 