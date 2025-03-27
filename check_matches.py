import pandas as pd

# Read the CSV file
df = pd.read_csv('cuisines.csv')

# Count how many recipes have matched images
total_recipes = len(df)
matched_recipes = df['image_file'].notna().sum()

print(f"\nTotal recipes: {total_recipes}")
print(f"Matched with images: {matched_recipes}")
print(f"Match rate: {matched_recipes/total_recipes*100:.1f}%\n")

print("Sample of matches:")
print("-" * 80)
sample = df.sample(n=5)
for idx, row in sample.iterrows():
    print(f"\nRecipe: {row['name']}")
    print(f"Image:  {row['image_file']}") 