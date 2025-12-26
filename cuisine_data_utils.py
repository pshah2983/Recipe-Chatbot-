import pandas as pd
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

class CuisineDataUtils:
    def __init__(self, csv_path: str = 'cuisines.csv'):
        """Initialize with the cuisine dataset."""
        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings for all text columns
        self.text_columns = ['name', 'description', 'cuisine', 'course', 'diet', 'ingredients', 'instructions']
        self.embeddings = {}
        for column in self.text_columns:
            self.embeddings[column] = self.model.encode(self.df[column].fillna('').astype(str))

    def get_cuisine_info(self, cuisine_name: str) -> Optional[Dict]:
        """Get information about a specific cuisine."""
        cuisine_name = cuisine_name.lower()
        mask = self.df['cuisine'].str.lower() == cuisine_name
        if not mask.any():
            return None
        
        cuisine_data = self.df[mask].iloc[0]
        return cuisine_data.to_dict()

    def search_by_ingredient(self, ingredient: str) -> List[Dict]:
        """Find cuisines that use a specific ingredient."""
        ingredient = ingredient.lower()
        mask = self.df['ingredients'].str.lower().str.contains(ingredient, na=False)
        return self.df[mask].to_dict('records')

    def search_by_cooking_technique(self, technique: str) -> List[Dict]:
        """Find cuisines that use a specific cooking technique."""
        technique = technique.lower()
        # Search in instructions for cooking techniques
        mask = self.df['instructions'].str.lower().str.contains(technique, na=False)
        return self.df[mask].to_dict('records')

    def get_similar_cuisines(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find cuisines similar to the query using semantic search."""
        query_embedding = self.model.encode([query])[0]
        
        # Combine all text columns for similarity search
        combined_text = self.df[self.text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        combined_embeddings = self.model.encode(combined_text)
        
        # Calculate similarities
        similarities = np.dot(combined_embeddings, query_embedding) / (
            np.linalg.norm(combined_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k most similar cuisines
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return self.df.iloc[top_indices].to_dict('records')

    def get_unique_cuisines(self) -> List[str]:
        """Get a list of all unique cuisines in the dataset."""
        return self.df['cuisine'].unique().tolist()

    def format_cuisine_response(self, cuisine_data: Dict) -> str:
        """Format cuisine information in a readable way."""
        response = f"Recipe: {cuisine_data.get('name', 'Unknown')}\n"
        response += f"Cuisine: {cuisine_data.get('cuisine', 'Unknown')}\n"
        response += f"Course: {cuisine_data.get('course', 'Not specified')}\n"
        response += f"Diet: {cuisine_data.get('diet', 'Not specified')}\n"
        response += f"Preparation Time: {cuisine_data.get('prep_time', 'Not specified')}\n\n"
        
        # Add description if available
        if 'description' in cuisine_data and pd.notna(cuisine_data['description']):
            response += f"Description: {cuisine_data['description']}\n\n"
        
        # Add ingredients if available
        if 'ingredients' in cuisine_data and pd.notna(cuisine_data['ingredients']):
            response += "Ingredients:\n"
            ingredients = cuisine_data['ingredients'].split(',') if isinstance(cuisine_data['ingredients'], str) else cuisine_data['ingredients']
            for ingredient in ingredients:
                response += f"- {ingredient.strip()}\n"
            response += "\n"
        
        # Add instructions if available
        if 'instructions' in cuisine_data and pd.notna(cuisine_data['instructions']):
            response += "Instructions:\n"
            instructions = cuisine_data['instructions'].split('.') if isinstance(cuisine_data['instructions'], str) else cuisine_data['instructions']
            for i, instruction in enumerate(instructions, 1):
                if instruction.strip():
                    response += f"{i}. {instruction.strip()}\n"
        
        return response 