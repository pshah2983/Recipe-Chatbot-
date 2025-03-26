import json
import re
from typing import List, Dict, Optional

class RecipeUtils:
    def __init__(self):
        self.ingredient_substitutions = {
            'eggs': [
                {'name': 'Flax eggs', 'ratio': '1 tbsp ground flax + 3 tbsp water', 'best_for': 'baking'},
                {'name': 'Applesauce', 'ratio': '1/4 cup per egg', 'best_for': 'baking'},
                {'name': 'Mashed banana', 'ratio': '1/4 cup per egg', 'best_for': 'baking'},
                {'name': 'Silken tofu', 'ratio': '1/4 cup per egg', 'best_for': 'baking'},
                {'name': 'Yogurt', 'ratio': '1/4 cup per egg', 'best_for': 'baking'}
            ],
            'milk': [
                {'name': 'Almond milk', 'ratio': '1:1', 'best_for': 'general cooking'},
                {'name': 'Soy milk', 'ratio': '1:1', 'best_for': 'general cooking'},
                {'name': 'Oat milk', 'ratio': '1:1', 'best_for': 'general cooking'},
                {'name': 'Coconut milk', 'ratio': '1:1', 'best_for': 'curries and desserts'},
                {'name': 'Rice milk', 'ratio': '1:1', 'best_for': 'general cooking'}
            ],
            'butter': [
                {'name': 'Olive oil', 'ratio': '3/4 cup per 1 cup butter', 'best_for': 'cooking'},
                {'name': 'Coconut oil', 'ratio': '1:1', 'best_for': 'baking'},
                {'name': 'Applesauce', 'ratio': '1/2 cup per 1 cup butter', 'best_for': 'baking'}
            ]
        }

        self.recipe_categories = {
            'breakfast': ['pancakes', 'waffles', 'oatmeal', 'smoothie', 'toast'],
            'lunch': ['sandwich', 'salad', 'soup', 'wrap', 'pasta'],
            'dinner': ['pizza', 'pasta', 'rice', 'curry', 'stir-fry'],
            'dessert': ['cake', 'cookies', 'pie', 'ice cream', 'pudding'],
            'snacks': ['chips', 'dip', 'nuts', 'fruit', 'vegetables']
        }

    def get_ingredient_substitution(self, ingredient: str) -> Optional[List[Dict]]:
        """Get substitution options for a given ingredient."""
        ingredient = ingredient.lower()
        return self.ingredient_substitutions.get(ingredient)

    def get_recipes_by_category(self, category: str) -> List[str]:
        """Get recipe suggestions for a given category."""
        category = category.lower()
        return self.recipe_categories.get(category, [])

    def extract_ingredients(self, text: str) -> List[str]:
        """Extract ingredients from text using regex patterns."""
        # Common measurement patterns
        patterns = [
            r'\d+\s*(?:cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|oz|ounce|ounces|g|gram|grams|kg|kilogram|kilograms)',
            r'\d+\s*(?:whole|piece|pieces)',
            r'\d+\s*(?:pound|pounds|lb|lbs)'
        ]
        
        ingredients = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Get the word after the measurement
                words = text[match.end():].split()
                if words:
                    ingredients.append(f"{match.group()} {words[0]}")
        
        return ingredients

    def format_recipe_response(self, recipe: Dict) -> str:
        """Format a recipe response in a readable way."""
        response = f"Recipe: {recipe['name']}\n\n"
        response += "Ingredients:\n"
        for ingredient in recipe.get('ingredients', []):
            response += f"- {ingredient}\n"
        
        response += "\nInstructions:\n"
        for i, instruction in enumerate(recipe.get('instructions', []), 1):
            response += f"{i}. {instruction}\n"
        
        if 'nutritional_info' in recipe:
            response += "\nNutritional Information:\n"
            for key, value in recipe['nutritional_info'].items():
                response += f"- {key}: {value}\n"
        
        return response 