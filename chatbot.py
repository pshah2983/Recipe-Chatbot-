import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from fuzzywuzzy import fuzz
from typing import List, Dict
import bcrypt
import json
from datetime import datetime
import os
import glob

class Chatbot:
    def __init__(self, csv_path='cuisines.csv'):
        """Initialize the chatbot with cuisine data."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Load and preprocess data
        self.df = pd.read_csv(csv_path)
        self.stop_words = set(stopwords.words('english'))
        
        # Add difficulty levels if not present
        if 'difficulty' not in self.df.columns:
            self.df['difficulty'] = 'medium'  # default value
        
        # Add cooking time if not present
        if 'cooking_time' not in self.df.columns:
            self.df['cooking_time'] = 30  # default value in minutes
        
        # Initialize recipe number mapping
        self.recipe_number_mapping = {}
        
        # Initialize TF-IDF vectorizer with better parameters
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Use bigrams for better phrase matching
            max_features=5000,   # Reduce features to focus on important terms
            min_df=2,            # Minimum document frequency
            max_df=0.95          # Maximum document frequency
        )
        
        # Prepare text data for TF-IDF
        self.text_columns = ['name', 'description', 'cuisine', 'course', 'diet', 'ingredients', 'instructions']
        self.combined_text = self._prepare_text_data()
        
        # Fit TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(self.combined_text)
        
        # Create response templates
        self.response_templates = {
            'greeting': [
                "Hello! I'm your cuisine assistant. How can I help you today?",
                "Hi there! I can help you discover delicious recipes and cuisines. What would you like to know?",
                "Hey! I'm here to help you explore different cuisines and recipes. What can I tell you about?"
            ],
            'farewell': [
                "Goodbye! Enjoy your culinary adventures!",
                "See you later! Happy cooking!",
                "Bye! Feel free to ask more questions about cuisines anytime!"
            ],
            'thanks': [
                "You're welcome! Let me know if you need more information about cuisines!",
                "No problem! I'm here to help with any cuisine-related questions!",
                "Glad to help! Feel free to ask more questions!"
            ],
            'unknown': [
                "I'm not sure about that. Could you try asking about specific cuisines, ingredients, or cooking methods?",
                "I don't understand. Try asking about recipes, cuisines, or cooking techniques.",
                "I'm not sure what you mean. You can ask about different cuisines, ingredients, or cooking methods."
            ]
        }
        
        # Store current recipe list for numbered requests
        self.current_recipe_list = None
        
        # Initialize user data storage
        self.users = {}
        self.comments = {}
        self.ratings = {}
        
        # Load existing user data if available
        self._load_user_data()

    def _load_user_data(self):
        """Load user data from JSON files."""
        try:
            with open('data/users.json', 'r') as f:
                self.users = json.load(f)
        except FileNotFoundError:
            self.users = {}
        
        try:
            with open('data/comments.json', 'r') as f:
                self.comments = json.load(f)
        except FileNotFoundError:
            self.comments = {}
        
        try:
            with open('data/ratings.json', 'r') as f:
                self.ratings = json.load(f)
        except FileNotFoundError:
            self.ratings = {}

    def _save_user_data(self):
        """Save user data to JSON files."""
        with open('data/users.json', 'w') as f:
            json.dump(self.users, f)
        with open('data/comments.json', 'w') as f:
            json.dump(self.comments, f)
        with open('data/ratings.json', 'w') as f:
            json.dump(self.ratings, f)

    def _prepare_text_data(self):
        """Prepare text data for TF-IDF vectorization."""
        combined_text = []
        for _, row in self.df.iterrows():
            text_parts = []
            for col in self.text_columns:
                if pd.notna(row[col]):
                    # Clean and normalize text
                    text = str(row[col]).lower()
                    text = re.sub(r'[^\w\s]', ' ', text)
                    text = ' '.join(text.split())  # Remove extra whitespace
                    text_parts.append(text)
            combined_text.append(' '.join(text_parts))
        return combined_text

    def _preprocess_query(self, query):
        """Preprocess the user query with improved handling."""
        # Convert to lowercase
        query = query.lower()
        
        # Handle special characters but keep important ones
        query = re.sub(r'[^\w\s&+-]', ' ', query)
        
        # Handle multiple spaces
        query = ' '.join(query.split())
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(query)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)

    def _get_best_match(self, query, top_k=3):
        """Find the best matching cuisine based on TF-IDF similarity."""
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top k matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return top_indices, similarities[top_indices]

    def _get_local_image_path(self, row):
        """Get image path for a recipe using the image_file column or recipe index."""
        try:
            # First try using the image_file column if it exists and has a value
            if 'image_file' in row.index and pd.notna(row['image_file']):
                image_filename = str(row['image_file'])
                if os.path.exists(os.path.join("image_for_cuisines", "data", image_filename)):
                    return f"/static/images/{image_filename}"
            
            # If image_file doesn't work, try using the recipe index
            recipe_idx = row.name + 1  # Convert to 1-based index
            possible_extensions = ['.jpg', '.jpeg', '.png']
            
            # Try different filename patterns
            patterns = [
                f"{recipe_idx}.jpg",  # Simple number.jpg
                f"{recipe_idx}.*",    # Any extension
                f"*{row['name'].lower().replace(' ', '_')}*.jpg"  # Recipe name in filename
            ]
            
            for pattern in patterns:
                # Use glob to find matching files
                matches = glob.glob(os.path.join("image_for_cuisines", "data", pattern))
                if matches:
                    # Get the first matching file
                    image_path = matches[0]
                    filename = os.path.basename(image_path)
                    return f"/static/images/{filename}"
            
            # If no image is found, return a default image path
            return "/static/images/default.jpg"
        
        except Exception as e:
            print(f"Error getting image path for recipe {row['name']}: {str(e)}")
            return "/static/images/default.jpg"

    def _format_recipe_list(self, recipes_df):
        """Format a list of recipes with categories by cuisine and dish type."""
        # Group recipes by cuisine
        cuisines = recipes_df['cuisine'].unique()
        
        response = '<div class="recipe-list">'
        
        for cuisine in cuisines:
            cuisine_recipes = recipes_df[recipes_df['cuisine'] == cuisine]
            
            # Add cuisine header
            response += f'<div class="cuisine-section">'
            response += f'<h3 class="cuisine-title">{cuisine} Cuisine</h3>'
            
            # Group by diet/dish type within each cuisine
            diet_types = cuisine_recipes['diet'].unique()
            for diet in diet_types:
                diet_recipes = cuisine_recipes[cuisine_recipes['diet'] == diet]
                if not diet_recipes.empty:
                    response += f'<div class="diet-section">'
                    response += f'<h4 class="diet-title">{diet}</h4>'
                    response += '<div class="recipe-list-items">'
                    
                    for idx, recipe in diet_recipes.iterrows():
                        # Use the actual index + 1 as the recipe number
                        recipe_number = idx + 1
                        
                        response += f'<div class="recipe-list-item">'
                        response += f'<div class="recipe-number">{recipe_number}</div>'
                        
                        # Add recipe image
                        image_path = self._get_local_image_path(recipe)
                        response += f'<div class="recipe-image"><img src="{image_path}" alt="{recipe["name"]}" class="recipe-thumbnail"></div>'
                        
                        response += f'<div class="recipe-list-info">'
                        response += f'<strong>{recipe["name"]}</strong>'
                        if pd.notna(recipe['course']):
                            response += f'<span class="recipe-course"> ({recipe["course"]})</span>'
                        response += '</div>'
                        response += '</div>'
                        
                        # Store the mapping of actual recipe number to DataFrame index
                        self.recipe_number_mapping[recipe_number] = idx
                    
                    response += '</div>'
                    response += '</div>'  # End diet-section
            
            response += '</div>'  # End cuisine-section
        
        response += '<p class="recipe-instruction"><strong>To view a complete recipe, type "give me recipe number X"</strong> (where X is the number shown next to the recipe)</p>'
        response += '</div>'
        
        # Store the current recipe list for reference
        self.current_recipe_list = recipes_df
        
        return response

    def _format_response(self, cuisine_data):
        """Format the response in a readable way using HTML."""
        response = '<div class="recipe-details">'
        
        # Recipe Title
        response += f'<h2 class="recipe-title">{cuisine_data["name"]}</h2>'
        
        # Basic Information
        response += '<div class="recipe-section">'
        response += f'<h3 class="recipe-section-title">Basic Information:</h3>'
        response += f'<ul class="recipe-info-list">'
        response += f'<li><strong>Cuisine:</strong> {cuisine_data["cuisine"]}</li>'
        response += f'<li><strong>Course:</strong> {cuisine_data["course"]}</li>'
        response += f'<li><strong>Diet:</strong> {cuisine_data["diet"]}</li>'
        if pd.notna(cuisine_data['prep_time']):
            response += f'<li><strong>Preparation Time:</strong> {cuisine_data["prep_time"]}</li>'
        response += '</ul></div>'
        
        # Description
        if pd.notna(cuisine_data['description']):
            response += '<div class="recipe-section">'
            response += f'<h3 class="recipe-section-title">Description:</h3>'
            response += f'<p class="recipe-description">{cuisine_data["description"]}</p>'
            response += '</div>'
        
        # Ingredients
        if pd.notna(cuisine_data['ingredients']):
            response += '<div class="recipe-section">'
            response += f'<h3 class="recipe-section-title">Ingredients:</h3>'
            response += '<ul class="ingredients-list">'
            ingredients = cuisine_data['ingredients'].split(',')
            for ingredient in ingredients:
                response += f'<li>{ingredient.strip()}</li>'
            response += '</ul></div>'
        
        # Instructions
        if pd.notna(cuisine_data['instructions']):
            response += '<div class="recipe-section">'
            response += f'<h3 class="recipe-section-title">Instructions:</h3>'
            response += '<ol class="instructions-list">'
            instructions = cuisine_data['instructions'].split('.')
            for instruction in instructions:
                if instruction.strip():
                    response += f'<li>{instruction.strip()}</li>'
            response += '</ol></div>'
        
        response += '</div>'
        return response

    def fuzzy_ingredient_search(self, ingredient: str, threshold: int = 80) -> List[Dict]:
        """Search for recipes using fuzzy matching on ingredients."""
        ingredient = ingredient.lower()
        matches = []
        
        for _, recipe in self.df.iterrows():
            if pd.isna(recipe['ingredients']):
                continue
            
            ingredients = recipe['ingredients'].lower().split(',')
            for ing in ingredients:
                if fuzz.ratio(ingredient.strip(), ing.strip()) >= threshold:
                    matches.append(recipe)
                    break
        
        return matches

    def multi_ingredient_search(self, ingredients: List[str], require_all: bool = True) -> List[Dict]:
        """Search for recipes containing multiple ingredients."""
        ingredients = [ing.lower() for ing in ingredients]
        matches = []
        
        for _, recipe in self.df.iterrows():
            if pd.isna(recipe['ingredients']):
                continue
            
            recipe_ingredients = recipe['ingredients'].lower()
            found_ingredients = [ing for ing in ingredients if ing in recipe_ingredients]
            
            if require_all and len(found_ingredients) == len(ingredients):
                matches.append(recipe)
            elif not require_all and found_ingredients:
                matches.append(recipe)
        
        return matches

    def search_by_cooking_time(self, max_time: int) -> List[Dict]:
        """Search for recipes that can be prepared within specified time."""
        return self.df[self.df['cooking_time'] <= max_time].to_dict('records')

    def search_by_difficulty(self, difficulty: str) -> List[Dict]:
        """Search for recipes by difficulty level."""
        difficulty = difficulty.lower()
        return self.df[self.df['difficulty'].str.lower() == difficulty].to_dict('records')

    def get_response(self, user_message):
        """Get response for user message with enhanced processing."""
        # Preprocess query
        processed_query = self._preprocess_query(user_message)
        original_query = user_message.lower()
        
        # Check for basic interactions first
        if any(word in original_query for word in ['hello', 'hi', 'hey']):
            return np.random.choice(self.response_templates['greeting'])
        elif any(word in original_query for word in ['bye', 'goodbye', 'see you']):
            return np.random.choice(self.response_templates['farewell'])
        elif any(word in original_query for word in ['thanks', 'thank you']):
            return np.random.choice(self.response_templates['thanks'])
        
        # Check for help request
        if 'help' in original_query or 'what can you do' in original_query:
            return self._get_help_info()
        
        # Check for numbered recipe request first
        number_match = re.search(r'(number|recipe)\s*[#]?\s*(\d+)', original_query)
        if number_match:
            try:
                recipe_number = int(number_match.group(2))
                # First try to get the recipe directly from the DataFrame
                if 1 <= recipe_number <= len(self.df):
                    recipe = self.df.iloc[recipe_number - 1]
                    return self._format_response(recipe)
                else:
                    return f"Sorry, I couldn't find recipe number {recipe_number}. Please try a number between 1 and {len(self.df)}."
            except (ValueError, IndexError):
                return "Sorry, I couldn't find that recipe number. Please try again with a valid recipe number."
        
        # Check for multi-ingredient search
        if ('recipes with' in original_query or 'dishes with' in original_query or 'containing' in original_query or 'made with' in original_query) and ('and' in original_query):
            ingredients = []
            query_parts = original_query.replace(',', ' and ')
            
            # Try to extract ingredients after "with", "containing", or "made with"
            for pattern in [
                r'(?:with|containing|made with)\s+(.*?)(?:$|and[\s,])',
                r'(?:with|containing|made with).*?and\s+(.*?)(?:$|\s+and[\s,])'
            ]:
                matches = re.findall(pattern, original_query)
                if matches:
                    for match in matches:
                        ingredients.extend([ing.strip() for ing in match.split('and')])
            
            # If no matches were found with the above patterns, try a simpler approach
            if not ingredients:
                keywords = ['with', 'containing', 'made with']
                for keyword in keywords:
                    if keyword in original_query:
                        parts = original_query.split(keyword, 1)[1].split('and')
                        ingredients.extend([ing.strip() for ing in parts])
                        break
            
            if ingredients:
                # Clean up ingredients list
                ingredients = [ing.strip() for ing in ingredients if ing.strip()]
                matches = self.multi_ingredient_search(ingredients)
                if matches:
                    self.current_recipe_list = pd.DataFrame(matches)
                    return self._format_recipe_list(self.current_recipe_list)
                else:
                    return f"I couldn't find any recipes containing {', '.join(ingredients)}. Try other ingredients or a broader search."
        
        # Check for cooking time queries
        time_match = re.search(r'(\d+)\s*(?:minute|minutes|min|mins)', original_query)
        if time_match and any(word in original_query for word in ['quick', 'under', 'less than']):
            max_time = int(time_match.group(1))
            matches = self.search_by_cooking_time(max_time)
            if matches:
                self.current_recipe_list = pd.DataFrame(matches)
                return self._format_recipe_list(self.current_recipe_list)
            else:
                return f"I couldn't find recipes that can be prepared in under {max_time} minutes. Try a longer time duration."
        
        # Check for difficulty level queries
        difficulty_levels = ['easy', 'medium', 'hard', 'simple', 'difficult', 'beginner', 'advanced']
        for level in difficulty_levels:
            if level in original_query:
                # Map similar terms to the standardized difficulty levels
                if level == 'simple' or level == 'beginner':
                    level = 'easy'
                elif level == 'difficult' or level == 'advanced':
                    level = 'hard'
                
                matches = self.search_by_difficulty(level)
                if matches:
                    self.current_recipe_list = pd.DataFrame(matches)
                    return self._format_recipe_list(self.current_recipe_list)
                else:
                    return f"I couldn't find any {level} recipes. Try another difficulty level."
        
        # Check for diet-specific queries
        diet_keywords = {
            'vegetarian': ['vegetarian', 'veg', 'no meat', 'veggie'],
            'vegan': ['vegan', 'plant based', 'no animal'],
            'non vegetarian': ['non-vegetarian', 'non veg', 'nonveg', 'meat', 'chicken', 'mutton', 'fish'],
            'gluten free': ['gluten free', 'gluten-free', 'no gluten'],
            'high protein': ['high protein', 'protein rich', 'protein-rich']
        }
        
        for diet_key, keywords in diet_keywords.items():
            if any(kw in original_query for kw in keywords):
                # Use exact matching for diet field
                diet_matches = self.df[self.df['diet'].str.lower() == diet_key.lower()]
                if not diet_matches.empty:
                    self.current_recipe_list = diet_matches
                    return self._format_recipe_list(diet_matches)
                else:
                    return f"I couldn't find any {diet_key} recipes. Try another dietary preference."
        
        # Check for cuisine-specific queries
        common_cuisines = [
            'indian', 'italian', 'chinese', 'mexican', 'thai', 'japanese', 'french',
            'greek', 'spanish', 'moroccan', 'lebanese', 'turkish', 'american', 'british',
            'punjabi', 'bengali', 'gujarati', 'south indian', 'north indian', 'maharashtrian',
            'kerala', 'tamil', 'andhra', 'rajasthani', 'kashmiri', 'goan'
        ]
        
        for cuisine in common_cuisines:
            if cuisine in original_query:
                cuisine_matches = self.df[self.df['cuisine'].str.lower().str.contains(cuisine, na=False)]
                if not cuisine_matches.empty:
                    self.current_recipe_list = cuisine_matches
                    return self._format_recipe_list(cuisine_matches)
        
        # Check for course-specific queries
        course_keywords = {
            'breakfast': ['breakfast', 'morning meal'],
            'lunch': ['lunch', 'midday meal'],
            'dinner': ['dinner', 'evening meal', 'supper'],
            'snack': ['snack', 'evening snack', 'tea time'],
            'dessert': ['dessert', 'sweet', 'pudding', 'ice cream'],
            'appetizer': ['appetizer', 'starter', 'first course'],
            'main course': ['main course', 'main dish', 'entree']
        }
        
        for course_key, keywords in course_keywords.items():
            if any(kw in original_query for kw in keywords):
                course_matches = self.df[self.df['course'].str.lower().str.contains(course_key, na=False)]
                if not course_matches.empty:
                    self.current_recipe_list = course_matches
                    return self._format_recipe_list(course_matches)
                else:
                    return f"I couldn't find any {course_key} recipes. Try another meal type."
        
        # Check for single-ingredient queries
        if any(word in original_query for word in ['ingredient', 'ingredients', 'contains', 'made with', 'using', 'with']):
            # Try to extract the ingredient name
            ingredient_patterns = [
                r'(?:with|contains|using|made with)\s+([a-zA-Z ]+)(?:$|[,\.])',
                r'recipes (?:with|contains|using|made with)\s+([a-zA-Z ]+)(?:$|[,\.])',
                r'dishes (?:with|contains|using|made with)\s+([a-zA-Z ]+)(?:$|[,\.])'
            ]
            
            ingredient = None
            for pattern in ingredient_patterns:
                match = re.search(pattern, original_query)
                if match:
                    ingredient = match.group(1).strip()
                    break
            
            if ingredient:
                # Use fuzzy matching for ingredient
                matches = self.fuzzy_ingredient_search(ingredient)
                if matches:
                    self.current_recipe_list = pd.DataFrame(matches)
                    return self._format_recipe_list(self.current_recipe_list)
                else:
                    # Fallback to basic string matching if fuzzy fails
                    ingredient_matches = self.df[self.df['ingredients'].str.lower().str.contains(ingredient, na=False)]
                    if not ingredient_matches.empty:
                        self.current_recipe_list = ingredient_matches
                        return self._format_recipe_list(ingredient_matches)
                    else:
                        return f"I couldn't find any recipes with {ingredient}. Try another ingredient."
        
        # Get best matching cuisine using TF-IDF
        top_indices, similarities = self._get_best_match(processed_query)
        
        # Set a higher similarity threshold for better accuracy
        similarity_threshold = 0.2
        if similarities[0] < similarity_threshold:
            # If low similarity, try keyword matching with common terms
            food_keywords = ['recipe', 'dish', 'food', 'cuisine', 'meal', 'cook', 'prepare']
            if any(keyword in original_query for keyword in food_keywords):
                # Return a general recommendation
                random_recipes = self.df.sample(min(10, len(self.df)))
                self.current_recipe_list = random_recipes
                return (
                    "I'm not sure exactly what you're looking for, but here are some recipes you might like:"
                    + self._format_recipe_list(random_recipes)
                )
            else:
                return np.random.choice(self.response_templates['unknown'])
        
        # Get the best match and return formatted response
        best_match = self.df.iloc[top_indices[0]]
        return self._format_response(best_match)
    
    def _get_help_info(self):
        """Return help information about what the chatbot can do."""
        help_text = """
        <div class="help-info">
            <h3>How to Use the Cuisine Chatbot:</h3>
            <ul>
                <li><strong>Search by cuisine:</strong> "Show me Indian recipes" or "I want Italian food"</li>
                <li><strong>Search by ingredients:</strong> "Recipes with rice and beans" or "Dishes containing paneer"</li>
                <li><strong>Search by diet:</strong> "Show vegetarian recipes" or "Non-veg dishes"</li>
                <li><strong>Search by course:</strong> "Breakfast recipes" or "Show me some desserts"</li>
                <li><strong>Search by cooking time:</strong> "Quick recipes under 30 minutes"</li>
                <li><strong>Search by difficulty:</strong> "Easy recipes" or "Show me some advanced dishes"</li>
                <li><strong>View specific recipe:</strong> After seeing a list, type "Give me recipe number 3"</li>
            </ul>
            <p>Try asking about specific cuisines or ingredients you're interested in!</p>
        </div>
        """
        return help_text

    def _format_recipe(self, recipe):
        """Format a single recipe for display."""
        image_path = self._get_local_image_path(recipe)
        
        formatted_recipe = f"""
        <div class='recipe-details'>
            <div class='recipe-header'>
                <div class='recipe-image-container'>
                    <img src='{image_path}' alt='{recipe['name']}' class='recipe-header-image'>
                </div>
                <div class='recipe-header-info'>
                    <h2 class='recipe-title'>{recipe['name']}</h2>
                    <div class='basic-info'>
                        <p><strong>Cuisine:</strong> {recipe['cuisine']}</p>
                        <p><strong>Course:</strong> {recipe['course']}</p>
                        <p><strong>Diet:</strong> {recipe['diet']}</p>
                        <p><strong>Preparation Time:</strong> {recipe['prep_time']}</p>
                    </div>
                </div>
            </div>
            <div class='recipe-section'>
                <h3 class='recipe-section-title'>Description</h3>
                <p class='recipe-description'>{recipe['description']}</p>
            </div>
            <div class='recipe-section'>
                <h3 class='recipe-section-title'>Ingredients</h3>
                <ul class='ingredients-list'>
                    {self._format_ingredients(recipe['ingredients'])}
                </ul>
            </div>
            <div class='recipe-section'>
                <h3 class='recipe-section-title'>Instructions</h3>
                <ol class='instructions-list'>
                    {self._format_instructions(recipe['instructions'])}
                </ol>
            </div>
        </div>
        """
        return formatted_recipe
