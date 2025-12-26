import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from fuzzywuzzy import fuzz
from typing import List, Dict, Optional, Any
import bcrypt
import json
from datetime import datetime
import os
import glob
import spacy
from sentence_transformers import SentenceTransformer
from themealdb_client import TheMealDBClient
from llm_service import llm_service, recipe_scaler, LLMService, RecipeScaler

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
        
        # NLU and semantic search setup
        self.spacy_nlp = spacy.load('en_core_web_sm')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.recipe_embeddings = self.sentence_model.encode(self.combined_text, show_progress_bar=False)
        self.themealdb = TheMealDBClient()
        
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
        self.current_recipe = None  # Store the last shown recipe for follow-ups
        
        # Initialize user data storage
        self.users = {}
        self.comments = {}
        self.ratings = {}
        
        # LLM Service for enhanced responses
        self.llm_service = llm_service
        self.recipe_scaler = recipe_scaler
        
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

    def extract_intent_entities(self, query):
        doc = self.spacy_nlp(query)
        entities = [(ent.label_, ent.text) for ent in doc.ents]
        return entities

    def semantic_search(self, query, top_k=3):
        query_emb = self.sentence_model.encode([query])[0]
        similarities = np.dot(self.recipe_embeddings, query_emb) / (
            np.linalg.norm(self.recipe_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8)
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

    def get_response(self, user_query, user_name=None):
        """Get response for user message with enhanced processing and personalization. Always try TheMealDB API first for recipe queries, then fall back to local CSV, and indicate data source."""
        original_query = user_query.lower()
        processed_query = self._preprocess_query(user_query)

        # Personalized greetings
        if any(word in original_query for word in ['hello', 'hi', 'hey']):
            if user_name:
                return f"Hello, {user_name}! How can I help you today?"
            return np.random.choice(self.response_templates['greeting'])
        elif any(word in original_query for word in ['bye', 'goodbye', 'see you']):
            return np.random.choice(self.response_templates['farewell'])
        elif any(word in original_query for word in ['thanks', 'thank you']):
            return np.random.choice(self.response_templates['thanks'])

        # Check for help request
        if 'help' in original_query or 'what can you do' in original_query:
            return self._get_help_info()

        # Check for follow-up queries about current recipe (health, substitutions, scaling, tips)
        follow_up_response = self.process_follow_up(user_query, user_name)
        if follow_up_response:
            return follow_up_response

        # Check for numbered recipe request first (local only)
        number_match = re.search(r'(number|recipe)\s*[#]?\s*(\d+)', original_query)
        if number_match:
            try:
                recipe_number = int(number_match.group(2))
                if 1 <= recipe_number <= len(self.df):
                    recipe = self.df.iloc[recipe_number - 1]
                    self.current_recipe = recipe  # Store for follow-up queries
                    return self._format_response(recipe) + '<br><i>(From local recipe database)</i>'
                else:
                    return f"Sorry, I couldn't find recipe number {recipe_number}. Please try a number between 1 and {len(self.df)}."
            except (ValueError, IndexError):
                return "Sorry, I couldn't find that recipe number. Please try again with a valid recipe number."

        # --- Improved cuisine and course (meal type) handling ---
        # Detect cuisine and course/meal type in the query
        common_cuisines = [
            'indian', 'italian', 'chinese', 'mexican', 'thai', 'japanese', 'french',
            'greek', 'spanish', 'moroccan', 'lebanese', 'turkish', 'american', 'british',
            'punjabi', 'bengali', 'gujarati', 'south indian', 'north indian', 'maharashtrian',
            'kerala', 'tamil', 'andhra', 'rajasthani', 'kashmiri', 'goan'
        ]
        course_keywords = {
            'breakfast': ['breakfast', 'morning meal'],
            'lunch': ['lunch', 'midday meal'],
            'dinner': ['dinner', 'evening meal', 'supper'],
            'snack': ['snack', 'evening snack', 'tea time'],
            'dessert': ['dessert', 'sweet', 'pudding', 'ice cream'],
            'appetizer': ['appetizer', 'starter', 'first course'],
            'main course': ['main course', 'main dish', 'entree']
        }
        found_cuisine = None
        found_course = None
        for cuisine in common_cuisines:
            if cuisine in original_query:
                found_cuisine = cuisine
                break
        for course_key, keywords in course_keywords.items():
            if any(kw in original_query for kw in keywords):
                found_course = course_key
                break
        # If both cuisine and course are found, try to get recipes for both
        if found_cuisine:
            # Try TheMealDB API for cuisine (and course if possible)
            api_results = self.themealdb.search_by_name(found_cuisine)
            filtered_api_meals = []
            if api_results and found_course:
                # Try to filter API results by course/meal type if possible
                for meal in api_results:
                    # TheMealDB uses 'strCategory' for meal type, e.g., 'Breakfast', 'Dessert', etc.
                    if 'strCategory' in meal and found_course.lower() in meal['strCategory'].lower():
                        filtered_api_meals.append(meal)
            if found_course and filtered_api_meals:
                meal_names = ', '.join([meal['strMeal'] for meal in filtered_api_meals[:5]])
                greeting = f"{user_name}, here are some {found_cuisine.title()} {found_course.title()} recipes from TheMealDB:" if user_name else f"Here are some {found_cuisine.title()} {found_course.title()} recipes from TheMealDB:"
                return f"{greeting}<br>{meal_names}<br><i>(From TheMealDB API)</i>"
            elif api_results:
                meal_names = ', '.join([meal['strMeal'] for meal in api_results[:5]])
                greeting = f"{user_name}, here are some {found_cuisine.title()} recipes from TheMealDB:" if user_name else f"Here are some {found_cuisine.title()} recipes from TheMealDB:"
                return f"{greeting}<br>{meal_names}<br><i>(From TheMealDB API)</i>"
            # Fallback to local CSV for cuisine and course
            local_matches = self.df[self.df['cuisine'].str.lower().str.contains(found_cuisine, na=False)]
            if found_course:
                local_matches = local_matches[local_matches['course'].str.lower().str.contains(found_course, na=False)]
            if not local_matches.empty:
                self.current_recipe_list = local_matches
                label = f"{found_cuisine.title()} {found_course.title()}" if found_course else found_cuisine.title()
                return self._format_recipe_list(local_matches) + f'<br><i>(From local recipe database: {label} recipes)</i>'
            # If nothing found, show a friendly message
            label = f"{found_cuisine.title()} {found_course.title()}" if found_course else found_cuisine.title()
            return f"Sorry, I couldn't find any {label} recipes in my database. Try another cuisine or meal type."

        # Multi-ingredient search (try API first, then local)
        if (('recipes with' in original_query or 'dishes with' in original_query or 'containing' in original_query or 'made with' in original_query) and ('and' in original_query)):
            ingredients = []
            query_parts = original_query.replace(',', ' and ')
            for pattern in [
                r'(?:with|containing|made with)\s+(.*?)(?:$|and[\s,])',
                r'(?:with|containing|made with).*?and\s+(.*?)(?:$|\s+and[\s,])'
            ]:
                matches = re.findall(pattern, original_query)
                if matches:
                    for match in matches:
                        ingredients.extend([ing.strip() for ing in match.split('and')])
            if not ingredients:
                keywords = ['with', 'containing', 'made with']
                for keyword in keywords:
                    if keyword in original_query:
                        parts = original_query.split(keyword, 1)[1].split('and')
                        ingredients.extend([ing.strip() for ing in parts])
                        break
            ingredients = [ing.strip() for ing in ingredients if ing.strip()]
            # Try TheMealDB API for each ingredient and combine results
            api_meals = []
            for ing in ingredients:
                api_results = self.themealdb.search_by_ingredient(ing)
                if api_results:
                    api_meals.extend(api_results)
            if api_meals:
                meal_names = ', '.join([meal['strMeal'] for meal in api_meals[:5]])
                greeting = f"{user_name}, here are recipes from TheMealDB with your ingredients:" if user_name else "Here are recipes from TheMealDB with your ingredients:"
                return f"{greeting}<br>{meal_names}<br><i>(From TheMealDB API)</i>"
            # Fallback to local
            matches = self.multi_ingredient_search(ingredients)
            if matches:
                self.current_recipe_list = pd.DataFrame(matches)
                return self._format_recipe_list(self.current_recipe_list) + '<br><i>(From local recipe database)</i>'
            else:
                return f"I couldn't find any recipes containing {', '.join(ingredients)}. Try other ingredients or a broader search."

        # Cooking time queries (local only)
        time_match = re.search(r'(\d+)\s*(?:minute|minutes|min|mins)', original_query)
        if time_match and any(word in original_query for word in ['quick', 'under', 'less than']):
            max_time = int(time_match.group(1))
            matches = self.search_by_cooking_time(max_time)
            if matches:
                self.current_recipe_list = pd.DataFrame(matches)
                return self._format_recipe_list(self.current_recipe_list) + '<br><i>(From local recipe database)</i>'
            else:
                return f"I couldn't find recipes that can be prepared in under {max_time} minutes. Try a longer time duration."

        # Difficulty level queries (local only)
        difficulty_levels = ['easy', 'medium', 'hard', 'simple', 'difficult', 'beginner', 'advanced']
        for level in difficulty_levels:
            if level in original_query:
                if level == 'simple' or level == 'beginner':
                    level = 'easy'
                elif level == 'difficult' or level == 'advanced':
                    level = 'hard'
                matches = self.search_by_difficulty(level)
                if matches:
                    self.current_recipe_list = pd.DataFrame(matches)
                    return self._format_recipe_list(self.current_recipe_list) + '<br><i>(From local recipe database)</i>'
                else:
                    return f"I couldn't find any {level} recipes. Try another difficulty level."

        # Diet-specific queries (local only)
        diet_keywords = {
            'vegetarian': ['vegetarian', 'veg', 'no meat', 'veggie'],
            'vegan': ['vegan', 'plant based', 'no animal'],
            'non vegetarian': ['non-vegetarian', 'non veg', 'nonveg', 'meat', 'chicken', 'mutton', 'fish'],
            'gluten free': ['gluten free', 'gluten-free', 'no gluten'],
            'high protein': ['high protein', 'protein rich', 'protein-rich']
        }
        for diet_key, keywords in diet_keywords.items():
            if any(kw in original_query for kw in keywords):
                diet_matches = self.df[self.df['diet'].str.lower() == diet_key.lower()]
                if not diet_matches.empty:
                    self.current_recipe_list = diet_matches
                    return self._format_recipe_list(diet_matches) + '<br><i>(From local recipe database)</i>'
                else:
                    return f"I couldn't find any {diet_key} recipes. Try another dietary preference."

        # Cuisine-specific queries (try API first, then local)
        common_cuisines = [
            'indian', 'italian', 'chinese', 'mexican', 'thai', 'japanese', 'french',
            'greek', 'spanish', 'moroccan', 'lebanese', 'turkish', 'american', 'british',
            'punjabi', 'bengali', 'gujarati', 'south indian', 'north indian', 'maharashtrian',
            'kerala', 'tamil', 'andhra', 'rajasthani', 'kashmiri', 'goan'
        ]
        for cuisine in common_cuisines:
            if cuisine in original_query:
                # Try TheMealDB API for cuisine
                api_results = self.themealdb.search_by_name(cuisine)
                if api_results:
                    meal_names = ', '.join([meal['strMeal'] for meal in api_results[:5]])
                    greeting = f"{user_name}, here are some {cuisine.title()} recipes from TheMealDB:" if user_name else f"Here are some {cuisine.title()} recipes from TheMealDB:"
                    return f"{greeting}<br>{meal_names}<br><i>(From TheMealDB API)</i>"
                # Fallback to local
                cuisine_matches = self.df[self.df['cuisine'].str.lower().str.contains(cuisine, na=False)]
                if not cuisine_matches.empty:
                    self.current_recipe_list = cuisine_matches
                    return self._format_recipe_list(cuisine_matches) + '<br><i>(From local recipe database)</i>'

        # Course-specific queries (local only)
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
                    return self._format_recipe_list(self.current_recipe_list) + '<br><i>(From local recipe database)</i>'
                else:
                    return f"I couldn't find any {course_key} recipes. Try another meal type."

        # Single-ingredient queries (try API first, then local)
        if any(word in original_query for word in ['ingredient', 'ingredients', 'contains', 'made with', 'using', 'with']):
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
                api_results = self.themealdb.search_by_ingredient(ingredient)
                if api_results:
                    meal_names = ', '.join([meal['strMeal'] for meal in api_results[:5]])
                    greeting = f"{user_name}, here are recipes from TheMealDB with {ingredient}:" if user_name else f"Here are recipes from TheMealDB with {ingredient}:"
                    return f"{greeting}<br>{meal_names}<br><i>(From TheMealDB API)</i>"
                matches = self.fuzzy_ingredient_search(ingredient)
                if matches:
                    self.current_recipe_list = pd.DataFrame(matches)
                    return self._format_recipe_list(self.current_recipe_list) + '<br><i>(From local recipe database)</i>'
                else:
                    ingredient_matches = self.df[self.df['ingredients'].str.lower().str.contains(ingredient, na=False)]
                    if not ingredient_matches.empty:
                        self.current_recipe_list = ingredient_matches
                        return self._format_recipe_list(ingredient_matches) + '<br><i>(From local recipe database)</i>'
                    else:
                        return f"I couldn't find any recipes with {ingredient}. Try another ingredient."

        # General recipe name search (try API first, then local)
        api_results = self.themealdb.search_by_name(user_query)
        if api_results:
            meal = api_results[0]
            greeting = f"{user_name}, here's a recipe from TheMealDB!" if user_name else "Here's a recipe from TheMealDB!"
            return f"{greeting}<br><b>{meal['strMeal']}</b><br>Category: {meal['strCategory']}<br>Instructions: {meal['strInstructions']}<br><i>(From TheMealDB API)</i>"

        # Fallback: best match from local CSV
        top_indices, similarities = self._get_best_match(processed_query)
        similarity_threshold = 0.2
        if similarities[0] < similarity_threshold:
            food_keywords = ['recipe', 'dish', 'food', 'cuisine', 'meal', 'cook', 'prepare']
            if any(keyword in original_query for keyword in food_keywords):
                random_recipes = self.df.sample(min(10, len(self.df)))
                self.current_recipe_list = random_recipes
                return (
                    "I'm not sure exactly what you're looking for, but here are some recipes you might like:" +
                    self._format_recipe_list(random_recipes) + '<br><i>(From local recipe database)</i>'
                )
            else:
                return np.random.choice(self.response_templates['unknown'])
        best_match = self.df.iloc[top_indices[0]]
        response = self._format_response(best_match)
        if user_name:
            response = f"<b>{user_name}, here's a recipe you might like:</b><br>" + response
        return response + '<br><i>(From local recipe database)</i>'
    
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

    def _format_ingredients(self, ingredients):
        """Format ingredients as HTML list items."""
        if pd.isna(ingredients):
            return "<li>Ingredients not available</li>"
        
        if isinstance(ingredients, str):
            ingredients = ingredients.split(',')
        
        return ''.join([f"<li>{ing.strip()}</li>" for ing in ingredients if ing.strip()])
    
    def _format_instructions(self, instructions):
        """Format instructions as HTML list items."""
        if pd.isna(instructions):
            return "<li>Instructions not available</li>"
        
        if isinstance(instructions, str):
            instructions = instructions.split('.')
        
        return ''.join([f"<li>{step.strip()}</li>" for step in instructions if step.strip()])

    def get_health_explanation(self, recipe: Dict = None) -> str:
        """Get health explanation for the current or specified recipe."""
        recipe_data = recipe if recipe is not None else self.current_recipe
        
        # Check if we have recipe data (handle None and pandas Series/DataFrame)
        if recipe_data is None:
            return "I don't have a recipe selected. Please search for a recipe first!"
        
        # Convert pandas Series to dict if needed
        if hasattr(recipe_data, 'to_dict'):
            recipe_data = recipe_data.to_dict()
        
        explanation = self.llm_service.explain_recipe_health(recipe_data)
        
        response = f"""
        <div class="health-explanation">
            <div class="icon">🌿</div>
            <h4>Health Benefits of {recipe_data.get('name', 'this recipe')}:</h4>
            <p>{explanation}</p>
        </div>
        """
        
        return response
    
    def get_ingredient_substitutions(self, ingredient: str = None) -> str:
        """Get ingredient substitutions."""
        if not ingredient and self.current_recipe is not None:
            # Get substitutions for main ingredients in current recipe
            recipe_data = self.current_recipe if isinstance(self.current_recipe, dict) else self.current_recipe.to_dict()
            ingredients = recipe_data.get('ingredients', '')
            
            # Find common substitutable ingredients
            common_substitutes = ['eggs', 'milk', 'butter', 'cream', 'paneer', 'chicken', 'rice', 'sugar', 'flour']
            found_ingredients = []
            for sub in common_substitutes:
                if sub in ingredients.lower():
                    found_ingredients.append(sub)
            
            if not found_ingredients:
                return "I couldn't find any common ingredients to substitute in this recipe. Ask me about a specific ingredient!"
            
            ingredient = found_ingredients[0]  # Use the first found
        
        substitutions = self.llm_service.get_ingredient_substitutions(ingredient)
        
        response = f"""
        <div class="substitution-info">
            <h4>🔄 Substitutions for {ingredient.title()}:</h4>
            <ul class="substitution-list">
        """
        
        for sub in substitutions:
            response += f"""
                <li>
                    <strong>{sub['name']}</strong> 
                    <span class="sub-ratio">({sub['ratio']})</span>
                    <span class="sub-type badge">{sub['type']}</span>
                </li>
            """
        
        response += """
            </ul>
        </div>
        """
        
        return response
    
    def scale_recipe(self, target_servings: int = None, query: str = None) -> str:
        """Scale a recipe to different number of servings."""
        if self.current_recipe is None:
            return "Please select a recipe first before scaling!"
        
        recipe_data = self.current_recipe if isinstance(self.current_recipe, dict) else self.current_recipe.to_dict()
        
        # Try to extract target servings from query
        if not target_servings and query:
            target_servings = self.recipe_scaler.extract_servings_from_query(query)
        
        if not target_servings:
            return "How many servings would you like? Try saying 'Make this for 8 people' or 'Scale to 6 servings'."
        
        # Assume default 4 servings if not specified
        original_servings = 4
        
        scaled_ingredients = self.recipe_scaler.scale_ingredients(
            recipe_data.get('ingredients', ''),
            original_servings,
            target_servings
        )
        
        response = f"""
        <div class="scaled-recipe">
            <h4>📐 {recipe_data.get('name', 'Recipe')} - Scaled for {target_servings} servings:</h4>
            <p class="scale-info">Original: {original_servings} servings → New: {target_servings} servings</p>
            <div class="scaled-ingredients">
                <h5>Adjusted Ingredients:</h5>
                <ul class="ingredients-list">
        """
        
        for ing in scaled_ingredients.split('\n') if '\n' in scaled_ingredients else scaled_ingredients.split(','):
            if ing.strip():
                response += f"<li>{ing.strip()}</li>"
        
        response += """
                </ul>
            </div>
        </div>
        """
        
        return response
    
    def get_cooking_tips(self) -> str:
        """Get contextual cooking tips for the current recipe."""
        if self.current_recipe is None:
            return "Show me a recipe first, and I'll give you some cooking tips!"
        
        recipe_data = self.current_recipe if isinstance(self.current_recipe, dict) else self.current_recipe.to_dict()
        tips = self.llm_service.generate_cooking_tips(recipe_data)
        
        if not tips:
            return "💡 Read through all instructions before starting to cook!"
        
        response = """
        <div class="cooking-tips">
            <h4>💡 Pro Tips for This Recipe:</h4>
            <ul>
        """
        
        for tip in tips:
            response += f"<li>{tip}</li>"
        
        response += """
            </ul>
        </div>
        """
        
        return response
    
    def get_similar_recipes(self) -> str:
        """Find recipes similar to the current one."""
        if self.current_recipe is None:
            return "Please view a recipe first, then ask for similar recipes!"
        
        recipe_data = self.current_recipe if isinstance(self.current_recipe, dict) else self.current_recipe.to_dict()
        
        # Use semantic search to find similar recipes
        query = f"{recipe_data.get('cuisine', '')} {recipe_data.get('course', '')} {recipe_data.get('diet', '')}"
        top_indices, _ = self.semantic_search(query, top_k=5)
        
        similar_recipes = self.df.iloc[top_indices]
        
        # Exclude the current recipe if it's in the results
        if 'name' in recipe_data:
            similar_recipes = similar_recipes[similar_recipes['name'] != recipe_data['name']]
        
        if similar_recipes.empty:
            return "I couldn't find similar recipes. Try a different search!"
        
        self.current_recipe_list = similar_recipes
        return self._format_recipe_list(similar_recipes.head(5)) + '<br><i>(Similar recipes from local database)</i>'
    
    def _build_json_response(self, response_text: str, recipe_data: Dict = None, source: str = None) -> Dict[str, Any]:
        """Build a structured JSON response for the frontend."""
        return {
            'response': response_text,
            'recipe': recipe_data,
            'source': source,
            'suggestions': self._get_follow_up_suggestions(recipe_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_follow_up_suggestions(self, recipe_data: Dict = None) -> List[Dict]:
        """Generate smart follow-up suggestions based on context."""
        suggestions = []
        
        if recipe_data:
            suggestions = [
                {'label': 'Want substitutions?', 'icon': '🔄', 'action': 'substitutions'},
                {'label': 'Similar recipes', 'icon': '👀', 'action': 'similar'},
                {'label': 'Health benefits', 'icon': '💚', 'action': 'healthy'},
                {'label': 'Scale recipe', 'icon': '📐', 'action': 'scale'},
                {'label': 'Cooking tips', 'icon': '💡', 'action': 'tips'},
            ]
        else:
            suggestions = [
                {'label': 'Show me Indian recipes', 'icon': '🍛', 'query': 'Show me Indian recipes'},
                {'label': 'Quick breakfast ideas', 'icon': '🌅', 'query': 'Quick breakfast recipes'},
                {'label': 'Vegetarian dishes', 'icon': '🥗', 'query': 'Vegetarian recipes'},
                {'label': 'Healthy dinner', 'icon': '🥦', 'query': 'Healthy dinner recipes'},
            ]
        
        return suggestions
    
    def process_follow_up(self, query: str, user_name: str = None) -> str:
        """Process follow-up queries about the current recipe."""
        query_lower = query.lower()
        
        # Health explanation
        if any(word in query_lower for word in ['healthy', 'health', 'nutritious', 'benefits', 'good for']):
            return self.get_health_explanation()
        
        # Substitutions
        if any(word in query_lower for word in ['substitut', 'replace', 'instead of', 'alternative']):
            # Try to extract specific ingredient
            ingredient_match = re.search(r'(?:substitute|replace|alternative for|instead of)\s+(\w+)', query_lower)
            ingredient = ingredient_match.group(1) if ingredient_match else None
            return self.get_ingredient_substitutions(ingredient)
        
        # Recipe scaling
        if any(word in query_lower for word in ['scale', 'servings', 'people', 'portions', 'double', 'half']):
            if 'double' in query_lower:
                return self.scale_recipe(target_servings=8)
            elif 'half' in query_lower:
                return self.scale_recipe(target_servings=2)
            else:
                return self.scale_recipe(query=query)
        
        # Similar recipes
        if any(word in query_lower for word in ['similar', 'like this', 'more like', 'related']):
            return self.get_similar_recipes()
        
        # Cooking tips
        if any(word in query_lower for word in ['tip', 'tips', 'advice', 'trick']):
            return self.get_cooking_tips()
        
        return None  # Not a follow-up query
