# Indian Cuisine Recipe Chatbot

A sophisticated retrieval-based chatbot that helps users discover and explore Indian recipes. The chatbot features an intelligent recipe search system with image matching, diet preferences, and an intuitive user interface.

## Features

### 1. Intelligent Recipe Search
- Natural language query processing
- Search by recipe name, cuisine type, or ingredients
- Diet preference filtering (Vegetarian, Non-Vegetarian, Vegan)
- Course-based filtering (Breakfast, Main Course, Dessert, etc.)
- Fuzzy matching for recipe names and ingredients

### 2. Advanced Image Handling
- 98.8% recipe-to-image match rate
- Intelligent image matching using:
  - Word overlap scoring (70% weight)
  - String similarity matching (30% weight)
  - Fallback mechanisms for unmatched recipes
- Default image support for rare unmatched cases

### 3. User Interface
- Clean, modern design
- Dark/Light mode support
- Responsive layout for all devices
- Interactive recipe cards with hover effects
- Modal view for detailed recipe information
- Side-by-side image and recipe information display

### 4. Recipe Information Display
- Recipe name and image
- Basic information (Cuisine, Course, Diet, Prep Time)
- Detailed description
- Ingredient list
- Step-by-step instructions
- Visual presentation of recipe details

## Technical Implementation

### 1. Core Components
- `app.py`: Flask application setup and routing
- `chatbot.py`: Main chatbot logic and recipe processing
- `update_image_files.py`: Image matching algorithm
- `cuisines.csv`: Recipe database
- Static files: Images, CSS, and JavaScript

### 2. Image Matching System
```python
def find_matching_image(recipe_name):
    # Clean and normalize recipe name
    clean_recipe = clean_name(recipe_name)
    recipe_words = get_words(recipe_name)
    
    # Find best match using combined scoring
    for img_file, info in image_info.items():
        # Word overlap score (70% weight)
        common_words = recipe_words.intersection(info['words'])
        word_score = len(common_words) / max(len(recipe_words), len(info['words']))
        
        # String similarity score (30% weight)
        sim_score = similar(clean_recipe, info['clean_name'])
        
        # Combined weighted score
        score = 0.7 * word_score + 0.3 * sim_score
```

### 3. Recipe Processing
- Recipe information extraction
- Diet and course categorization
- Ingredient parsing and formatting
- Instruction formatting
- Image path resolution

### 4. User Interface Components
```html
<!-- Recipe Card Structure -->
<div class="recipe-card">
    <img src="${recipe.image}" alt="${recipe.name}" class="recipe-image">
    <h3>${recipe.name}</h3>
    <p>${recipe.description}</p>
    <div class="recipe-meta">
        <span class="cuisine-tag">${recipe.cuisine}</span>
        <span class="diet-tag">${recipe.diet}</span>
    </div>
</div>

<!-- Recipe Detail Modal -->
<div class="recipe-header">
    <div class="recipe-image-container">
        <img src="${image_path}" alt="${recipe_name}" class="recipe-header-image">
    </div>
    <div class="recipe-header-info">
        <h2 class="recipe-title">${recipe_name}</h2>
        <div class="basic-info">...</div>
    </div>
</div>
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/indian-cuisine-chatbot.git
cd indian-cuisine-chatbot
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Access the chatbot at `http://localhost:5000`

## Usage Examples

1. Search by cuisine:
```
User: "Show me Gujarati recipes"
```

2. Search by diet preference:
```
User: "I want vegetarian breakfast recipes"
```

3. Search by recipe number:
```
User: "Show me recipe number 42"
```

4. Search by ingredients:
```
User: "Recipes with paneer and spinach"
```

## File Structure
```
indian-cuisine-chatbot/
├── app.py
├── chatbot.py
├── update_image_files.py
├── check_matches.py
├── assign_images.py
├── cuisines.csv
├── requirements.txt
├── static/
│   ├── css/
│   └── images/
├── templates/
│   └── index.html
└── image_for_cuisines/
    └── data/
```

## Dependencies
- Flask
- Pandas
- NumPy
- NLTK
- Python-Levenshtein
- Scikit-learn

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Recipe data sourced from various Indian cuisine collections (via Kaggle)
- Image dataset curated from authentic Indian recipe sources
- Special thanks to contributors, maintainers, and recipe authors
