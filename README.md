# Indian Cuisine & World Recipe Chatbot

A modern, retrieval-based chatbot that helps users discover and explore Indian and world recipes. The chatbot features intelligent recipe search (NLU, semantic search), API integration, image matching, user personalization, and a beautiful, theme-aware interface.

## Major Features (v2+)

### 1. API-First Recipe Retrieval
- **TheMealDB API integration**: Always tries to fetch recipes from TheMealDB first (by name, ingredient, cuisine, etc.), with clear user messaging if the API is used.
- **CSV fallback**: If the API fails or has no results, falls back to the local recipe database (cuisines.csv), and clearly indicates the data source in the chat.
- **Supports all query types**: By name, ingredient, multi-ingredient, cuisine, course, etc.

### 2. Natural Language Understanding & Semantic Search
- **spaCy** for NLU and intent/entity extraction.
- **Sentence Transformers** for semantic search and better query matching.
- **Fuzzy and TF-IDF matching** for robust local search.

### 3. Modern User Interface
- **Unified, theme-aware design**: Consistent navigation, card style, and layout across Home, Chat, Login, and Register pages.
- **Light/Dark mode**: Theme toggle on every page, with persistent user preference.
- **Responsive layout**: Works on desktop and mobile.
- **Personalized responses**: Uses the user's name if logged in.
- **Chat history**: Saved and shown for logged-in users on the home page.

### 4. Recipe Information Display
- Recipe name, image, cuisine, course, diet, prep time
- Description, ingredients, step-by-step instructions
- Visual, interactive recipe cards and modals
- Data source (API or local) always shown in chat

### 5. Project Structure & Codebase Improvements
- Cleaned up unused files and clarified project structure
- All static and template files organized for maintainability
- Requirements updated for NLU, semantic search, and API support
- Compatibility fixes for numpy, pandas, sklearn, etc.

## Technical Stack
- **Backend**: Python, Flask, Pandas, NumPy, NLTK, spaCy, Sentence Transformers, scikit-learn, requests
- **Frontend**: HTML, CSS (theme-aware, responsive), JavaScript
- **Data**: TheMealDB API, cuisines.csv, images

## Example Usage
- "Show me Chinese breakfast recipes"
- "Recipes with paneer and spinach"
- "I want vegetarian dinner recipes"
- "Show me recipe number 42"
- "Quick Indian lunch recipes under 30 minutes"

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
4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```
5. Run the application:
```bash
python app.py
```
6. Access the chatbot at `http://localhost:5000`

## File Structure
```
indian-cuisine-chatbot/
├── app.py
├── chatbot.py
├── themealdb_client.py
├── update_image_files.py
├── check_matches.py
├── assign_images.py
├── cuisines.csv
├── requirements.txt
├── requirements-nlu.txt
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── templates/
│   ├── index.html
│   ├── home.html
│   ├── login.html
│   └── register.html
├── data/
│   ├── recipes.json
│   └── responses.json
└── image_for_cuisines/
    └── data/
```

## Dependencies
- Flask
- Pandas
- NumPy
- NLTK
- spaCy
- sentence-transformers
- scikit-learn
- requests
- Python-Levenshtein

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Recipe data sourced from various Indian and world cuisine collections (via Kaggle, TheMealDB)
- Image dataset curated from authentic recipe sources
- Special thanks to contributors, maintainers, and recipe authors
