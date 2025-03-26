# Recipe Chatbot

A retrieval-based chatbot that helps users discover and explore recipes from various cuisines. The chatbot uses TF-IDF vectorization and cosine similarity for recipe matching, providing relevant recipe suggestions based on user queries.

## Features

- **Smart Recipe Search**: Search recipes by:
  - Cuisine type (e.g., Indian, Italian, Chinese)
  - Ingredients (single or multiple)
  - Diet preferences (vegetarian, vegan, non-vegetarian)
  - Course type (breakfast, lunch, dinner, dessert)
  - Cooking time
  - Difficulty level

- **Natural Language Processing**:
  - TF-IDF based recipe matching
  - Fuzzy matching for ingredient names
  - Multi-ingredient search support
  - Context-aware responses

- **User-Friendly Interface**:
  - Clean and organized recipe display
  - Categorized recipe listings by cuisine and diet
  - Detailed recipe information including:
    - Basic information (cuisine, course, diet)
    - Description
    - Ingredients list
    - Step-by-step instructions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/recipe-chatbot.git
cd recipe-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
flask db init
flask db migrate
flask db upgrade
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Start chatting with the recipe bot! You can:
   - Ask for recipes by cuisine: "Show me Indian recipes"
   - Search by ingredients: "Recipes with rice and beans"
   - Filter by diet: "Show vegetarian recipes"
   - Get specific recipes: "Give me recipe number 3"

## Project Structure

```
recipe-chatbot/
├── app.py                 # Flask application
├── chatbot.py            # Chatbot logic and recipe processing
├── cuisines.csv          # Recipe dataset
├── requirements.txt      # Python dependencies
├── static/              # Static files
├── templates/           # HTML templates
└── data/               # User data storage
```

## Dependencies

- Flask
- Pandas
- NumPy
- scikit-learn
- NLTK
- fuzzywuzzy
- python-Levenshtein
- bcrypt
- python-dotenv
- gunicorn

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Recipe dataset provided by Kaggle
- Special thanks to contributors and maintainers 
