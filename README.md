# 🍳 Indian Cuisine & World Recipe Chatbot

A modern, AI-powered chatbot that helps users discover and explore Indian and world recipes. Features intelligent recipe search (NLU, semantic search), LLM-powered explanations, beautiful glassmorphism UI, and rich interactive features.

![Recipe Chatbot](https://img.shields.io/badge/Recipe-Chatbot-FF6B6B?style=for-the-badge&logo=cookiecutter)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-green?style=for-the-badge&logo=flask)

## ✨ Major Features

### 1. 🎨 Modern UI/UX
- **Glassmorphism Design**: Beautiful frosted glass effects with smooth gradients
- **Light/Dark Mode**: Theme toggle with persistent preference
- **Recipe Cards with Hover Effects**: Animated cards that reveal ingredients on hover
- **Typing Indicator**: Animated dots while the bot is "thinking"
- **Responsive Layout**: Works beautifully on desktop and mobile

### 2. 🤖 AI-Powered Features
- **LLM Integration (Google Gemini)**: Natural conversation and intelligent responses
- **Recipe Health Explanations**: "Why is this recipe healthy?" → AI explains nutritional benefits
- **Smart Ingredient Substitutions**: Get alternatives for dietary needs (vegan, gluten-free, etc.)
- **Contextual Cooking Tips**: AI-generated tips based on ingredients and cuisine

### 3. 🔍 Intelligent Search
- **TheMealDB API Integration**: Always tries to fetch recipes from API first
- **Semantic Search**: Sentence Transformers for better query matching
- **Fuzzy Matching**: Robust ingredient search with fuzzywuzzy
- **Multi-filter Support**: Search by cuisine, course, diet, time, and difficulty

### 4. 📐 Recipe Scaling
- **Adjustable Servings**: Scale recipes for any number of people
- **Natural Language**: "Make this for 8 people" → Auto-adjust quantities
- **Smart Fraction Handling**: Converts decimals to readable fractions

### 5. 📤 Social Sharing
- **WhatsApp**: Share recipes with friends
- **Twitter/X**: Tweet your favorite discoveries
- **Facebook**: Share on your feed
- **Copy Link**: Quick link copying with toast notification

### 6. 💡 Smart Follow-ups
After showing a recipe, the chatbot suggests:
- "Want substitutions?" → Get ingredient alternatives
- "See similar recipes?" → Find related dishes
- "Health benefits?" → Learn nutritional info
- "Scale recipe?" → Adjust for different servings
- "Cooking tips?" → Get pro tips for the recipe

## 🛠️ Technical Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python, Flask, Pandas, NumPy |
| **NLP/AI** | spaCy, Sentence Transformers, Google Gemini |
| **Search** | scikit-learn TF-IDF, Fuzzy Matching |
| **Database** | SQLite, Flask-SQLAlchemy |
| **Frontend** | HTML5, CSS3 (Glassmorphism), Vanilla JS |
| **APIs** | TheMealDB, Spoonacular |

## 📋 Example Queries

```
• "Show me Chinese breakfast recipes"
• "Recipes with paneer and spinach"
• "I want vegetarian dinner recipes"
• "Quick recipes under 30 minutes"
• "Why is this recipe healthy?"
• "What can I substitute for eggs?"
• "Make this for 8 people"
• "Show me similar recipes"
```

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/recipe-chatbot.git
cd recipe-chatbot
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

5. **(Optional) Set up Gemini API Key:**
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Set the environment variable:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
   - Or edit `llm_service.py` directly

6. **Run the application:**
```bash
python app.py
```

7. **Access the chatbot at** `http://localhost:5000`

## 📁 Project Structure

```
recipe-chatbot/
├── app.py                    # Flask application & routes
├── chatbot.py                # Main chatbot logic
├── llm_service.py            # LLM integration & recipe scaling
├── themealdb_client.py       # TheMealDB API client
├── spoonacular_client.py     # Spoonacular API client
├── cuisine_data_utils.py     # Cuisine data utilities
├── recipe_utils.py           # Recipe helper utilities
├── cuisines.csv              # Local recipe database (~11MB)
├── requirements.txt          # Python dependencies
├── static/
│   ├── css/
│   │   └── modern-style.css  # Glassmorphism design system
│   ├── js/
│   │   └── chat.js           # Interactive JavaScript
│   └── images/               # Recipe images
├── templates/
│   ├── index.html            # Main chat interface
│   ├── home.html             # Homepage
│   ├── login.html            # Login page
│   └── register.html         # Registration page
├── data/
│   ├── recipes.json          # Supplementary recipe data
│   └── responses.json        # Pre-defined responses
└── instance/
    └── recipes.db            # SQLite database
```

## 🎨 Design System

The UI uses a custom CSS design system with:
- **CSS Variables** for easy theming
- **Glassmorphism** effects with backdrop blur
- **Gradient accents** for visual appeal
- **Smooth animations** and micro-interactions
- **Responsive breakpoints** for all devices

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for LLM features | Optional |
| `SECRET_KEY` | Flask secret key (auto-generated) | Auto |

### Customization

- **Modify themes**: Edit CSS variables in `modern-style.css`
- **Add cuisines**: Update `cuisines.csv` with new recipes
- **Customize responses**: Edit templates in `chatbot.py`

## 📊 Dependencies

```
flask==2.0.1
flask-sqlalchemy==2.5.1
flask-login==0.5.0
flask-migrate==3.1.0
pandas==1.3.3
numpy==1.21.2
scikit-learn==0.24.2
nltk==3.6.3
spacy
sentence-transformers
google-generativeai
fuzzywuzzy
python-Levenshtein
requests
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Recipe data sourced from various Indian and world cuisine collections (via Kaggle, TheMealDB)
- Image dataset curated from authentic recipe sources
- Google Gemini for LLM capabilities
- Open source community for amazing libraries

---

**Made with ❤️ for food lovers everywhere** 🍳
