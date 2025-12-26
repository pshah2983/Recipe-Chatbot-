import requests
from fuzzywuzzy import fuzz

SPOONACULAR_API_KEY = "1adc3d4a8f7f4e2aade5e7e154965780"
BASE_URL = "https://api.spoonacular.com/recipes/complexSearch"
DETAIL_URL = "https://api.spoonacular.com/recipes/{id}/information"

def search_recipe_by_name(recipe_name, min_similarity=80):
    params = {
        "query": recipe_name,
        "number": 1,
        "apiKey": SPOONACULAR_API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        return None
    data = response.json()
    if not data.get("results"):
        return None
    recipe = data["results"][0]
    # Fuzzy match check
    if fuzz.token_set_ratio(recipe_name.lower(), recipe["title"].lower()) < min_similarity:
        return None
    # Fetch detailed info
    detail_response = requests.get(
        DETAIL_URL.format(id=recipe["id"]),
        params={"apiKey": SPOONACULAR_API_KEY}
    )
    if detail_response.status_code != 200:
        return None
    detail = detail_response.json()
    return {
        "name": detail.get("title"),
        "image": detail.get("image"),
        "ingredients": [i["original"] for i in detail.get("extendedIngredients", [])],
        "instructions": detail.get("instructions"),
        "source": "Spoonacular"
    }
