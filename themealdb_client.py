import requests

class TheMealDBClient:
    BASE_URL = "https://www.themealdb.com/api/json/v1/1/"

    def search_by_name(self, name):
        url = f"{self.BASE_URL}search.php?s={name}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('meals', [])
        return []

    def search_by_ingredient(self, ingredient):
        url = f"{self.BASE_URL}filter.php?i={ingredient}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('meals', [])
        return []

    def lookup_by_id(self, meal_id):
        url = f"{self.BASE_URL}lookup.php?i={meal_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('meals', [])
        return []

    def list_categories(self):
        url = f"{self.BASE_URL}list.php?c=list"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('meals', [])
        return []
