from flask import Flask, render_template, request
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


# =========================================================
# LOAD ML MODEL
# =========================================================

with open("model/nutrimap_model.pkl", "rb") as file:
    model_data = pickle.load(file)

vectorizer = model_data["vectorizer"]


# =========================================================
# FAMILIAR INDIAN / COMMON RECIPES
# =========================================================

FAMILIAR_RECIPES = [

    # -----------------------------------------------------
    # RICE RECIPES
    # -----------------------------------------------------

    {
        "name": "Tomato Rice",
        "ingredients": [
            "rice",
            "tomato",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 210,
        "protein": 5,
        "fat": 6,
        "carbohydrates": 35,
        "fiber": 3
    },

    {
        "name": "Carrot Beans Rice",
        "ingredients": [
            "rice",
            "carrot",
            "beans",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 215,
        "protein": 5,
        "fat": 6,
        "carbohydrates": 35,
        "fiber": 4
    },

    {
        "name": "Tomato Carrot Rice",
        "ingredients": [
            "rice",
            "tomato",
            "carrot",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 215,
        "protein": 5,
        "fat": 6,
        "carbohydrates": 35,
        "fiber": 4
    },

    {
        "name": "Tomato Beans Rice",
        "ingredients": [
            "rice",
            "tomato",
            "beans",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 215,
        "protein": 5,
        "fat": 6,
        "carbohydrates": 35,
        "fiber": 4
    },

    {
        "name": "Vegetable Rice",
        "ingredients": [
            "rice",
            "carrot",
            "beans",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 220,
        "protein": 6,
        "fat": 6,
        "carbohydrates": 36,
        "fiber": 4
    },

    {
        "name": "Simple Onion Rice",
        "ingredients": [
            "rice",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 190,
        "protein": 4,
        "fat": 5,
        "carbohydrates": 34,
        "fiber": 2
    },

    {
        "name": "Lemon Rice",
        "ingredients": [
            "rice",
            "lemon",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 200,
        "protein": 5,
        "fat": 7,
        "carbohydrates": 32,
        "fiber": 2
    },

    {
        "name": "Curd Rice",
        "ingredients": [
            "rice",
            "curd",
            "salt"
        ],
        "calories": 190,
        "protein": 6,
        "fat": 5,
        "carbohydrates": 30,
        "fiber": 1
    },

    {
        "name": "Rasam Rice",
        "ingredients": [
            "rice",
            "tomato",
            "pepper",
            "cumin",
            "salt"
        ],
        "calories": 190,
        "protein": 5,
        "fat": 3,
        "carbohydrates": 34,
        "fiber": 3
    },

    # -----------------------------------------------------
    # CHICKEN RECIPES
    # -----------------------------------------------------

    {
        "name": "Simple Chicken Rice",
        "ingredients": [
            "rice",
            "chicken",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 320,
        "protein": 24,
        "fat": 9,
        "carbohydrates": 35,
        "fiber": 2
    },

    {
        "name": "Tomato Chicken Rice",
        "ingredients": [
            "rice",
            "chicken",
            "tomato",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 330,
        "protein": 25,
        "fat": 9,
        "carbohydrates": 36,
        "fiber": 3
    },

    {
        "name": "Chicken Carrot Rice",
        "ingredients": [
            "rice",
            "chicken",
            "carrot",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 325,
        "protein": 25,
        "fat": 8,
        "carbohydrates": 36,
        "fiber": 3
    },

    {
        "name": "Chicken Beans Rice",
        "ingredients": [
            "rice",
            "chicken",
            "beans",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 325,
        "protein": 25,
        "fat": 8,
        "carbohydrates": 35,
        "fiber": 4
    },

    {
        "name": "Chicken Vegetable Rice",
        "ingredients": [
            "rice",
            "chicken",
            "carrot",
            "beans",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 335,
        "protein": 26,
        "fat": 9,
        "carbohydrates": 37,
        "fiber": 4
    },

    {
        "name": "Tomato Chicken",
        "ingredients": [
            "chicken",
            "tomato",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 250,
        "protein": 27,
        "fat": 9,
        "carbohydrates": 10,
        "fiber": 2
    },

    {
        "name": "Chicken Carrot Curry",
        "ingredients": [
            "chicken",
            "carrot",
            "onion",
            "tomato",
            "oil",
            "salt"
        ],
        "calories": 260,
        "protein": 27,
        "fat": 9,
        "carbohydrates": 12,
        "fiber": 3
    },

    {
        "name": "Chicken Beans Curry",
        "ingredients": [
            "chicken",
            "beans",
            "onion",
            "tomato",
            "oil",
            "salt"
        ],
        "calories": 255,
        "protein": 27,
        "fat": 8,
        "carbohydrates": 11,
        "fiber": 3
    },

    # -----------------------------------------------------
    # VEGETABLE RECIPES
    # -----------------------------------------------------

    {
        "name": "Carrot Beans Curry",
        "ingredients": [
            "carrot",
            "beans",
            "onion",
            "tomato",
            "oil",
            "salt"
        ],
        "calories": 150,
        "protein": 4,
        "fat": 5,
        "carbohydrates": 22,
        "fiber": 5
    },

    {
        "name": "Tomato Carrot Curry",
        "ingredients": [
            "tomato",
            "carrot",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 140,
        "protein": 3,
        "fat": 5,
        "carbohydrates": 21,
        "fiber": 5
    },

    {
        "name": "Beans Tomato Curry",
        "ingredients": [
            "beans",
            "tomato",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 135,
        "protein": 4,
        "fat": 5,
        "carbohydrates": 20,
        "fiber": 5
    },

    {
        "name": "Mixed Vegetable Curry",
        "ingredients": [
            "carrot",
            "beans",
            "tomato",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 155,
        "protein": 4,
        "fat": 5,
        "carbohydrates": 23,
        "fiber": 6
    },

    {
        "name": "Vegetable Soup",
        "ingredients": [
            "carrot",
            "beans",
            "tomato",
            "onion",
            "pepper",
            "salt",
            "water"
        ],
        "calories": 100,
        "protein": 3,
        "fat": 2,
        "carbohydrates": 15,
        "fiber": 4
    },

    {
        "name": "Tomato Soup",
        "ingredients": [
            "tomato",
            "onion",
            "pepper",
            "salt",
            "water"
        ],
        "calories": 90,
        "protein": 2,
        "fat": 2,
        "carbohydrates": 14,
        "fiber": 3
    },

    # -----------------------------------------------------
    # DAL RECIPES
    # -----------------------------------------------------

    {
        "name": "Dal Rice",
        "ingredients": [
            "rice",
            "dal",
            "tomato",
            "onion",
            "salt"
        ],
        "calories": 225,
        "protein": 9,
        "fat": 5,
        "carbohydrates": 37,
        "fiber": 5
    },

    {
        "name": "Dal Curry",
        "ingredients": [
            "dal",
            "onion",
            "tomato",
            "salt"
        ],
        "calories": 180,
        "protein": 10,
        "fat": 4,
        "carbohydrates": 25,
        "fiber": 7
    },

    {
        "name": "Tomato Dal",
        "ingredients": [
            "dal",
            "tomato",
            "onion",
            "salt"
        ],
        "calories": 185,
        "protein": 10,
        "fat": 4,
        "carbohydrates": 26,
        "fiber": 7
    },

    # -----------------------------------------------------
    # BREAKFAST RECIPES
    # -----------------------------------------------------

    {
        "name": "Idli",
        "ingredients": [
            "rice",
            "urad dal",
            "salt"
        ],
        "calories": 150,
        "protein": 5,
        "fat": 1,
        "carbohydrates": 30,
        "fiber": 2
    },

    {
        "name": "Dosa",
        "ingredients": [
            "rice",
            "urad dal",
            "oil",
            "salt"
        ],
        "calories": 170,
        "protein": 5,
        "fat": 4,
        "carbohydrates": 28,
        "fiber": 2
    },

    {
        "name": "Onion Dosa",
        "ingredients": [
            "rice",
            "urad dal",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 180,
        "protein": 5,
        "fat": 4,
        "carbohydrates": 29,
        "fiber": 2
    },

    {
        "name": "Vegetable Dosa",
        "ingredients": [
            "rice",
            "urad dal",
            "onion",
            "carrot",
            "beans",
            "oil",
            "salt"
        ],
        "calories": 195,
        "protein": 6,
        "fat": 5,
        "carbohydrates": 30,
        "fiber": 3
    },

    {
        "name": "Vegetable Upma",
        "ingredients": [
            "rava",
            "onion",
            "carrot",
            "beans",
            "oil",
            "salt"
        ],
        "calories": 180,
        "protein": 5,
        "fat": 5,
        "carbohydrates": 29,
        "fiber": 3
    },

    {
        "name": "Onion Upma",
        "ingredients": [
            "rava",
            "onion",
            "oil",
            "salt"
        ],
        "calories": 170,
        "protein": 4,
        "fat": 4,
        "carbohydrates": 28,
        "fiber": 2
    },

    {
        "name": "Pongal",
        "ingredients": [
            "rice",
            "moong dal",
            "pepper",
            "cumin",
            "salt"
        ],
        "calories": 210,
        "protein": 7,
        "fat": 5,
        "carbohydrates": 31,
        "fiber": 3
    },

    {
        "name": "Chapati",
        "ingredients": [
            "atta",
            "water",
            "salt"
        ],
        "calories": 120,
        "protein": 4,
        "fat": 2,
        "carbohydrates": 22,
        "fiber": 3
    },

    # -----------------------------------------------------
    # SNACKS
    # -----------------------------------------------------

    {
        "name": "Chana Sundal",
        "ingredients": [
            "chickpea",
            "salt"
        ],
        "calories": 160,
        "protein": 8,
        "fat": 3,
        "carbohydrates": 22,
        "fiber": 6
    },

    {
        "name": "Boiled Chickpeas",
        "ingredients": [
            "chickpea",
            "salt"
        ],
        "calories": 150,
        "protein": 8,
        "fat": 3,
        "carbohydrates": 23,
        "fiber": 6
    }
]


# =========================================================
# BASIC PANTRY INGREDIENTS
# =========================================================

BASIC_PANTRY = {
    "salt",
    "water",
    "oil",
    "cooking oil",
    "vegetable oil"
}


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# INGREDIENT NORMALIZATION
# =========================================================

def normalize_ingredient(text):

    text = clean_text(text)

    replacements = {

        # Vegetables
        "tomatoes": "tomato",
        "onions": "onion",
        "carrots": "carrot",
        "potatoes": "potato",
        "beans": "beans",
        "green beans": "beans",
        "peas": "peas",

        # Rice
        "white rice": "rice",
        "brown rice": "rice",
        "long grain rice": "rice",
        "cooked rice": "rice",
        "uncooked rice": "rice",

        # Chicken
        "chicken breast": "chicken",
        "chicken pieces": "chicken",

        # Dal
        "toor dal": "toor dal",
        "pigeon pea": "toor dal",

        # Chickpea
        "chickpeas": "chickpea",
        "chana": "chickpea",

        # Dairy
        "yogurt": "curd",

        # Allergy synonyms
        "groundnut": "peanut",
        "groundnuts": "peanut",
        "peanuts": "peanut"
    }

    if text in replacements:
        text = replacements[text]

    return text


# =========================================================
# STRICT INGREDIENT CHECK
# =========================================================

def recipe_uses_only_available_ingredients(
    recipe_ingredients,
    user_ingredients
):

    available = {
        normalize_ingredient(item)
        for item in user_ingredients
    }

    # Pantry ingredients are automatically available
    available.update(
        normalize_ingredient(item)
        for item in BASIC_PANTRY
    )

    for recipe_item in recipe_ingredients:

        recipe_item = normalize_ingredient(
            recipe_item
        )

        # Recipe contains an ingredient
        # that the user does not have
        if recipe_item not in available:

            return False

    return True


# =========================================================
# INGREDIENT MATCH SCORE
# =========================================================

def ingredient_match_score(
    recipe_ingredients,
    user_ingredients
):

    available = {
        normalize_ingredient(item)
        for item in user_ingredients
    }

    available.update(
        normalize_ingredient(item)
        for item in BASIC_PANTRY
    )

    matched = 0

    for recipe_item in recipe_ingredients:

        recipe_item = normalize_ingredient(
            recipe_item
        )

        if recipe_item in available:

            matched += 1

    if len(recipe_ingredients) == 0:

        return 0

    return matched / len(recipe_ingredients)


# =========================================================
# DIET FILTER
# =========================================================

NON_VEGETARIAN = {
    "chicken",
    "mutton",
    "beef",
    "pork",
    "fish",
    "prawn",
    "shrimp",
    "crab",
    "meat",
    "lamb",
    "egg"
}


VEGAN_EXCLUDED = NON_VEGETARIAN | {
    "milk",
    "curd",
    "cheese",
    "butter",
    "ghee",
    "cream",
    "honey"
}


def diet_allowed(recipe, diet):

    recipe_ingredients = {
        normalize_ingredient(item)
        for item in recipe["ingredients"]
    }

    if diet == "Vegetarian":

        if recipe_ingredients & {
            normalize_ingredient(item)
            for item in NON_VEGETARIAN
        }:

            return False

    elif diet == "Vegan":

        if recipe_ingredients & {
            normalize_ingredient(item)
            for item in VEGAN_EXCLUDED
        }:

            return False

    return True


# =========================================================
# ALLERGY FILTER
# =========================================================

def allergy_allowed(recipe, allergy):

    if not allergy:

        return True

    allergy = clean_text(allergy)

    if allergy in [
        "none",
        "no",
        "no allergy"
    ]:

        return True

    allergy_words = {

        normalize_ingredient(item)

        for item in allergy.split(",")

        if item.strip()
    }

    recipe_ingredients = {

        normalize_ingredient(item)

        for item in recipe["ingredients"]
    }

    if allergy_words & recipe_ingredients:

        return False

    return True


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# RECOMMENDATION
# =========================================================

@app.route(
    "/recommend",
    methods=["POST"]
)
def recommend():

    # -----------------------------------------------------
    # USER INPUT
    # -----------------------------------------------------

    ingredients = request.form.get(
        "ingredients",
        ""
    ).strip()

    allergy = request.form.get(
        "allergy",
        ""
    ).strip()

    diet = request.form.get(
        "diet",
        request.form.get(
            "diet_preference",
            "Non Vegetarian"
        )
    )

    goal = request.form.get(
        "goal",
        request.form.get(
            "health_goal",
            "Maintain Weight"
        )
    )

    condition = request.form.get(
        "condition",
        request.form.get(
            "health_condition",
            "None"
        )
    )


    # -----------------------------------------------------
    # EMPTY INGREDIENT CHECK
    # -----------------------------------------------------

    if not ingredients:

        return render_template(
            "result.html",
            recipes=[],
            message=(
                "Please enter your available "
                "ingredients."
            )
        )


    # -----------------------------------------------------
    # SPLIT USER INGREDIENTS
    # -----------------------------------------------------

    user_ingredients = [

        normalize_ingredient(item)

        for item in ingredients.split(",")

        if item.strip()
    ]


    # -----------------------------------------------------
    # FIND VALID RECIPES
    # -----------------------------------------------------

    candidates = []


    for recipe in FAMILIAR_RECIPES:

        # Strict ingredient check
        if not recipe_uses_only_available_ingredients(
            recipe["ingredients"],
            user_ingredients
        ):

            continue


        # Diet check
        if not diet_allowed(
            recipe,
            diet
        ):

            continue


        # Allergy check
        if not allergy_allowed(
            recipe,
            allergy
        ):

            continue


        # Ingredient score
        score = ingredient_match_score(
            recipe["ingredients"],
            user_ingredients
        )


        candidates.append(
            (
                recipe,
                score
            )
        )


    # -----------------------------------------------------
    # NO RECIPES FOUND
    # -----------------------------------------------------

    if not candidates:

        return render_template(
            "result.html",
            recipes=[],
            message=(
                "No familiar recipe can be prepared "
                "using the available ingredients. "
                "Please add more ingredients."
            )
        )


    # =====================================================
    # TF-IDF + COSINE SIMILARITY
    # =====================================================

    user_vector = vectorizer.transform(
        [ingredients]
    )


    recipe_texts = [

        " ".join(
            recipe["ingredients"]
        )

        for recipe, score in candidates
    ]


    recipe_vectors = vectorizer.transform(
        recipe_texts
    )


    similarity_scores = cosine_similarity(
        user_vector,
        recipe_vectors
    ).flatten()


    # =====================================================
    # FINAL SCORE
    # =====================================================

    ranked = []


    for i, item in enumerate(candidates):

        recipe = item[0]

        ingredient_score = item[1]

        ml_score = similarity_scores[i]


        # Ingredient availability is more important
        final_score = (
            0.75 * ingredient_score
            +
            0.25 * ml_score
        )


        ranked.append(
            (
                recipe,
                final_score
            )
        )


    # =====================================================
    # HEALTH GOAL RANKING
    # =====================================================

    if goal == "Weight Loss":

        ranked.sort(
            key=lambda x: (
                x[1],
                -x[0]["calories"]
            ),
            reverse=True
        )

    elif goal == "Weight Gain":

        ranked.sort(
            key=lambda x: (
                x[1],
                x[0]["calories"]
            ),
            reverse=True
        )

    else:

        ranked.sort(
            key=lambda x: x[1],
            reverse=True
        )


    # =====================================================
    # TOP 5 RECIPES
    # =====================================================

    selected = ranked[:5]


    # =====================================================
    # PREPARE RESULT
    # =====================================================

    recommendations = []


    for recipe, score in selected:

        recommendations.append({

            "name": recipe["name"],

            "calories":
                recipe["calories"],

            "protein":
                recipe["protein"],

            "fat":
                recipe["fat"],

            "carbohydrates":
                recipe["carbohydrates"],

            "fiber":
                recipe["fiber"],

            "ingredients":
                recipe["ingredients"],

            "image": ""
        })


    # =====================================================
    # RESULT PAGE
    # =====================================================

    return render_template(
        "result.html",
        recipes=recommendations,
        message=""
    )


# =========================================================
# 7-DAY MEAL PLAN
# ADDITIONAL FEATURE
# =========================================================

@app.route("/timetable")
def timetable():

    return render_template(
        "timetable.html"
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )