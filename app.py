from flask import Flask, render_template, request, session
import pickle
import re
import os

from nutrition_engine import (
    calculate_bmi,
    bmi_category,
    nutrition_fit,
    mapping_reasons,
    health_label,
    health_note,
)
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "nutrimap-ai-local-secret")


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

@app.route("/recommend", methods=["POST"])
def recommend():
    """Generate ingredient-, diet-, allergy-, goal- and health-aware recipes."""

    ingredients = request.form.get("ingredients", "").strip()
    allergy = request.form.get("allergy", "").strip()
    diet = request.form.get("diet", request.form.get("diet_preference", "Non Vegetarian"))
    goal = request.form.get("goal", request.form.get("health_goal", "Maintain Weight"))
    condition = request.form.get("condition", request.form.get("health_condition", "None"))
    name = request.form.get("name", "").strip()
    age = request.form.get("age", "").strip()
    gender = request.form.get("gender", "").strip()
    weight = request.form.get("weight", "").strip()
    height = request.form.get("height", "").strip()

    if not ingredients:
        return render_template("result.html", recipes=[], message="Please enter your available ingredients.")

    user_ingredients = [
        normalize_ingredient(item) for item in ingredients.split(",") if item.strip()
    ]

    # BMI is used only as additional context; it does not diagnose disease.
    bmi = calculate_bmi(weight, height)
    profile = {
        "name": name or "User",
        "age": age or "Not provided",
        "gender": gender or "Not provided",
        "weight": weight or "Not provided",
        "height": height or "Not provided",
        "bmi": bmi,
        "bmi_category": bmi_category(bmi),
        "diet": diet,
        "condition": condition,
        "goal": goal,
        "allergy": allergy or "None",
        "ingredients": user_ingredients,
        "health_label": health_label(condition, goal),
        "health_note": health_note(condition),
    }

    candidates = []
    for recipe in FAMILIAR_RECIPES:
        if not recipe_uses_only_available_ingredients(recipe["ingredients"], user_ingredients):
            continue
        if not diet_allowed(recipe, diet):
            continue
        if not allergy_allowed(recipe, allergy):
            continue
        ingredient_score = ingredient_match_score(recipe["ingredients"], user_ingredients)
        candidates.append((recipe, ingredient_score))

    if not candidates:
        return render_template(
            "result.html",
            recipes=[],
            profile=profile,
            message=("No familiar recipe can be prepared using the available ingredients. "
                     "Please add one or two more ingredients."),
        )

    user_vector = vectorizer.transform([clean_text(ingredients)])
    recipe_texts = [" ".join(recipe["ingredients"]) for recipe, _ in candidates]
    recipe_vectors = vectorizer.transform(recipe_texts)
    similarity_scores = cosine_similarity(user_vector, recipe_vectors).flatten()

    candidate_recipes = [recipe for recipe, _ in candidates]
    ranked = []
    for i, (recipe, ingredient_score) in enumerate(candidates):
        ml_score = float(similarity_scores[i])
        nutrition_score, condition_score, goal_score = nutrition_fit(
            recipe, candidate_recipes, condition, goal
        )
        # NutriMap score: ingredient availability + semantic similarity + nutrition fit.
        final_score = (
            0.45 * ingredient_score +
            0.20 * ml_score +
            0.35 * nutrition_score
        )
        ranked.append((recipe, final_score, ingredient_score, ml_score, nutrition_score, condition_score, goal_score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    selected = ranked[:5]

    recommendations = []
    for recipe, final_score, ingredient_score, ml_score, nutrition_score, condition_score, goal_score in selected:
        recommendations.append({
            "name": recipe["name"],
            "calories": recipe["calories"],
            "protein": recipe["protein"],
            "fat": recipe["fat"],
            "carbohydrates": recipe["carbohydrates"],
            "fiber": recipe["fiber"],
            "ingredients": recipe["ingredients"],
            "image": "",
            "match_score": round(final_score * 100, 1),
            "ingredient_score": round(ingredient_score * 100, 1),
            "similarity_score": round(ml_score * 100, 1),
            "nutrition_score": round(nutrition_score * 100, 1),
            "condition_score": round(condition_score * 100, 1) if condition_score is not None else None,
            "goal_score": round(goal_score * 100, 1),
            "reasons": mapping_reasons(recipe, candidate_recipes, condition, goal, bmi),
        })

    # Keep the latest profile in a signed cookie-backed session for the dynamic timetable.
    session["nutrition_profile"] = profile
    session["recommendations"] = recommendations

    return render_template(
        "result.html",
        recipes=recommendations,
        profile=profile,
        message="",
    )


# =========================================================
# 7-DAY MEAL PLAN
# ADDITIONAL FEATURE
# =========================================================

@app.route("/timetable")
def timetable():
    recommendations = session.get("recommendations", [])
    profile = session.get("nutrition_profile", {})

    # Build a simple, transparent weekly plan from the recipes that were just ranked.
    if recommendations:
        names = [r["name"] for r in recommendations]
    else:
        names = [r["name"] for r in FAMILIAR_RECIPES[:5]]

    breakfast_names = [n for n in names if any(k in n.lower() for k in ["idli", "dosa", "upma", "pongal", "chapati", "rice"])] or names
    lunch_names = names
    dinner_names = list(reversed(names))
    snack_names = [n for n in names if any(k in n.lower() for k in ["chickpea", "sundal", "soup"])] or names

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly_plan = []
    for i, day in enumerate(days):
        weekly_plan.append({
            "day": day,
            "meals": [
                {"name": "🌅 Breakfast", "food": breakfast_names[i % len(breakfast_names)], "time": "08:00 AM"},
                {"name": "☀️ Lunch", "food": lunch_names[i % len(lunch_names)], "time": "01:00 PM"},
                {"name": "🍎 Evening Snack", "food": snack_names[i % len(snack_names)], "time": "04:30 PM"},
                {"name": "🌙 Dinner", "food": dinner_names[i % len(dinner_names)], "time": "08:00 PM"},
            ],
        })

    return render_template("timetable.html", weekly_plan=weekly_plan, profile=profile)


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )