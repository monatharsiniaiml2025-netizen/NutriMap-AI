"""Health and nutrition mapping engine for NutriMap AI.

This module converts a user's health condition and goal into transparent
nutrition priorities, then scores candidate recipes against those priorities.
It is a decision-support layer for a student project, not a medical diagnosis.
"""


def _norm(value):
    return str(value or "").strip().lower()


def calculate_bmi(weight_kg, height_cm):
    try:
        weight = float(weight_kg)
        height = float(height_cm) / 100.0
        if weight <= 0 or height <= 0:
            return None
        return round(weight / (height * height), 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def bmi_category(bmi):
    if bmi is None:
        return "Not provided"
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Healthy range"
    if bmi < 30:
        return "Overweight"
    return "Obesity range"


# Direction: low/high/moderate. We deliberately use nutrients that exist in
# the curated recipe catalogue. Sodium is not stored for those recipes, so we
# never pretend to perform sodium-based hypertension filtering.
CONDITION_RULES = {
    "diabetes": {
        "label": "Diabetes-aware",
        "weights": {"carbohydrates": ("low", 0.50), "fiber": ("high", 0.35), "calories": ("low", 0.15)},
        "note": "Prioritizes lower carbohydrate and higher-fiber recipes.",
    },
    "hypertension": {
        "label": "Hypertension-aware",
        "weights": {"fat": ("low", 0.45), "fiber": ("high", 0.35), "calories": ("low", 0.20)},
        "note": "Uses the available nutrition fields as a secondary proxy; sodium is not stored in the curated recipe catalogue.",
    },
    "obesity": {
        "label": "Obesity-aware",
        "weights": {"calories": ("low", 0.55), "fiber": ("high", 0.30), "fat": ("low", 0.15)},
        "note": "Prioritizes lower-calorie, higher-fiber recipes.",
    },
    "heart disease": {
        "label": "Heart-health-aware",
        "weights": {"fat": ("low", 0.45), "fiber": ("high", 0.35), "calories": ("low", 0.20)},
        "note": "Prioritizes lower-fat and higher-fiber recipes.",
    },
    "thyroid": {
        "label": "Balanced nutrition",
        "weights": {"protein": ("high", 0.40), "fiber": ("high", 0.25), "calories": ("moderate", 0.35)},
        "note": "Uses a balanced nutrition profile rather than claiming a disease-specific diet.",
    },
}

GOAL_RULES = {
    "weight loss": {"weights": {"calories": ("low", 0.45), "fiber": ("high", 0.30), "protein": ("high", 0.25)}},
    "weight gain": {"weights": {"calories": ("high", 0.45), "protein": ("high", 0.45), "fiber": ("moderate", 0.10)}},
    "maintain weight": {"weights": {"calories": ("moderate", 0.40), "protein": ("high", 0.30), "fiber": ("high", 0.30)}},
}


def _range_score(value, minimum, maximum, direction):
    if maximum <= minimum:
        return 0.5
    x = (float(value) - minimum) / (maximum - minimum)
    x = max(0.0, min(1.0, x))
    if direction == "low":
        return 1.0 - x
    if direction == "high":
        return x
    return 1.0 - abs(x - 0.5) * 2.0


def _score_rules(recipe, candidates, rules):
    if not rules:
        return 0.5
    scores = []
    for nutrient, (direction, weight) in rules.items():
        values = [float(r.get(nutrient, 0) or 0) for r in candidates]
        value = float(recipe.get(nutrient, 0) or 0)
        scores.append(_range_score(value, min(values), max(values), direction) * weight)
    total_weight = sum(weight for _, weight in rules.values())
    return round(sum(scores) / total_weight, 4) if total_weight else 0.5


def nutrition_fit(recipe, candidates, condition, goal):
    condition_key = _norm(condition)
    goal_key = _norm(goal)
    condition_rule = CONDITION_RULES.get(condition_key)
    goal_rule = GOAL_RULES.get(goal_key, GOAL_RULES["maintain weight"])

    goal_score = _score_rules(recipe, candidates, goal_rule["weights"])
    if condition_rule:
        condition_score = _score_rules(recipe, candidates, condition_rule["weights"])
        score = 0.60 * condition_score + 0.40 * goal_score
    else:
        condition_score = None
        score = goal_score
    return round(score, 4), round(condition_score, 4) if condition_score is not None else None, round(goal_score, 4)


def mapping_reasons(recipe, candidates, condition, goal, bmi=None):
    reasons = []
    condition_key = _norm(condition)
    goal_key = _norm(goal)
    rule = CONDITION_RULES.get(condition_key)

    if rule:
        if "carbohydrates" in rule["weights"] and rule["weights"]["carbohydrates"][0] == "low":
            reasons.append("lower carbohydrate profile")
        if "fiber" in rule["weights"] and rule["weights"]["fiber"][0] == "high":
            reasons.append("higher fiber profile")
        if "fat" in rule["weights"] and rule["weights"]["fat"][0] == "low":
            reasons.append("lower fat profile")
        if "calories" in rule["weights"] and rule["weights"]["calories"][0] == "low":
            reasons.append("lower calorie profile")

    if goal_key == "weight loss":
        reasons.append("supports the weight-loss goal")
    elif goal_key == "weight gain":
        reasons.append("supports the weight-gain goal")
    else:
        reasons.append("supports a balanced-maintenance goal")

    if bmi is not None:
        reasons.append(f"BMI context: {bmi_category(bmi)}")
    return reasons[:4]


def health_label(condition, goal):
    rule = CONDITION_RULES.get(_norm(condition))
    if rule:
        return rule["label"]
    return "Goal-aware"


def health_note(condition):
    rule = CONDITION_RULES.get(_norm(condition))
    return rule["note"] if rule else "Recommendations are ranked using the selected health goal and available nutrition values."
