import pandas as pd
import ast
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/recipes.csv")

print("Dataset loaded!")
print("Total recipes:", len(df))


# ==========================================
# STEP 2: REMOVE UNNECESSARY COLUMN
# ==========================================

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


# ==========================================
# STEP 3: HANDLE MISSING VALUES
# ==========================================

df["recipe_name"] = df["recipe_name"].fillna("")
df["ingredients_list"] = df["ingredients_list"].fillna("")


# ==========================================
# STEP 4: CONVERT INGREDIENTS TO TEXT
# ==========================================

def convert_ingredients(value):

    try:
        ingredients = ast.literal_eval(value)

        if isinstance(ingredients, list):
            return " ".join(str(item) for item in ingredients)

        return str(value)

    except:
        return str(value)


df["ingredients_text"] = df["ingredients_list"].apply(
    convert_ingredients
)


# ==========================================
# STEP 5: TF-IDF VECTORIZATION
# ==========================================

print("\nCreating TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix = vectorizer.fit_transform(
    df["ingredients_text"]
)

print("TF-IDF matrix created!")
print("Matrix shape:", tfidf_matrix.shape)


# ==========================================
# STEP 6: CREATE MODEL FOLDER
# ==========================================

os.makedirs("model", exist_ok=True)


# ==========================================
# STEP 7: SAVE MODEL
# ==========================================

model_data = {
    "data": df,
    "vectorizer": vectorizer,
    "tfidf_matrix": tfidf_matrix
}

with open("model/nutrimap_model.pkl", "wb") as file:

    pickle.dump(model_data, file)


print("\n================================")
print("MODEL TRAINING COMPLETED!")
print("================================")

print("Model saved successfully!")
print("Location: model/nutrimap_model.pkl")

