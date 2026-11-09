import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix


# ============================================================
# JOB FRAUD DETECTION - DEBUG PREDICTION
# ============================================================

print("=" * 70)
print("JOB FRAUD DETECTION - DEBUG PREDICTION")
print("=" * 70)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

FEATURE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "features"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "models"
)

FEATURE_DATA_FILE = os.path.join(
    FEATURE_DIR,
    "feature_data.pkl"
)

FINAL_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "final_xgboost_model.pkl"
)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nFeature directory:")
print(FEATURE_DIR)

print("\nModel directory:")
print(MODEL_DIR)


# ============================================================
# 2. CHECK FILES
# ============================================================

print("\n" + "=" * 70)
print("[1] CHECKING REQUIRED FILES")
print("=" * 70)

required_files = [
    FEATURE_DATA_FILE,
    FINAL_MODEL_FILE
]

for file_path in required_files:

    if os.path.exists(file_path):
        print("PASS:", file_path)
    else:
        print("FAIL:", file_path)
        raise FileNotFoundError(file_path)


# ============================================================
# 3. LOAD FEATURE DATA
# ============================================================

print("\n" + "=" * 70)
print("[2] LOADING FEATURE DATA")
print("=" * 70)

feature_data = joblib.load(FEATURE_DATA_FILE)

print("PASS: Feature data loaded.")

print("\nAvailable keys:")

for key in feature_data.keys():
    print(" -", key)


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("[3] LOADING FINAL XGBOOST MODEL")
print("=" * 70)

model = joblib.load(FINAL_MODEL_FILE)

print("PASS: Final XGBoost model loaded.")
print("Model type:", type(model).__name__)


# ============================================================
# 5. LOAD TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("[4] LOADING TF-IDF VECTORIZER")
print("=" * 70)

tfidf = feature_data["tfidf"]

print("PASS: TF-IDF loaded.")

print("TF-IDF vocabulary size:")
print(len(tfidf.vocabulary_))


# ============================================================
# 6. LOAD FEATURE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("[5] LOADING FEATURE INFORMATION")
print("=" * 70)

tfidf_feature_names = feature_data["tfidf_feature_names"]

red_flag_columns = feature_data["red_flag_columns"]

combined_feature_names = feature_data["combined_feature_names"]

print("TF-IDF features       :", len(tfidf_feature_names))
print("Red-flag features     :", len(red_flag_columns))
print("Combined features     :", len(combined_feature_names))

print("\nRed-flag feature names:")

for i, name in enumerate(red_flag_columns):
    print(f"{i + 1:02d}. {name}")


# ============================================================
# 7. SAMPLE JOB
# ============================================================

print("\n" + "=" * 70)
print("[6] TEST JOB POSTING")
print("=" * 70)

job = {
    "title": "Software Developer",

    "company_profile":
        "We are a technology company developing software solutions for businesses.",

    "description":
        "We are looking for a software developer to join our engineering team. "
        "The candidate will work on web applications and backend services.",

    "requirements":
        "Experience with JavaScript, React, Node.js and database technologies.",

    "benefits":
        "Competitive salary, health insurance and professional development opportunities.",

    "salary_range":
        "$60000-$80000",

    "employment_type":
        "Full-time",

    "required_experience":
        "2 years",

    "required_education":
        "Bachelor's degree",

    "industry":
        "Information Technology",

    "function":
        "Engineering",

    "department":
        "Software Development",

    "telecommuting":
        0,

    "has_company_logo":
        1,

    "has_questions":
        1
}


# ============================================================
# 8. BUILD COMBINED TEXT
# ============================================================

print("\n" + "=" * 70)
print("[7] BUILDING COMBINED TEXT")
print("=" * 70)


text_fields = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "department"
]


text_parts = []

for field in text_fields:

    value = job.get(field, "")

    if value is None:
        value = ""

    text_parts.append(str(value))


combined_text = " ".join(text_parts)


print("Combined text:")
print("-" * 70)
print(combined_text)
print("-" * 70)

print("\nCharacter count:", len(combined_text))
print("Word count:", len(combined_text.split()))


# ============================================================
# 9. TF-IDF TRANSFORMATION
# ============================================================

print("\n" + "=" * 70)
print("[8] TF-IDF ANALYSIS")
print("=" * 70)

tfidf_features = tfidf.transform([combined_text])

print("TF-IDF shape:")
print(tfidf_features.shape)

print("Non-zero TF-IDF values:")
print(tfidf_features.nnz)


# ============================================================
# 10. SHOW TF-IDF WORDS
# ============================================================

print("\nTop TF-IDF words in this job:")

tfidf_array = tfidf_features.toarray()[0]

nonzero_indices = np.where(tfidf_array > 0)[0]

word_scores = []

for index in nonzero_indices:

    word_scores.append(
        (
            tfidf_feature_names[index],
            float(tfidf_array[index])
        )
    )


word_scores.sort(
    key=lambda x: x[1],
    reverse=True
)


for word, score in word_scores[:30]:

    print(
        f"{word:<30} {score:.6f}"
    )


# ============================================================
# 11. RED-FLAG FEATURES
# ============================================================

print("\n" + "=" * 70)
print("[9] RED-FLAG FEATURE ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# IMPORTANT:
# The project must use the SAME red-flag generation logic
# used during training.
#
# If red-flag values were saved in the original preprocessing
# pipeline, this section checks the feature-data structure.
# ------------------------------------------------------------


red_flags = []

for column in red_flag_columns:

    value = 0

    lower_text = combined_text.lower()

    # Generic keyword checks for debugging only.
    # These do NOT replace the original training pipeline.

    if column.lower() in lower_text:
        value = 1

    red_flags.append(value)


red_flag_array = np.array(
    red_flags,
    dtype=np.float32
).reshape(1, -1)


print("Red-flag shape:")
print(red_flag_array.shape)

print("\nDetected debug red flags:")

detected_count = 0

for i, value in enumerate(red_flag_array[0]):

    if value > 0:

        print(
            f"YES - {red_flag_columns[i]}"
        )

        detected_count += 1


if detected_count == 0:

    print("No red flags detected.")

else:

    print(
        "\nTotal detected:",
        detected_count
    )


# ============================================================
# 12. COMBINE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("[10] COMBINING FEATURES")
print("=" * 70)


red_flag_sparse = csr_matrix(
    red_flag_array
)


combined_features = hstack(
    [
        tfidf_features,
        red_flag_sparse
    ],
    format="csr"
)


print("TF-IDF shape:")
print(tfidf_features.shape)

print("Red-flag shape:")
print(red_flag_sparse.shape)

print("Combined shape:")
print(combined_features.shape)


expected_features = model.n_features_in_

print("\nModel expected features:")
print(expected_features)

print("Actual features:")
print(combined_features.shape[1])


if combined_features.shape[1] != expected_features:

    raise ValueError(
        "Feature dimension mismatch!"
    )

print("PASS: Feature dimensions match.")


# ============================================================
# 13. MODEL PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("[11] XGBOOST PREDICTION")
print("=" * 70)


prediction = model.predict(
    combined_features
)[0]


probabilities = model.predict_proba(
    combined_features
)[0]


probability_legitimate = float(
    probabilities[0]
)

probability_fraud = float(
    probabilities[1]
)


print("\nPrediction:", prediction)

print(
    "Legitimate probability:",
    f"{probability_legitimate:.6f}"
)

print(
    "Fraud probability:",
    f"{probability_fraud:.6f}"
)


if prediction == 1:

    print("\nFINAL RESULT: FRAUDULENT")

else:

    print("\nFINAL RESULT: LEGITIMATE")


# ============================================================
# 14. MODEL FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("[12] MODEL FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)


importance_values = model.feature_importances_


print(
    "Importance vector length:",
    len(importance_values)
)


if len(importance_values) != len(combined_feature_names):

    print(
        "WARNING: Feature importance length does not match "
        "combined feature names."
    )

else:

    feature_importance_pairs = []

    for i, importance in enumerate(importance_values):

        feature_importance_pairs.append(
            (
                combined_feature_names[i],
                float(importance)
            )
        )


    feature_importance_pairs.sort(
        key=lambda x: x[1],
        reverse=True
    )


    print("\nTop 30 model features:")

    for feature, importance in feature_importance_pairs[:30]:

        print(
            f"{feature:<40} {importance:.8f}"
        )


# ============================================================
# 15. IMPORTANT DIAGNOSTIC
# ============================================================

print("\n" + "=" * 70)
print("[13] DIAGNOSTIC SUMMARY")
print("=" * 70)


print(
    "Prediction:",
    "Fraudulent" if prediction == 1 else "Legitimate"
)

print(
    "Fraud probability:",
    f"{probability_fraud * 100:.2f}%"
)

print(
    "Legitimate probability:",
    f"{probability_legitimate * 100:.2f}%"
)

print(
    "TF-IDF non-zero features:",
    tfidf_features.nnz
)

print(
    "Detected red flags:",
    detected_count
)

print(
    "Total input features:",
    combined_features.shape[1]
)

print(
    "Model expected features:",
    expected_features
)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("DEBUG PREDICTION COMPLETED")
print("=" * 70)

print(
    """
IMPORTANT:

This script is being used to investigate the prediction
behavior of the selected XGBoost production model.

It does NOT retrain the model.

It does NOT change the trained model.

The purpose is to verify:

1. Combined text
2. TF-IDF transformation
3. Red-flag features
4. Feature dimensions
5. XGBoost probability
6. Important model features

The output will help determine whether the extremely high
fraud probability comes from the trained model itself or
from a difference between training preprocessing and
prediction-time preprocessing.
"""
)

print("=" * 70)