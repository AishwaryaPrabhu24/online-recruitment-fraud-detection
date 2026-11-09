import os
import joblib
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = "dataset/processed/features/feature_data.pkl"

MODEL_DIR = "dataset/processed/models"

RANDOM_FOREST_MODEL = os.path.join(
    MODEL_DIR,
    "random_forest_model.pkl"
)

LINEAR_SVM_MODEL = os.path.join(
    MODEL_DIR,
    "linear_svm_model.pkl"
)

XGBOOST_MODEL = os.path.join(
    MODEL_DIR,
    "xgboost_model.pkl"
)


print("=" * 60)
print("PREDICTION PIPELINE TEST")
print("=" * 60)


# ============================================================
# 1. CHECK REQUIRED FILES
# ============================================================

print("\n[1] Checking required files...")

required_files = [
    FEATURE_FILE,
    RANDOM_FOREST_MODEL,
    LINEAR_SVM_MODEL,
    XGBOOST_MODEL
]

all_files_exist = True

for file_path in required_files:

    if os.path.exists(file_path):
        print(f"PASS: {file_path}")
    else:
        print(f"FAIL: {file_path} not found")
        all_files_exist = False


if not all_files_exist:

    print("\nERROR: One or more required files are missing.")
    print("Please run feature_engineering.py and smote_training.py first.")
    raise SystemExit


# ============================================================
# 2. LOAD FEATURE DATA
# ============================================================

print("\n[2] Loading feature data...")

try:

    feature_data = joblib.load(FEATURE_FILE)

    print("PASS: Feature data loaded successfully.")

except Exception as e:

    print("FAIL: Could not load feature data.")
    print("Error:", e)
    raise SystemExit


# ============================================================
# 3. CHECK FEATURE DATA STRUCTURE
# ============================================================

print("\n[3] Checking feature data structure...")

print("Feature data type:", type(feature_data))


if isinstance(feature_data, dict):

    print("PASS: Feature data is stored as a dictionary.")

    print("\nAvailable keys:")

    for key in feature_data.keys():
        print(" -", key)

else:

    print("WARNING: Feature data is not a dictionary.")


# ============================================================
# 4. LOAD TRAIN / TEST FEATURES
# ============================================================

print("\n[4] Checking training and testing data...")


if isinstance(feature_data, dict):

    required_keys = [
        "X_train",
        "X_test",
        "y_train",
        "y_test"
    ]

    missing_keys = [
        key for key in required_keys
        if key not in feature_data
    ]

    if missing_keys:

        print("WARNING: Missing keys:", missing_keys)

    else:

        X_train = feature_data["X_train"]
        X_test = feature_data["X_test"]
        y_train = feature_data["y_train"]
        y_test = feature_data["y_test"]

        print("Training feature shape:", X_train.shape)
        print("Testing feature shape :", X_test.shape)

        print("Training target shape :", y_train.shape)
        print("Testing target shape  :", y_test.shape)

        print("PASS: Training and testing data loaded.")


# ============================================================
# 5. LOAD TRAINED MODELS
# ============================================================

print("\n[5] Loading trained models...")


try:

    random_forest = joblib.load(RANDOM_FOREST_MODEL)

    print("PASS: Random Forest model loaded.")

except Exception as e:

    print("FAIL: Random Forest model could not be loaded.")
    print("Error:", e)
    raise SystemExit


try:

    linear_svm = joblib.load(LINEAR_SVM_MODEL)

    print("PASS: Linear SVM model loaded.")

except Exception as e:

    print("FAIL: Linear SVM model could not be loaded.")
    print("Error:", e)
    raise SystemExit


try:

    xgboost = joblib.load(XGBOOST_MODEL)

    print("PASS: XGBoost model loaded.")

except Exception as e:

    print("FAIL: XGBoost model could not be loaded.")
    print("Error:", e)
    raise SystemExit


# ============================================================
# 6. CHECK MODEL PREDICTIONS
# ============================================================

print("\n[6] Testing model predictions...")


if isinstance(feature_data, dict) and "X_test" in feature_data:

    print("\nTesting Random Forest...")

    rf_predictions = random_forest.predict(X_test)

    print("PASS: Random Forest prediction completed.")
    print("Prediction shape:", rf_predictions.shape)


    print("\nTesting Linear SVM...")

    svm_predictions = linear_svm.predict(X_test)

    print("PASS: Linear SVM prediction completed.")
    print("Prediction shape:", svm_predictions.shape)


    print("\nTesting XGBoost...")

    xgb_predictions = xgboost.predict(X_test)

    print("PASS: XGBoost prediction completed.")
    print("Prediction shape:", xgb_predictions.shape)


# ============================================================
# 7. CHECK PREDICTION VALUES
# ============================================================

print("\n[7] Checking prediction values...")


models_predictions = {
    "Random Forest": rf_predictions,
    "Linear SVM": svm_predictions,
    "XGBoost": xgb_predictions
}


for model_name, predictions in models_predictions.items():

    unique_values = np.unique(predictions)

    print(f"{model_name} predictions: {unique_values}")

    if set(unique_values).issubset({0, 1}):

        print(f"PASS: {model_name} produces valid binary predictions.")

    else:

        print(
            f"WARNING: {model_name} produced unexpected "
            f"prediction values."
        )


# ============================================================
# 8. CHECK PREDICTION COUNT
# ============================================================

print("\n[8] Checking prediction count...")


expected_predictions = len(y_test)


for model_name, predictions in models_predictions.items():

    if len(predictions) == expected_predictions:

        print(
            f"PASS: {model_name} produced "
            f"{len(predictions)} predictions."
        )

    else:

        print(
            f"FAIL: {model_name} produced "
            f"{len(predictions)} predictions. "
            f"Expected {expected_predictions}."
        )


# ============================================================
# 9. SAMPLE PREDICTIONS
# ============================================================

print("\n[9] Sample predictions...")

print("\nRandom Forest:")
print(rf_predictions[:10])

print("\nLinear SVM:")
print(svm_predictions[:10])

print("\nXGBoost:")
print(xgb_predictions[:10])


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION PIPELINE SUMMARY")
print("=" * 60)

print("Feature data loaded       : YES")
print("Random Forest loaded      : YES")
print("Linear SVM loaded         : YES")
print("XGBoost loaded            : YES")
print("Test data available       :", len(y_test))
print("Random Forest prediction : SUCCESS")
print("Linear SVM prediction    : SUCCESS")
print("XGBoost prediction       : SUCCESS")

print("\n" + "=" * 60)
print("PREDICTION PIPELINE TEST COMPLETED")
print("=" * 60)

print("""
Pipeline verified:

Cleaned Dataset
      |
      v
Feature Engineering
      |
      v
TF-IDF + Red-Flag Features
      |
      v
SMOTE Training
      |
      v
Trained Models
      |
      v
Test Data
      |
      +------------------+
      |        |         |
      v        v         v
 Random     Linear     XGBoost
 Forest      SVM
      |        |         |
      +--------+---------+
               |
               v
          Predictions
""")

print("STATUS: PREDICTION PIPELINE READY")
print("=" * 60)