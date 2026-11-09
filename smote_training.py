import os
import joblib
import numpy as np

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier

FEATURE_FILE = os.path.join(
    "dataset",
    "processed",
    "features",
    "feature_data.pkl"
)

MODEL_DIR = os.path.join(
    "dataset",
    "processed",
    "models"
)

RESULTS_DIR = os.path.join(
    "dataset",
    "processed",
    "results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


SMOTE_FILE = os.path.join(
    RESULTS_DIR,
    "smote_info.pkl"
)

RF_FILE = os.path.join(
    MODEL_DIR,
    "random_forest_model.pkl"
)

SVM_FILE = os.path.join(
    MODEL_DIR,
    "linear_svm_model.pkl"
)

XGB_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_model.pkl"
)


print("=" * 60)
print("SMOTE + MODEL TRAINING")
print("=" * 60)


# ============================================================
# 1. CHECK FEATURE FILE
# ============================================================

print("\n[1] Checking feature file...")

if not os.path.exists(FEATURE_FILE):

    raise FileNotFoundError(
        f"Feature file not found: {FEATURE_FILE}"
    )

print(
    "PASS: Feature file exists."
)

print(
    "File:",
    FEATURE_FILE
)


# ============================================================
# 2. LOAD FEATURE DATA
# ============================================================

print("\n[2] Loading feature-engineering output...")

try:

    feature_data = joblib.load(
        FEATURE_FILE
    )

except Exception as e:

    print(
        "\nERROR: Could not load feature_data.pkl"
    )

    print(
        "Reason:",
        e
    )

    print(
        "\nPlease run feature_engineering.py again "
        "after replacing its saving method with joblib."
    )

    raise


print(
    "PASS: Feature data loaded successfully."
)

print(
    "Available keys:"
)

print(
    list(feature_data.keys())
)


# ============================================================
# 3. CHECK REQUIRED DATA
# ============================================================

print("\n[3] Checking required data...")

required_keys = [
    "X_train",
    "X_test",
    "y_train",
    "y_test"
]

for key in required_keys:

    if key not in feature_data:

        raise KeyError(
            f"Required key not found: {key}"
        )

    print(
        f"PASS: {key} found."
    )


X_train = feature_data["X_train"]
X_test = feature_data["X_test"]

y_train = feature_data["y_train"]
y_test = feature_data["y_test"]


print(
    "\nTraining feature shape:",
    X_train.shape
)

print(
    "Testing feature shape :",
    X_test.shape
)

print(
    "Training target shape :",
    y_train.shape
)

print(
    "Testing target shape  :",
    y_test.shape
)


# ============================================================
# 4. CHECK TARGET DISTRIBUTION
# ============================================================

print("\n[4] Target distribution BEFORE SMOTE")

print(
    "\nTraining target:"
)

print(
    y_train.value_counts()
)

print(
    "\nTesting target:"
)

print(
    y_test.value_counts()
)


# ============================================================
# 5. APPLY SMOTE
# ============================================================

print("\n" + "=" * 60)
print("[5] APPLYING SMOTE")
print("=" * 60)


print(
    "\nOriginal training feature shape:",
    X_train.shape
)

print(
    "Original training target shape:",
    y_train.shape
)


smote = SMOTE(
    random_state=42
)


print(
    "\nRunning SMOTE..."
)


X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


print(
    "PASS: SMOTE completed."
)


print(
    "\nTraining features after SMOTE:",
    X_train_smote.shape
)

print(
    "Training target after SMOTE:",
    y_train_smote.shape
)


print(
    "\nTarget distribution AFTER SMOTE:"
)

unique_values, counts = np.unique(
    y_train_smote,
    return_counts=True
)

for value, count in zip(
    unique_values,
    counts
):

    if value == 0:

        label = "Genuine"

    else:

        label = "Fraudulent"

    print(
        f"{label:12s}: {count}"
    )


# ============================================================
# 6. SAVE SMOTE INFORMATION
# ============================================================

print("\n[6] Saving SMOTE output information...")


smote_data = {

    "original_training_shape":
        X_train.shape,

    "original_target_shape":
        y_train.shape,

    "smote_training_shape":
        X_train_smote.shape,

    "smote_target_shape":
        y_train_smote.shape,

    "X_train_smote":
        X_train_smote,

    "y_train_smote":
        y_train_smote,

    "X_test":
        X_test,

    "y_test":
        y_test
}


joblib.dump(
    smote_data,
    SMOTE_FILE
)


print(
    "PASS: SMOTE information saved."
)

print(
    "File:",
    SMOTE_FILE
)


# ============================================================
# 7. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("[7] TRAINING RANDOM FOREST")
print("=" * 60)


rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight=None
)


print(
    "Training Random Forest..."
)


rf_model.fit(
    X_train_smote,
    y_train_smote
)


print(
    "PASS: Random Forest training completed."
)


joblib.dump(
    rf_model,
    RF_FILE
)


print(
    "PASS: Random Forest saved."
)

print(
    "File:",
    RF_FILE
)


# ============================================================
# 8. TRAIN LINEAR SVM
# ============================================================

print("\n" + "=" * 60)
print("[8] TRAINING LINEAR SVM")
print("=" * 60)


svm_model = LinearSVC(
    random_state=42,
    max_iter=5000
)


print(
    "Training Linear SVM..."
)


svm_model.fit(
    X_train_smote,
    y_train_smote
)


print(
    "PASS: Linear SVM training completed."
)


joblib.dump(
    svm_model,
    SVM_FILE
)


print(
    "PASS: Linear SVM saved."
)

print(
    "File:",
    SVM_FILE
)


# ============================================================
# 9. TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("[9] TRAINING XGBOOST")
print("=" * 60)


xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


print(
    "Training XGBoost..."
)


xgb_model.fit(
    X_train_smote,
    y_train_smote
)


print(
    "PASS: XGBoost training completed."
)


joblib.dump(
    xgb_model,
    XGB_FILE
)


print(
    "PASS: XGBoost saved."
)

print(
    "File:",
    XGB_FILE
)


# ============================================================
# 10. VERIFY SAVED MODELS
# ============================================================

print("\n" + "=" * 60)
print("[10] VERIFYING SAVED MODELS")
print("=" * 60)


model_files = [
    RF_FILE,
    SVM_FILE,
    XGB_FILE,
    SMOTE_FILE
]


for file_path in model_files:

    if os.path.exists(file_path):

        print(
            "PASS:",
            file_path
        )

    else:

        raise FileNotFoundError(
            f"Expected file was not created: {file_path}"
        )


# ============================================================
# 11. VERIFY MODEL FEATURE DIMENSIONS
# ============================================================

print("\n[11] Checking model feature dimensions...")


print(
    "Expected feature count:",
    X_train_smote.shape[1]
)


if hasattr(
    rf_model,
    "n_features_in_"
):

    print(
        "Random Forest:",
        rf_model.n_features_in_
    )


if hasattr(
    svm_model,
    "n_features_in_"
):

    print(
        "SVM:",
        svm_model.n_features_in_
    )


if hasattr(
    xgb_model,
    "n_features_in_"
):

    print(
        "XGBoost:",
        xgb_model.n_features_in_
    )


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SMOTE + MODEL TRAINING SUMMARY")
print("=" * 60)


print(
    "\nOriginal training samples:",
    X_train.shape[0]
)

print(
    "Training samples after SMOTE:",
    X_train_smote.shape[0]
)

print(
    "Testing samples:",
    X_test.shape[0]
)

print(
    "Number of features:",
    X_train_smote.shape[1]
)


print(
    "\nModels trained:"
)

print(
    "1. Random Forest"
)

print(
    "2. Linear SVM"
)

print(
    "3. XGBoost"
)


print(
    "\nSaved files:"
)

print(
    RF_FILE
)

print(
    SVM_FILE
)

print(
    XGB_FILE
)

print(
    SMOTE_FILE
)


print("\n" + "=" * 60)
print("SMOTE + MODEL TRAINING COMPLETED")
print("STATUS: MODELS READY FOR EVALUATION")
print("=" * 60)