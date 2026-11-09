import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
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

EVALUATION_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "evaluation"
)

FEATURE_DATA_FILE = os.path.join(
    FEATURE_DIR,
    "feature_data.pkl"
)

FINAL_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "final_xgboost_model.pkl"
)

os.makedirs(EVALUATION_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FINAL XGBOOST PRODUCTION MODEL VALIDATION")
print("=" * 70)


# ============================================================
# 1. CHECK REQUIRED FILES
# ============================================================

print("\n[1] Checking required files...")

required_files = {
    "Feature data": FEATURE_DATA_FILE,
    "Final XGBoost model": FINAL_MODEL_FILE
}

for name, path in required_files.items():

    if os.path.exists(path):
        print(f"PASS: {name} found.")
        print(path)
    else:
        print(f"FAIL: {name} not found.")
        print(path)
        raise FileNotFoundError(path)


# ============================================================
# 2. LOAD FEATURE DATA
# ============================================================

print("\n" + "=" * 70)
print("[2] Loading saved test dataset")
print("=" * 70)

feature_data = joblib.load(FEATURE_DATA_FILE)

print("PASS: Feature data loaded.")

required_keys = [
    "X_test",
    "y_test",
    "tfidf",
    "combined_feature_names"
]

for key in required_keys:

    if key not in feature_data:
        raise KeyError(
            f"Required key '{key}' is missing from feature_data.pkl"
        )

print("\nAvailable keys:")

for key in feature_data.keys():
    print(f" - {key}")


# ============================================================
# 3. EXTRACT TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("[3] Extracting test data")
print("=" * 70)

X_test = feature_data["X_test"]
y_test = np.asarray(feature_data["y_test"])

print(f"X_test shape : {X_test.shape}")
print(f"y_test shape : {y_test.shape}")


# ============================================================
# 4. LOAD FINAL XGBOOST MODEL
# ============================================================

print("\n" + "=" * 70)
print("[4] Loading final production model")
print("=" * 70)

model = joblib.load(FINAL_MODEL_FILE)

print("PASS: Final XGBoost model loaded.")
print(f"Model type: {type(model).__name__}")


# ============================================================
# 5. VERIFY MODEL DIMENSIONS
# ============================================================

print("\n" + "=" * 70)
print("[5] Verifying feature dimensions")
print("=" * 70)

dataset_features = X_test.shape[1]

if hasattr(model, "n_features_in_"):
    model_features = model.n_features_in_
else:
    model_features = None

print(f"Dataset features : {dataset_features}")
print(f"Model features   : {model_features}")

if model_features is not None:

    if dataset_features != model_features:
        raise ValueError(
            f"Feature mismatch! Dataset has {dataset_features} "
            f"features but model expects {model_features}."
        )

print("PASS: Model and dataset feature dimensions match.")


# ============================================================
# 6. VERIFY TARGET VALUES
# ============================================================

print("\n" + "=" * 70)
print("[6] Checking target values")
print("=" * 70)

unique_targets = np.unique(y_test)

print(f"Target values: {unique_targets}")

if not set(unique_targets).issubset({0, 1}):
    raise ValueError(
        "Target contains values other than 0 and 1."
    )

genuine_count = int(np.sum(y_test == 0))
fraud_count = int(np.sum(y_test == 1))

print(f"Genuine jobs    : {genuine_count}")
print(f"Fraudulent jobs : {fraud_count}")

print("PASS: Target values verified.")


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("[7] Generating predictions with final XGBoost")
print("=" * 70)

print("Running model prediction...")

y_pred = model.predict(X_test)

print("PASS: Class predictions generated.")

# Convert probabilities
if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(X_test)

    probability_fraud = probabilities[:, 1]
    probability_legitimate = probabilities[:, 0]

    print("PASS: Prediction probabilities generated.")

else:

    probability_fraud = None
    probability_legitimate = None

    print("WARNING: Model does not support predict_proba().")


# ============================================================
# 8. CHECK PREDICTION ALIGNMENT
# ============================================================

print("\n" + "=" * 70)
print("[8] Checking prediction alignment")
print("=" * 70)

print(f"Actual labels : {len(y_test)}")
print(f"Predictions   : {len(y_pred)}")

if len(y_test) != len(y_pred):

    raise ValueError(
        "Prediction length does not match target length."
    )

print("PASS: Prediction lengths match.")


# ============================================================
# 9. CALCULATE METRICS
# ============================================================

print("\n" + "=" * 70)
print("[9] Calculating production model metrics")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

if probability_fraud is not None:

    roc_auc = roc_auc_score(
        y_test,
        probability_fraud
    )

else:

    roc_auc = None


# ============================================================
# 10. DISPLAY METRICS
# ============================================================

print("\nFINAL XGBOOST PRODUCTION MODEL")

print("-" * 70)

print(f"Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f}  ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f}  ({recall * 100:.2f}%)")
print(f"F1-score  : {f1:.4f}  ({f1 * 100:.2f}%)")

if roc_auc is not None:
    print(f"ROC-AUC   : {roc_auc:.4f}  ({roc_auc * 100:.2f}%)")


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("[10] Confusion Matrix")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

print()
print("                    Predicted")
print("                 Genuine  Fraud")
print(f"Actual Genuine   {tn:7d} {fp:6d}")
print(f"Actual Fraud     {fn:7d} {tp:6d}")

print("\nConfusion matrix values:")

print(f"True Negative  : {tn}")
print(f"False Positive : {fp}")
print(f"False Negative : {fn}")
print(f"True Positive  : {tp}")


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("[11] Classification Report")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Genuine",
        "Fraudulent"
    ],
    zero_division=0
)

print(report)


# ============================================================
# 13. PREDICTION DISTRIBUTION
# ============================================================

print("=" * 70)
print("[12] Prediction Distribution")
print("=" * 70)

predicted_genuine = int(np.sum(y_pred == 0))
predicted_fraud = int(np.sum(y_pred == 1))

print(f"Predicted Genuine    : {predicted_genuine}")
print(f"Predicted Fraudulent : {predicted_fraud}")


# ============================================================
# 14. CORRECT / INCORRECT PREDICTIONS
# ============================================================

correct_predictions = int(np.sum(y_test == y_pred))
incorrect_predictions = int(np.sum(y_test != y_pred))

print("\nCorrect predictions   :", correct_predictions)
print("Incorrect predictions :", incorrect_predictions)


# ============================================================
# 15. SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("[13] Sample Prediction Verification")
print("=" * 70)

sample_count = min(20, len(y_test))

print(
    "\nShowing first "
    f"{sample_count} test samples:"
)

print("-" * 70)

for i in range(sample_count):

    actual = int(y_test[i])
    predicted = int(y_pred[i])

    actual_label = (
        "Fraudulent"
        if actual == 1
        else "Genuine"
    )

    predicted_label = (
        "Fraudulent"
        if predicted == 1
        else "Genuine"
    )

    result = (
        "CORRECT"
        if actual == predicted
        else "WRONG"
    )

    if probability_fraud is not None:

        fraud_probability = (
            float(probability_fraud[i]) * 100
        )

        print(
            f"Sample {i + 1:02d} | "
            f"Actual: {actual_label:10s} | "
            f"Predicted: {predicted_label:10s} | "
            f"Fraud probability: "
            f"{fraud_probability:7.2f}% | "
            f"{result}"
        )

    else:

        print(
            f"Sample {i + 1:02d} | "
            f"Actual: {actual_label:10s} | "
            f"Predicted: {predicted_label:10s} | "
            f"{result}"
        )


# ============================================================
# 16. FIND WRONG PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("[14] Incorrect Prediction Analysis")
print("=" * 70)

wrong_indices = np.where(
    y_test != y_pred
)[0]

print(
    f"Total incorrect predictions: "
    f"{len(wrong_indices)}"
)

if len(wrong_indices) > 0:

    print("\nFirst 20 incorrect samples:")

    print("-" * 70)

    for index in wrong_indices[:20]:

        actual = int(y_test[index])
        predicted = int(y_pred[index])

        actual_label = (
            "Fraudulent"
            if actual == 1
            else "Genuine"
        )

        predicted_label = (
            "Fraudulent"
            if predicted == 1
            else "Genuine"
        )

        if probability_fraud is not None:

            fraud_probability = (
                float(probability_fraud[index]) * 100
            )

            print(
                f"Index {index:5d} | "
                f"Actual: {actual_label:10s} | "
                f"Predicted: {predicted_label:10s} | "
                f"Fraud probability: "
                f"{fraud_probability:7.2f}%"
            )

        else:

            print(
                f"Index {index:5d} | "
                f"Actual: {actual_label:10s} | "
                f"Predicted: {predicted_label:10s}"
            )

else:

    print("Excellent: No incorrect predictions found.")


# ============================================================
# 17. SAVE DETAILED RESULTS
# ============================================================

print("\n" + "=" * 70)
print("[15] Saving validation results")
print("=" * 70)

results = pd.DataFrame({
    "actual": y_test.astype(int),
    "predicted": y_pred.astype(int)
})

if probability_fraud is not None:

    results["probability_legitimate"] = (
        probability_legitimate
    )

    results["probability_fraud"] = (
        probability_fraud
    )

results["correct"] = (
    results["actual"] == results["predicted"]
)

results["actual_label"] = results["actual"].map({
    0: "Genuine",
    1: "Fraudulent"
})

results["predicted_label"] = results["predicted"].map({
    0: "Genuine",
    1: "Fraudulent"
})


results_file = os.path.join(
    EVALUATION_DIR,
    "final_xgboost_validation_results.csv"
)

results.to_csv(
    results_file,
    index=False
)

print("PASS: Detailed validation results saved.")
print(results_file)


# ============================================================
# 18. SAVE METRICS
# ============================================================

metrics_data = {
    "model": "XGBoost",
    "model_type": type(model).__name__,
    "test_samples": len(y_test),
    "genuine_samples": genuine_count,
    "fraudulent_samples": fraud_count,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "true_negative": tn,
    "false_positive": fp,
    "false_negative": fn,
    "true_positive": tp,
    "correct_predictions": correct_predictions,
    "incorrect_predictions": incorrect_predictions
}

if roc_auc is not None:
    metrics_data["roc_auc"] = roc_auc

metrics_file = os.path.join(
    EVALUATION_DIR,
    "final_xgboost_validation_metrics.csv"
)

pd.DataFrame(
    [metrics_data]
).to_csv(
    metrics_file,
    index=False
)

print("PASS: Validation metrics saved.")
print(metrics_file)


# ============================================================
# 19. FINAL VERDICT
# ============================================================

print("\n" + "=" * 70)
print("FINAL PRODUCTION MODEL VERDICT")
print("=" * 70)

print()
print("Selected production model : XGBoost")
print(f"Accuracy                  : {accuracy * 100:.2f}%")
print(f"Precision                 : {precision * 100:.2f}%")
print(f"Recall                    : {recall * 100:.2f}%")
print(f"F1-score                  : {f1 * 100:.2f}%")

if roc_auc is not None:
    print(f"ROC-AUC                   : {roc_auc * 100:.2f}%")

print()
print(f"Correct predictions       : {correct_predictions}")
print(f"Incorrect predictions     : {incorrect_predictions}")

print()
print("Confusion Matrix:")
print(f"  True Negatives  : {tn}")
print(f"  False Positives : {fp}")
print(f"  False Negatives : {fn}")
print(f"  True Positives  : {tp}")

print()

if accuracy >= 0.95 and f1 >= 0.80:

    print(
        "VERDICT: PASS"
    )

    print(
        "The selected XGBoost model demonstrates "
        "strong performance on the saved test dataset "
        "and is suitable as the production model."
    )

else:

    print(
        "VERDICT: REVIEW REQUIRED"
    )

    print(
        "The selected XGBoost model does not meet "
        "the configured validation thresholds."
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("FINAL XGBOOST VALIDATION COMPLETED")
print("STATUS: SUCCESS")
print("=" * 70)
