import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)


# ---------------------------------------------------------
# 1. LOAD FEATURE DATA
# ---------------------------------------------------------

print("\n[1] Loading feature data...")

feature_path = (
    "dataset/processed/features/feature_data.pkl"
)

try:

    feature_data = joblib.load(feature_path)

    print("PASS: Feature data loaded successfully.")

except Exception as e:

    print("FAIL: Could not load feature data.")
    print("Error:", e)
    raise


X_test = feature_data["X_test"]
y_test = feature_data["y_test"]


print("Test feature shape:", X_test.shape)
print("Test target shape :", y_test.shape)


# ---------------------------------------------------------
# 2. LOAD TRAINED MODELS
# ---------------------------------------------------------

print("\n[2] Loading trained models...")

model_paths = {

    "Random Forest":
        "dataset/processed/models/random_forest_model.pkl",

    "Linear SVM":
        "dataset/processed/models/linear_svm_model.pkl",

    "XGBoost":
        "dataset/processed/models/xgboost_model.pkl"
}


models = {}


for name, path in model_paths.items():

    try:

        models[name] = joblib.load(path)

        print(f"PASS: {name} loaded.")

    except Exception as e:

        print(f"FAIL: Could not load {name}.")
        print("Error:", e)
        raise


# ---------------------------------------------------------
# 3. EVALUATE MODELS
# ---------------------------------------------------------

results = {}


print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)


for name, model in models.items():

    print(f"\n{'-' * 70}")
    print(name)
    print(f"{'-' * 70}")


    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    y_pred = model.predict(X_test)


    # -----------------------------------------------------
    # Basic Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


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


    # -----------------------------------------------------
    # ROC-AUC
    # -----------------------------------------------------

    roc_auc = None


    if hasattr(model, "predict_proba"):

        y_score = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            y_score
        )


    elif hasattr(model, "decision_function"):

        y_score = model.decision_function(X_test)

        roc_auc = roc_auc_score(
            y_test,
            y_score
        )


    # -----------------------------------------------------
    # Confusion Matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    # -----------------------------------------------------
    # Display Results
    # -----------------------------------------------------

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")


    if roc_auc is not None:

        print(f"ROC-AUC   : {roc_auc:.4f}")

    else:

        print("ROC-AUC   : Not available")


    print("\nConfusion Matrix:")

    print(cm)


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Genuine",
                "Fraudulent"
            ],
            zero_division=0
        )
    )


    # -----------------------------------------------------
    # Store Results
    # -----------------------------------------------------

    results[name] = {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc_auc
    }


# ---------------------------------------------------------
# 4. COMPARISON TABLE
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)


print(
    f"{'Model':<18}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'ROC-AUC':<12}"
)


print("-" * 70)


for name, metrics in results.items():

    roc = (

        f"{metrics['roc_auc']:.4f}"

        if metrics["roc_auc"] is not None

        else "N/A"
    )


    print(

        f"{name:<18}"
        f"{metrics['accuracy']:<12.4f}"
        f"{metrics['precision']:<12.4f}"
        f"{metrics['recall']:<12.4f}"
        f"{metrics['f1']:<12.4f}"
        f"{roc:<12}"
    )


# ---------------------------------------------------------
# 5. METRIC INFORMATION
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("METRIC COMPARISON")
print("=" * 70)


for name, metrics in results.items():

    print(f"\n{name}")

    print(f"  Accuracy : {metrics['accuracy']:.4f}")

    print(f"  Precision: {metrics['precision']:.4f}")

    print(f"  Recall   : {metrics['recall']:.4f}")

    print(f"  F1-score : {metrics['f1']:.4f}")


    if metrics["roc_auc"] is not None:

        print(
            f"  ROC-AUC  : "
            f"{metrics['roc_auc']:.4f}"
        )


# ---------------------------------------------------------
# 6. SAVE RESULTS
# ---------------------------------------------------------

results_path = (
    "dataset/processed/models/"
    "model_comparison_results.pkl"
)


joblib.dump(
    results,
    results_path
)


print("\n")
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(
    "File:",
    results_path
)


print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)