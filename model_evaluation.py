import os
import pickle
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


# ============================================================
# CONFIGURATION
# ============================================================

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

SMOTE_FILE = os.path.join(
    "dataset",
    "processed",
    "results",
    "smote_info.pkl"
)

RESULTS_DIR = os.path.join(
    MODEL_DIR,
    "evaluation"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def load_serialized_file(file_path):
    """
    Load a serialized Python object.

    First tries standard pickle.
    If pickle fails, tries joblib.
    """

    print(
        f"\nLoading file: {file_path}"
    )

    # --------------------------------------------------------
    # Try standard pickle
    # --------------------------------------------------------

    try:

        with open(
            file_path,
            "rb"
        ) as file:

            data = pickle.load(file)

        print(
            "PASS: Loaded using pickle."
        )

        return data

    except Exception as pickle_error:

        print(
            "INFO: Standard pickle loading failed."
        )

        print(
            "Pickle error:",
            pickle_error
        )


    # --------------------------------------------------------
    # Try joblib
    # --------------------------------------------------------

    try:

        data = joblib.load(
            file_path
        )

        print(
            "PASS: Loaded using joblib."
        )

        return data

    except Exception as joblib_error:

        print(
            "ERROR: Joblib loading also failed."
        )

        print(
            "Joblib error:",
            joblib_error
        )

        raise ValueError(
            f"Unable to load serialized file: {file_path}"
        )


# ============================================================
# START
# ============================================================

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


# ============================================================
# 1. CHECK REQUIRED FILES
# ============================================================

print("\n[1] Checking required files...")

required_files = [
    FEATURE_FILE,
    RF_FILE,
    SVM_FILE,
    XGB_FILE,
    SMOTE_FILE
]

for file_path in required_files:

    if os.path.exists(file_path):

        print(
            "PASS:",
            file_path
        )

    else:

        print(
            "ERROR: File not found:",
            file_path
        )

        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\n[2] Loading test data...")

feature_data = load_serialized_file(
    FEATURE_FILE
)


# ------------------------------------------------------------
# Check loaded object
# ------------------------------------------------------------

if not isinstance(
    feature_data,
    dict
):

    raise TypeError(
        "feature_data.pkl does not contain a dictionary."
    )


print(
    "\nAvailable feature-data keys:"
)

print(
    list(
        feature_data.keys()
    )
)


# ------------------------------------------------------------
# Check required keys
# ------------------------------------------------------------

required_feature_keys = [
    "X_test",
    "y_test"
]

for key in required_feature_keys:

    if key not in feature_data:

        raise KeyError(
            f"{key} not found in feature_data.pkl"
        )


# ------------------------------------------------------------
# Extract test data
# ------------------------------------------------------------

X_test = feature_data[
    "X_test"
]

y_test = feature_data[
    "y_test"
]


print(
    "\nPASS: Test data loaded successfully."
)

print(
    "Test feature shape:",
    X_test.shape
)

print(
    "Test target shape :",
    y_test.shape
)


# ============================================================
# 3. LOAD SMOTE INFORMATION
# ============================================================

print("\n[3] Loading SMOTE information...")

smote_data = load_serialized_file(
    SMOTE_FILE
)


if isinstance(
    smote_data,
    dict
):

    print(
        "Available SMOTE keys:"
    )

    print(
        list(
            smote_data.keys()
        )
    )


print(
    "PASS: SMOTE information loaded."
)


# ============================================================
# 4. TEST TARGET DISTRIBUTION
# ============================================================

print(
    "\nTest target distribution:"
)

print(
    y_test.value_counts()
)


# ============================================================
# 5. LOAD TRAINED MODELS
# ============================================================

print("\n[4] Loading trained models...")


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

rf_model = load_serialized_file(
    RF_FILE
)

print(
    "PASS: Random Forest loaded."
)


# ------------------------------------------------------------
# SVM
# ------------------------------------------------------------

svm_model = load_serialized_file(
    SVM_FILE
)

print(
    "PASS: SVM loaded."
)


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

xgb_model = load_serialized_file(
    XGB_FILE
)

print(
    "PASS: XGBoost loaded."
)


# ============================================================
# 6. CHECK FEATURE DIMENSIONS
# ============================================================

print(
    "\n[5] Checking feature dimensions..."
)

print(
    "Test feature count:",
    X_test.shape[1]
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

if hasattr(
    rf_model,
    "n_features_in_"
):

    print(
        "Random Forest expected features:",
        rf_model.n_features_in_
    )

    if (
        rf_model.n_features_in_
        != X_test.shape[1]
    ):

        raise ValueError(
            "Random Forest feature count does not match X_test."
        )


# ------------------------------------------------------------
# SVM
# ------------------------------------------------------------

if hasattr(
    svm_model,
    "n_features_in_"
):

    print(
        "SVM expected features:",
        svm_model.n_features_in_
    )

    if (
        svm_model.n_features_in_
        != X_test.shape[1]
    ):

        raise ValueError(
            "SVM feature count does not match X_test."
        )


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

if hasattr(
    xgb_model,
    "n_features_in_"
):

    print(
        "XGBoost expected features:",
        xgb_model.n_features_in_
    )

    if (
        xgb_model.n_features_in_
        != X_test.shape[1]
    ):

        raise ValueError(
            "XGBoost feature count does not match X_test."
        )


print(
    "PASS: Feature dimensions are compatible."
)


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "[6] GENERATING TEST PREDICTIONS"
)

print(
    "=" * 60
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

print(
    "\nPredicting with Random Forest..."
)

rf_predictions = rf_model.predict(
    X_test
)

print(
    "PASS: Random Forest predictions generated."
)


# ------------------------------------------------------------
# SVM
# ------------------------------------------------------------

print(
    "\nPredicting with SVM..."
)

svm_predictions = svm_model.predict(
    X_test
)

print(
    "PASS: SVM predictions generated."
)


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

print(
    "\nPredicting with XGBoost..."
)

xgb_predictions = xgb_model.predict(
    X_test
)

print(
    "PASS: XGBoost predictions generated."
)


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    print(
        "\n" + "-" * 60
    )

    print(
        model_name
    )

    print(
        "-" * 60
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-Score  : {f1:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Genuine",
                "Fraudulent"
            ],
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    }


# ============================================================
# 9. EVALUATE ALL MODELS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "[7] MODEL PERFORMANCE"
)

print(
    "=" * 60
)


rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)


svm_results = evaluate_model(
    "SVM",
    y_test,
    svm_predictions
)


xgb_results = evaluate_model(
    "XGBoost",
    y_test,
    xgb_predictions
)


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

results = pd.DataFrame([
    rf_results,
    svm_results,
    xgb_results
])


print(
    "\n" + "=" * 60
)

print(
    "[8] MODEL COMPARISON"
)

print(
    "=" * 60
)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE MODEL COMPARISON
# ============================================================

results_file = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)


results.to_csv(
    results_file,
    index=False
)


print(
    "\nPASS: Model comparison saved."
)

print(
    "File:",
    results_file
)


# ============================================================
# 12. CONFUSION MATRICES
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "[9] CREATING CONFUSION MATRICES"
)

print(
    "=" * 60
)


model_predictions = {

    "Random Forest":
        rf_predictions,

    "SVM":
        svm_predictions,

    "XGBoost":
        xgb_predictions

}


for model_name, predictions in model_predictions.items():

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(
        "\n" + model_name
    )

    print(
        cm
    )


    plt.figure(
        figsize=(6, 5)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Genuine",
            "Fraudulent"
        ],
        yticklabels=[
            "Genuine",
            "Fraudulent"
        ]
    )


    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )


    plt.tight_layout()


    safe_name = (
        model_name
        .lower()
        .replace(
            " ",
            "_"
        )
    )


    plot_file = os.path.join(
        RESULTS_DIR,
        f"{safe_name}_confusion_matrix.png"
    )


    plt.savefig(
        plot_file,
        dpi=300
    )


    plt.close()


    print(
        "PASS: Confusion matrix saved:",
        plot_file
    )


# ============================================================
# 13. ROC-AUC EVALUATION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "[10] ROC-AUC EVALUATION"
)

print(
    "=" * 60
)


roc_results = []


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

rf_probabilities = rf_model.predict_proba(
    X_test
)[:, 1]


rf_auc = roc_auc_score(
    y_test,
    rf_probabilities
)


roc_results.append({
    "Model": "Random Forest",
    "ROC_AUC": rf_auc
})


print(
    f"Random Forest ROC-AUC : {rf_auc:.4f}"
)


# ------------------------------------------------------------
# SVM
# ------------------------------------------------------------

svm_scores = svm_model.decision_function(
    X_test
)


svm_auc = roc_auc_score(
    y_test,
    svm_scores
)


roc_results.append({
    "Model": "SVM",
    "ROC_AUC": svm_auc
})


print(
    f"SVM ROC-AUC           : {svm_auc:.4f}"
)


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

xgb_probabilities = xgb_model.predict_proba(
    X_test
)[:, 1]


xgb_auc = roc_auc_score(
    y_test,
    xgb_probabilities
)


roc_results.append({
    "Model": "XGBoost",
    "ROC_AUC": xgb_auc
})


print(
    f"XGBoost ROC-AUC       : {xgb_auc:.4f}"
)


# ============================================================
# 14. SAVE ROC-AUC RESULTS
# ============================================================

roc_results_df = pd.DataFrame(
    roc_results
)


roc_file = os.path.join(
    RESULTS_DIR,
    "roc_auc_results.csv"
)


roc_results_df.to_csv(
    roc_file,
    index=False
)


print(
    "\nPASS: ROC-AUC results saved."
)

print(
    "File:",
    roc_file
)


# ============================================================
# 15. CREATE ROC CURVE
# ============================================================

print(
    "\n[11] Creating ROC curve..."
)


rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probabilities
)


svm_fpr, svm_tpr, _ = roc_curve(
    y_test,
    svm_scores
)


xgb_fpr, xgb_tpr, _ = roc_curve(
    y_test,
    xgb_probabilities
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_auc:.3f})"
)


plt.plot(
    svm_fpr,
    svm_tpr,
    label=f"SVM (AUC = {svm_auc:.3f})"
)


plt.plot(
    xgb_fpr,
    xgb_tpr,
    label=f"XGBoost (AUC = {xgb_auc:.3f})"
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve Comparison"
)

plt.legend()

plt.tight_layout()


roc_plot = os.path.join(
    RESULTS_DIR,
    "roc_curve_comparison.png"
)


plt.savefig(
    roc_plot,
    dpi=300
)


plt.close()


print(
    "PASS: ROC curve saved:",
    roc_plot
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "MODEL EVALUATION SUMMARY"
)

print(
    "=" * 60
)


print(
    "\nTest samples:",
    len(y_test)
)


print(
    "Test feature count:",
    X_test.shape[1]
)


print(
    "\nModels evaluated:"
)

print(
    "1. Random Forest"
)

print(
    "2. SVM"
)

print(
    "3. XGBoost"
)


print(
    "\nPerformance:"
)

print(
    results.to_string(
        index=False
    )
)


print(
    "\nROC-AUC:"
)

print(
    roc_results_df.to_string(
        index=False
    )
)


print(
    "\nEvaluation files saved in:"
)

print(
    RESULTS_DIR
)


print(
    "\n" + "=" * 60
)

print(
    "MODEL EVALUATION COMPLETED"
)

print(
    "STATUS: RESULTS READY FOR ANALYSIS + SHAP"
)

print(
    "=" * 60
)