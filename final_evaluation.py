import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_DIR = os.path.join(
    "dataset",
    "processed",
    "predictions"
)

EVALUATION_DIR = os.path.join(
    "dataset",
    "processed",
    "evaluation"
)

PREDICTION_FILE = os.path.join(
    PREDICTION_DIR,
    "test_predictions.joblib"
)

RESULTS_CSV = os.path.join(
    PREDICTION_DIR,
    "test_prediction_results.csv"
)


os.makedirs(
    EVALUATION_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

METRICS_CSV = os.path.join(
    EVALUATION_DIR,
    "final_model_metrics.csv"
)

METRICS_PLOT = os.path.join(
    EVALUATION_DIR,
    "model_metrics_comparison.png"
)

RECALL_PLOT = os.path.join(
    EVALUATION_DIR,
    "recall_comparison.png"
)

F1_PLOT = os.path.join(
    EVALUATION_DIR,
    "f1_comparison.png"
)

CONFUSION_PLOT = os.path.join(
    EVALUATION_DIR,
    "confusion_matrices.png"
)

CLASS_DISTRIBUTION_PLOT = os.path.join(
    EVALUATION_DIR,
    "test_class_distribution.png"
)

FINAL_REPORT = os.path.join(
    EVALUATION_DIR,
    "final_evaluation_report.txt"
)


print("=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)


# ============================================================
# 1. CHECK REQUIRED FILES
# ============================================================

print("\n[1] Checking required files...")
print("=" * 60)


if not os.path.exists(PREDICTION_FILE):

    raise FileNotFoundError(
        f"\nPrediction file not found:\n{PREDICTION_FILE}"
    )


print("PASS: Test prediction file found.")
print(PREDICTION_FILE)


if os.path.exists(RESULTS_CSV):

    print("PASS: Prediction results CSV found.")
    print(RESULTS_CSV)

else:

    print(
        "INFO: Prediction results CSV not found."
    )

    print(
        "Metrics will be calculated from saved predictions."
    )


# ============================================================
# 2. LOAD TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("[2] Loading test prediction data")
print("=" * 60)


print("Loading using Joblib...")


try:

    prediction_data = joblib.load(
        PREDICTION_FILE
    )

except Exception as error:

    raise RuntimeError(
        "\nUnable to load test_predictions.joblib.\n"
        f"Error: {error}"
    )


print(
    "PASS: Test prediction data loaded successfully."
)


if not isinstance(
    prediction_data,
    dict
):

    raise TypeError(
        "test_predictions.joblib must contain a dictionary."
    )


print("\nAvailable keys:")

for key in prediction_data.keys():

    print(" -", key)


# ============================================================
# 3. EXTRACT PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("[3] Extracting test predictions")
print("=" * 60)


required_prediction_keys = [

    "y_test",

    "random_forest_predictions",

    "svm_predictions",

    "xgboost_predictions"

]


for key in required_prediction_keys:

    if key not in prediction_data:

        raise KeyError(
            f"Required key '{key}' not found "
            "in test_predictions.joblib."
        )


y_test = np.asarray(
    prediction_data["y_test"]
)


rf_predictions = np.asarray(
    prediction_data[
        "random_forest_predictions"
    ]
)


svm_predictions = np.asarray(
    prediction_data[
        "svm_predictions"
    ]
)


xgb_predictions = np.asarray(
    prediction_data[
        "xgboost_predictions"
    ]
)


print(
    "Test samples:",
    len(y_test)
)

print(
    "Random Forest predictions:",
    len(rf_predictions)
)

print(
    "SVM predictions:",
    len(svm_predictions)
)

print(
    "XGBoost predictions:",
    len(xgb_predictions)
)


# ============================================================
# 4. CHECK PREDICTION ALIGNMENT
# ============================================================

print("\n" + "=" * 60)
print("[4] Checking prediction alignment")
print("=" * 60)


prediction_lengths = [

    len(y_test),

    len(rf_predictions),

    len(svm_predictions),

    len(xgb_predictions)

]


if len(set(prediction_lengths)) != 1:

    raise ValueError(
        "Prediction lengths do not match."
    )


print(
    "PASS: All prediction lengths match."
)


# ============================================================
# 5. CHECK TARGET VALUES
# ============================================================

print("\n" + "=" * 60)
print("[5] Checking target values")
print("=" * 60)


unique_targets = np.unique(
    y_test
)


print(
    "Target values:",
    unique_targets
)


if not set(
    unique_targets
).issubset({0, 1}):

    raise ValueError(
        "Target contains values other than 0 and 1."
    )


print(
    "PASS: Target contains only 0 and 1."
)


genuine_count = int(
    np.sum(y_test == 0)
)

fraudulent_count = int(
    np.sum(y_test == 1)
)


print(
    "\nGenuine jobs    :",
    genuine_count
)

print(
    "Fraudulent jobs :",
    fraudulent_count
)


# ============================================================
# 6. CALCULATE FINAL METRICS
# ============================================================

print("\n" + "=" * 60)
print("[6] Calculating final model metrics")
print("=" * 60)


def calculate_metrics(
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


    return {

        "model": model_name,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1

    }


rf_metrics = calculate_metrics(

    "Random Forest",

    y_test,

    rf_predictions

)


svm_metrics = calculate_metrics(

    "SVM",

    y_test,

    svm_predictions

)


xgb_metrics = calculate_metrics(

    "XGBoost",

    y_test,

    xgb_predictions

)


metrics_df = pd.DataFrame([

    rf_metrics,

    svm_metrics,

    xgb_metrics

])


print("\nFinal model metrics:")

print(
    metrics_df.to_string(
        index=False
    )
)


# ============================================================
# 7. SAVE FINAL METRICS
# ============================================================

print("\n" + "=" * 60)
print("[7] Saving final metrics")
print("=" * 60)


metrics_df.to_csv(

    METRICS_CSV,

    index=False

)


print(
    "PASS: Final metrics saved."
)

print(
    "File:",
    METRICS_CSV
)


# ============================================================
# 8. PRINT PERCENTAGE METRICS
# ============================================================

print("\n" + "=" * 60)
print("[8] Final metric percentages")
print("=" * 60)


for _, row in metrics_df.iterrows():

    print(
        f"\n{row['model']}"
    )

    print(
        f"Accuracy  : {row['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision : {row['precision'] * 100:.2f}%"
    )

    print(
        f"Recall    : {row['recall'] * 100:.2f}%"
    )

    print(
        f"F1-Score  : {row['f1_score'] * 100:.2f}%"
    )


# ============================================================
# 9. MODEL METRICS COMPARISON GRAPH
# ============================================================

print("\n" + "=" * 60)
print("[9] Creating model metrics comparison graph")
print("=" * 60)


models = metrics_df["model"]

x = np.arange(
    len(models)
)

width = 0.20


plt.figure(
    figsize=(10, 6)
)


plt.bar(

    x - 1.5 * width,

    metrics_df["accuracy"],

    width,

    label="Accuracy"

)


plt.bar(

    x - 0.5 * width,

    metrics_df["precision"],

    width,

    label="Precision"

)


plt.bar(

    x + 0.5 * width,

    metrics_df["recall"],

    width,

    label="Recall"

)


plt.bar(

    x + 1.5 * width,

    metrics_df["f1_score"],

    width,

    label="F1-Score"

)


plt.xticks(

    x,

    models

)


plt.ylabel(
    "Score"
)


plt.xlabel(
    "Model"
)


plt.title(
    "Model Performance Comparison"
)


plt.ylim(
    0,
    1.05
)


plt.legend()


plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    METRICS_PLOT,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "PASS: Model comparison graph saved."
)

print(
    METRICS_PLOT
)


# ============================================================
# 10. RECALL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("[10] Creating recall comparison graph")
print("=" * 60)


plt.figure(
    figsize=(8, 5)
)


bars = plt.bar(

    models,

    metrics_df["recall"]

)


plt.ylabel(
    "Recall"
)


plt.xlabel(
    "Model"
)


plt.title(
    "Fraud Detection Recall Comparison"
)


plt.ylim(
    0,
    1.05
)


plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)


for bar, value in zip(

    bars,

    metrics_df["recall"]

):

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height()
        + 0.02,

        f"{value:.3f}",

        ha="center"

    )


plt.tight_layout()


plt.savefig(

    RECALL_PLOT,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "PASS: Recall comparison saved."
)

print(
    RECALL_PLOT
)


# ============================================================
# 11. F1-SCORE COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("[11] Creating F1-score comparison graph")
print("=" * 60)


plt.figure(
    figsize=(8, 5)
)


bars = plt.bar(

    models,

    metrics_df["f1_score"]

)


plt.ylabel(
    "F1-Score"
)


plt.xlabel(
    "Model"
)


plt.title(
    "Model F1-Score Comparison"
)


plt.ylim(
    0,
    1.05
)


plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)


for bar, value in zip(

    bars,

    metrics_df["f1_score"]

):

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height()
        + 0.02,

        f"{value:.3f}",

        ha="center"

    )


plt.tight_layout()


plt.savefig(

    F1_PLOT,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "PASS: F1-score comparison saved."
)

print(
    F1_PLOT
)


# ============================================================
# 12. CREATE CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 60)
print("[12] Creating confusion matrices")
print("=" * 60)


confusion_data = {

    "Random Forest":
        rf_predictions,

    "SVM":
        svm_predictions,

    "XGBoost":
        xgb_predictions

}


fig, axes = plt.subplots(

    1,

    3,

    figsize=(15, 4)

)


for ax, (model_name, predictions) in zip(

    axes,

    confusion_data.items()

):

    cm = confusion_matrix(

        y_test,

        predictions

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

        ],

        ax=ax

    )


    ax.set_title(
        model_name
    )


    ax.set_xlabel(
        "Predicted"
    )


    ax.set_ylabel(
        "Actual"
    )


plt.suptitle(
    "Confusion Matrix Comparison"
)


plt.tight_layout()


plt.savefig(

    CONFUSION_PLOT,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "PASS: Confusion matrix comparison saved."
)

print(
    CONFUSION_PLOT
)


# ============================================================
# 13. TEST CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("[13] Creating test class distribution graph")
print("=" * 60)


class_counts = pd.Series({

    "Genuine":
        genuine_count,

    "Fraudulent":
        fraudulent_count

})


plt.figure(
    figsize=(7, 5)
)


bars = plt.bar(

    class_counts.index,

    class_counts.values

)


plt.ylabel(
    "Number of Jobs"
)


plt.xlabel(
    "Class"
)


plt.title(
    "Test Dataset Class Distribution"
)


plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)


for bar, value in zip(

    bars,

    class_counts.values

):

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        str(value),

        ha="center",

        va="bottom"

    )


plt.tight_layout()


plt.savefig(

    CLASS_DISTRIBUTION_PLOT,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "PASS: Class distribution graph saved."
)

print(
    CLASS_DISTRIBUTION_PLOT
)


# ============================================================
# 14. CALCULATE CONFUSION MATRIX VALUES
# ============================================================

print("\n" + "=" * 60)
print("[14] Calculating confusion matrix values")
print("=" * 60)


confusion_summary = []


for model_name, predictions in confusion_data.items():

    tn, fp, fn, tp = confusion_matrix(

        y_test,

        predictions

    ).ravel()


    confusion_summary.append({

        "model":
            model_name,

        "true_negative":
            int(tn),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "true_positive":
            int(tp)

    })


confusion_df = pd.DataFrame(
    confusion_summary
)


print(
    confusion_df.to_string(
        index=False
    )
)


CONFUSION_CSV = os.path.join(

    EVALUATION_DIR,

    "confusion_matrix_values.csv"

)


confusion_df.to_csv(

    CONFUSION_CSV,

    index=False

)


print(
    "\nPASS: Confusion matrix values saved."
)

print(
    CONFUSION_CSV
)


# ============================================================
# 15. GENERATE FINAL TEXT REPORT
# ============================================================

print("\n" + "=" * 60)
print("[15] Generating final evaluation report")
print("=" * 60)


report_lines = []


report_lines.append(
    "=" * 60
)

report_lines.append(
    "MACHINE LEARNING BASED ONLINE RECRUITMENT FRAUD DETECTION"
)

report_lines.append(
    "FINAL MODEL EVALUATION REPORT"
)

report_lines.append(
    "=" * 60
)


report_lines.append(
    ""
)


report_lines.append(
    "TEST DATASET"
)


report_lines.append(
    "-" * 40
)


report_lines.append(
    f"Total test samples : {len(y_test)}"
)


report_lines.append(
    f"Genuine jobs       : {genuine_count}"
)


report_lines.append(
    f"Fraudulent jobs    : {fraudulent_count}"
)


fraud_percentage = (

    fraudulent_count
    / len(y_test)
    * 100

)


report_lines.append(
    f"Fraud percentage   : {fraud_percentage:.2f}%"
)


report_lines.append(
    ""
)


report_lines.append(
    "MODEL PERFORMANCE"
)


report_lines.append(
    "-" * 40
)


for _, row in metrics_df.iterrows():

    report_lines.append(
        f"\n{row['model']}"
    )

    report_lines.append(
        f"Accuracy  : {row['accuracy']:.4f}"
    )

    report_lines.append(
        f"Precision : {row['precision']:.4f}"
    )

    report_lines.append(
        f"Recall    : {row['recall']:.4f}"
    )

    report_lines.append(
        f"F1-Score  : {row['f1_score']:.4f}"
    )


report_lines.append(
    ""
)


report_lines.append(
    "CONFUSION MATRIX VALUES"
)


report_lines.append(
    "-" * 40
)


for _, row in confusion_df.iterrows():

    report_lines.append(
        f"\n{row['model']}"
    )

    report_lines.append(
        f"True Negative  : {row['true_negative']}"
    )

    report_lines.append(
        f"False Positive : {row['false_positive']}"
    )

    report_lines.append(
        f"False Negative : {row['false_negative']}"
    )

    report_lines.append(
        f"True Positive  : {row['true_positive']}"
    )


report_lines.append(
    ""
)


report_lines.append(
    "EVALUATION OUTPUTS"
)


report_lines.append(
    "-" * 40
)


report_lines.append(
    "1. final_model_metrics.csv"
)

report_lines.append(
    "2. confusion_matrix_values.csv"
)

report_lines.append(
    "3. model_metrics_comparison.png"
)

report_lines.append(
    "4. recall_comparison.png"
)

report_lines.append(
    "5. f1_comparison.png"
)

report_lines.append(
    "6. confusion_matrices.png"
)

report_lines.append(
    "7. test_class_distribution.png"
)


report_lines.append(
    ""
)


report_lines.append(
    "STATUS: FINAL MODEL EVALUATION COMPLETED"
)


report_lines.append(
    "=" * 60
)


with open(

    FINAL_REPORT,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        "\n".join(report_lines)
    )


print(
    "PASS: Final evaluation report saved."
)

print(
    FINAL_REPORT
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL EVALUATION SUMMARY")
print("=" * 60)


print(
    "Test samples       :",
    len(y_test)
)

print(
    "Genuine jobs       :",
    genuine_count
)

print(
    "Fraudulent jobs    :",
    fraudulent_count
)


print("\nModels evaluated:")

print(
    "1. Random Forest"
)

print(
    "2. SVM"
)

print(
    "3. XGBoost"
)


print("\nFinal metrics:")

print(
    metrics_df.to_string(
        index=False
    )
)


print("\nGenerated files:")

print(
    " - final_model_metrics.csv"
)

print(
    " - confusion_matrix_values.csv"
)

print(
    " - model_metrics_comparison.png"
)

print(
    " - recall_comparison.png"
)

print(
    " - f1_comparison.png"
)

print(
    " - confusion_matrices.png"
)

print(
    " - test_class_distribution.png"
)

print(
    " - final_evaluation_report.txt"
)


print("\nOutput directory:")

print(
    EVALUATION_DIR
)


print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION COMPLETED")
print("STATUS: READY FOR FINAL RESULTS AND DEPLOYMENT")
print("=" * 60)