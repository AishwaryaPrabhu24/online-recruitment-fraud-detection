import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


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

OUTPUT_DIR = os.path.join(
    "dataset",
    "processed",
    "predictions"
)

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "test_predictions.joblib"
)


RF_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "random_forest_model.pkl"
)

SVM_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "linear_svm_model.pkl"
)

XGB_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_model.pkl"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 60)
print("TEST PREDICTION")
print("=" * 60)


# ============================================================
# 1. CHECK REQUIRED FILES
# ============================================================

print("\n[1] Checking required files...")


required_files = {

    "Feature data":
        FEATURE_FILE,

    "Random Forest model":
        RF_MODEL_FILE,

    "SVM model":
        SVM_MODEL_FILE,

    "XGBoost model":
        XGB_MODEL_FILE

}


for file_name, file_path in required_files.items():

    if not os.path.exists(file_path):

        print("\nERROR: Required file not found.")

        print("Type:", file_name)

        print("File:", file_path)

        raise FileNotFoundError(
            file_path
        )

    print(
        f"PASS: {file_name}"
    )

    print(
        f"      {file_path}"
    )


# ============================================================
# 2. LOAD FEATURE DATA USING JOBLIB
# ============================================================

print("\n" + "=" * 60)
print("[2] Loading feature-engineering data")
print("=" * 60)


print(
    "Loading feature data using Joblib..."
)


try:

    feature_data = joblib.load(
        FEATURE_FILE
    )

except Exception as error:

    raise RuntimeError(

        "\nUnable to load feature data using Joblib.\n"

        f"File: {FEATURE_FILE}\n"

        f"Error: {error}"

    )


print(
    "PASS: Feature data loaded successfully."
)


# ============================================================
# 3. CHECK FEATURE DATA STRUCTURE
# ============================================================

print("\n" + "=" * 60)
print("[3] Checking feature data structure")
print("=" * 60)


if not isinstance(
    feature_data,
    dict
):

    raise TypeError(

        "Feature data must be a dictionary."

    )


print(
    "Available feature-data keys:"
)


for key in feature_data.keys():

    print(
        " -",
        key
    )


# ============================================================
# 4. FIND X_TEST
# ============================================================

print("\n" + "=" * 60)
print("[4] Loading test features")
print("=" * 60)


possible_x_test_keys = [

    "X_test",

    "X_test_combined",

    "X_test_features",

    "X_test_final"

]


X_test = None

x_test_key = None


for key in possible_x_test_keys:

    if key in feature_data:

        X_test = feature_data[key]

        x_test_key = key

        break


if X_test is None:

    raise KeyError(

        "Could not find X_test in feature_data.pkl.\n"

        "Expected one of:\n"

        f"{possible_x_test_keys}"

    )


print(
    "X_test loaded from:",
    x_test_key
)


print(
    "X_test shape:",
    X_test.shape
)


# ============================================================
# 5. FIND Y_TEST
# ============================================================

print("\n" + "=" * 60)
print("[5] Loading test target")
print("=" * 60)


possible_y_test_keys = [

    "y_test",

    "y_test_final",

    "test_target",

    "y_test_labels"

]


y_test = None

y_test_key = None


for key in possible_y_test_keys:

    if key in feature_data:

        y_test = feature_data[key]

        y_test_key = key

        break


if y_test is None:

    raise KeyError(

        "Could not find y_test in feature_data.pkl.\n"

        "Expected one of:\n"

        f"{possible_y_test_keys}"

    )


print(
    "y_test loaded from:",
    y_test_key
)


print(
    "y_test shape:",
    np.asarray(y_test).shape
)


# ============================================================
# 6. CHECK TEST DATA
# ============================================================

print("\n" + "=" * 60)
print("[6] Checking test data")
print("=" * 60)


test_rows = X_test.shape[0]

target_rows = len(y_test)


print(
    "Test feature rows :",
    test_rows
)

print(
    "Test target rows  :",
    target_rows
)


if test_rows != target_rows:

    raise ValueError(

        "Feature and target row counts do not match."

    )


print(
    "PASS: Test feature and target rows match."
)


# ------------------------------------------------------------
# Target distribution
# ------------------------------------------------------------

print(
    "\nTest target distribution:"
)


target_distribution = pd.Series(

    np.asarray(y_test),

    name="fraudulent"

).value_counts().sort_index()


print(
    target_distribution
)


# ============================================================
# 7. CHECK TEST FEATURE TYPE
# ============================================================

print("\n" + "=" * 60)
print("[7] Checking feature representation")
print("=" * 60)


print(
    "Feature object type:",
    type(X_test)
)


if hasattr(
    X_test,
    "format"
):

    print(
        "Sparse matrix format:",
        X_test.format
    )


if hasattr(
    X_test,
    "shape"
):

    print(
        "Number of test samples:",
        X_test.shape[0]
    )

    print(
        "Number of features:",
        X_test.shape[1]
    )


print(
    "PASS: Test feature matrix is available."
)


# ============================================================
# 8. LOAD TRAINED MODELS USING JOBLIB
# ============================================================

print("\n" + "=" * 60)
print("[8] Loading trained models")
print("=" * 60)


print(
    "Loading Random Forest..."
)


try:

    rf_model = joblib.load(
        RF_MODEL_FILE
    )

except Exception as error:

    raise RuntimeError(

        f"Unable to load Random Forest model.\n"
        f"File: {RF_MODEL_FILE}\n"
        f"Error: {error}"

    )


print(
    "PASS: Random Forest loaded."
)


print(
    "Loading SVM..."
)


try:

    svm_model = joblib.load(
        SVM_MODEL_FILE
    )

except Exception as error:

    raise RuntimeError(

        f"Unable to load SVM model.\n"
        f"File: {SVM_MODEL_FILE}\n"
        f"Error: {error}"

    )


print(
    "PASS: SVM loaded."
)


print(
    "Loading XGBoost..."
)


try:

    xgb_model = joblib.load(
        XGB_MODEL_FILE
    )

except Exception as error:

    raise RuntimeError(

        f"Unable to load XGBoost model.\n"
        f"File: {XGB_MODEL_FILE}\n"
        f"Error: {error}"

    )


print(
    "PASS: XGBoost loaded."
)


# ============================================================
# 9. CHECK MODEL FEATURE DIMENSIONS
# ============================================================

print("\n" + "=" * 60)
print("[9] Checking model feature dimensions")
print("=" * 60)


test_feature_count = X_test.shape[1]


print(
    "Test feature count:",
    test_feature_count
)


models = {

    "Random Forest":
        rf_model,

    "SVM":
        svm_model,

    "XGBoost":
        xgb_model

}


for model_name, model in models.items():

    if hasattr(
        model,
        "n_features_in_"
    ):

        model_feature_count = (
            model.n_features_in_
        )

        print(
            f"{model_name} expected features:",
            model_feature_count
        )

        if model_feature_count != test_feature_count:

            raise ValueError(

                f"\n{model_name} feature mismatch.\n"

                f"Model expects: "
                f"{model_feature_count}\n"

                f"Test data contains: "
                f"{test_feature_count}"

            )

    else:

        print(
            f"{model_name}: "
            "feature count attribute not available."
        )


print(
    "\nPASS: Feature dimensions are compatible."
)


# ============================================================
# 10. RANDOM FOREST PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("[10] RANDOM FOREST PREDICTION")
print("=" * 60)


rf_predictions = rf_model.predict(
    X_test
)


print(
    "PASS: Random Forest predictions generated."
)


print(
    "Prediction count:",
    len(rf_predictions)
)


print(
    "\nPrediction distribution:"
)


print(
    pd.Series(
        rf_predictions,
        name="prediction"
    ).value_counts().sort_index()
)


# ============================================================
# 11. SVM PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("[11] SVM PREDICTION")
print("=" * 60)


svm_predictions = svm_model.predict(
    X_test
)


print(
    "PASS: SVM predictions generated."
)


print(
    "Prediction count:",
    len(svm_predictions)
)


print(
    "\nPrediction distribution:"
)


print(
    pd.Series(
        svm_predictions,
        name="prediction"
    ).value_counts().sort_index()
)


# ============================================================
# 12. XGBOOST PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("[12] XGBOOST PREDICTION")
print("=" * 60)


xgb_predictions = xgb_model.predict(
    X_test
)


print(
    "PASS: XGBoost predictions generated."
)


print(
    "Prediction count:",
    len(xgb_predictions)
)


print(
    "\nPrediction distribution:"
)


print(
    pd.Series(
        xgb_predictions,
        name="prediction"
    ).value_counts().sort_index()
)


# ============================================================
# 13. CHECK PREDICTION ALIGNMENT
# ============================================================

print("\n" + "=" * 60)
print("[13] CHECKING PREDICTION ALIGNMENT")
print("=" * 60)


if len(rf_predictions) != len(y_test):

    raise ValueError(
        "Random Forest prediction count does not match y_test."
    )


if len(svm_predictions) != len(y_test):

    raise ValueError(
        "SVM prediction count does not match y_test."
    )


if len(xgb_predictions) != len(y_test):

    raise ValueError(
        "XGBoost prediction count does not match y_test."
    )


print(
    "PASS: All prediction counts match test labels."
)


# ============================================================
# 14. CALCULATE MODEL METRICS
# ============================================================

print("\n" + "=" * 60)
print("[14] CALCULATING MODEL METRICS")
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


    print(
        "\n" + model_name
    )

    print(
        "-" * 40
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


    return {

        "model":
            model_name,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1_score":
            f1

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


# ============================================================
# 15. CREATE RESULTS TABLE
# ============================================================

print("\n" + "=" * 60)
print("[15] MODEL RESULTS")
print("=" * 60)


results_df = pd.DataFrame([

    rf_metrics,

    svm_metrics,

    xgb_metrics

])


print(
    "\n"
)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 16. CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 60)
print("[16] CONFUSION MATRICES")
print("=" * 60)


def save_confusion_matrix(

    model_name,

    y_true,

    y_pred,

    output_file

):

    cm = confusion_matrix(

        y_true,

        y_pred

    )


    print(
        f"\n{model_name}"
    )


    print(
        "Confusion Matrix:"
    )


    print(
        cm
    )


    plt.figure(
        figsize=(6, 5)
    )


    plt.imshow(
        cm,
        interpolation="nearest"
    )


    plt.title(
        f"{model_name} - Confusion Matrix"
    )


    plt.colorbar()


    class_names = [
        "Genuine",
        "Fraudulent"
    ]


    tick_marks = np.arange(
        len(class_names)
    )


    plt.xticks(
        tick_marks,
        class_names
    )


    plt.yticks(
        tick_marks,
        class_names
    )


    threshold = cm.max() / 2.0


    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            plt.text(

                j,
                i,

                str(cm[i, j]),

                horizontalalignment="center",

                color="white"
                if cm[i, j] > threshold
                else "black"

            )


    plt.xlabel(
        "Predicted Label"
    )


    plt.ylabel(
        "Actual Label"
    )


    plt.tight_layout()


    plt.savefig(

        output_file,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    print(
        "PASS: Confusion matrix saved."
    )


    print(
        "File:",
        output_file
    )


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

rf_cm_file = os.path.join(

    OUTPUT_DIR,

    "random_forest_confusion_matrix.png"

)


save_confusion_matrix(

    "Random Forest",

    y_test,

    rf_predictions,

    rf_cm_file

)


# ------------------------------------------------------------
# SVM
# ------------------------------------------------------------

svm_cm_file = os.path.join(

    OUTPUT_DIR,

    "svm_confusion_matrix.png"

)


save_confusion_matrix(

    "SVM",

    y_test,

    svm_predictions,

    svm_cm_file

)


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

xgb_cm_file = os.path.join(

    OUTPUT_DIR,

    "xgboost_confusion_matrix.png"

)


save_confusion_matrix(

    "XGBoost",

    y_test,

    xgb_predictions,

    xgb_cm_file

)


# ============================================================
# 17. CLASSIFICATION REPORTS
# ============================================================

print("\n" + "=" * 60)
print("[17] CLASSIFICATION REPORTS")
print("=" * 60)


print(
    "\nRandom Forest"
)

print(
    "-" * 40
)


print(
    classification_report(

        y_test,

        rf_predictions,

        target_names=[
            "Genuine",
            "Fraudulent"
        ],

        zero_division=0

    )
)


print(
    "\nSVM"
)

print(
    "-" * 40
)


print(
    classification_report(

        y_test,

        svm_predictions,

        target_names=[
            "Genuine",
            "Fraudulent"
        ],

        zero_division=0

    )
)


print(
    "\nXGBoost"
)

print(
    "-" * 40
)


print(
    classification_report(

        y_test,

        xgb_predictions,

        target_names=[
            "Genuine",
            "Fraudulent"
        ],

        zero_division=0

    )
)


# ============================================================
# 18. CREATE PREDICTION DATAFRAME
# ============================================================

print("\n" + "=" * 60)
print("[18] CREATING PREDICTION TABLE")
print("=" * 60)


prediction_df = pd.DataFrame({

    "actual":

        np.asarray(y_test),

    "random_forest":

        np.asarray(rf_predictions),

    "svm":

        np.asarray(svm_predictions),

    "xgboost":

        np.asarray(xgb_predictions)

})


print(
    "\nFirst 10 predictions:"
)


print(
    prediction_df.head(10).to_string(
        index=False
    )
)


print(
    "\nPASS: Prediction table created."
)


# ============================================================
# 19. SAVE RESULTS USING JOBLIB
# ============================================================

print("\n" + "=" * 60)
print("[19] SAVING TEST PREDICTION RESULTS")
print("=" * 60)


prediction_data = {

    "y_test":
        np.asarray(y_test),

    "random_forest_predictions":
        np.asarray(rf_predictions),

    "svm_predictions":
        np.asarray(svm_predictions),

    "xgboost_predictions":
        np.asarray(xgb_predictions),

    "results":
        results_df,

    "prediction_table":
        prediction_df

}


joblib.dump(

    prediction_data,

    RESULT_FILE

)


print(
    "PASS: Test prediction results saved."
)


print(
    "File:",
    RESULT_FILE
)


# ============================================================
# 20. SAVE RESULTS CSV
# ============================================================

CSV_RESULT_FILE = os.path.join(

    OUTPUT_DIR,

    "test_prediction_results.csv"

)


results_df.to_csv(

    CSV_RESULT_FILE,

    index=False

)


print(
    "PASS: Model results CSV saved."
)


print(
    "File:",
    CSV_RESULT_FILE
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TEST PREDICTION SUMMARY")
print("=" * 60)


print(
    "Test samples:",
    len(y_test)
)


print(
    "Number of features:",
    X_test.shape[1]
)


print(
    "\nTest data:"
)


print(
    "Genuine jobs    :",
    int(
        (np.asarray(y_test) == 0).sum()
    )
)


print(
    "Fraudulent jobs :",
    int(
        (np.asarray(y_test) == 1).sum()
    )
)


print(
    "\nModels tested:"
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
    "\nGenerated files:"
)


print(
    " - test_predictions.joblib"
)


print(
    " - test_prediction_results.csv"
)


print(
    " - random_forest_confusion_matrix.png"
)


print(
    " - svm_confusion_matrix.png"
)


print(
    " - xgboost_confusion_matrix.png"
)


print(
    "\nOutput directory:"
)


print(
    OUTPUT_DIR
)


print("\n" + "=" * 60)
print("TEST PREDICTION COMPLETED")
print("STATUS: TEST DATA SUCCESSFULLY PASSED THROUGH ALL MODELS")
print("=" * 60)