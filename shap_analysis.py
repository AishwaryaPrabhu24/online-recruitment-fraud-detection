import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = "dataset/processed/features/feature_data.pkl"

MODEL_DIR = "dataset/processed/models"

OUTPUT_DIR = "dataset/processed/shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_SIZE = 200

# Background size used for interventional SHAP
BACKGROUND_SIZE = 50

RANDOM_STATE = 42


print("=" * 60)
print("SHAP ANALYSIS")
print("=" * 60)


# ============================================================
# 1. CHECK REQUIRED FILES
# ============================================================

print("\n[1] Checking required files...")


if not os.path.exists(FEATURE_FILE):

    raise FileNotFoundError(
        f"Feature file not found:\n{FEATURE_FILE}"
    )


model_files = {

    "Random Forest":
        os.path.join(
            MODEL_DIR,
            "random_forest_model.pkl"
        ),

    "SVM":
        os.path.join(
            MODEL_DIR,
            "linear_svm_model.pkl"
        ),

    "XGBoost":
        os.path.join(
            MODEL_DIR,
            "xgboost_model.pkl"
        )
}


for model_name, model_path in model_files.items():

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"{model_name} model not found:\n{model_path}"
        )


print("PASS: All required files found.")


# ============================================================
# 2. LOAD FEATURE DATA
# ============================================================

print("\n[2] Loading feature-engineering data...")

print("Using Joblib loader...")


try:

    feature_data = joblib.load(
        FEATURE_FILE
    )

except Exception as error:

    raise RuntimeError(
        "\nUnable to load feature_data.pkl.\n"
        f"Original error: {error}"
    )


print("PASS: Feature data loaded successfully.")


# ============================================================
# 3. INSPECT FEATURE DATA
# ============================================================

print("\n[3] Inspecting feature data...")


if not isinstance(feature_data, dict):

    raise TypeError(
        "feature_data.pkl does not contain a dictionary."
    )


print("Available keys:")


for key in feature_data.keys():

    print(" -", key)


# ============================================================
# 4. LOAD TEST DATA
# ============================================================

print("\n[4] Loading test data...")


# ------------------------------------------------------------
# Find X_test
# ------------------------------------------------------------

possible_x_test_keys = [

    "X_test",
    "X_test_combined",
    "X_test_features",
    "X_test_final"

]


X_test = None

x_test_key_used = None


for key in possible_x_test_keys:

    if key in feature_data:

        X_test = feature_data[key]

        x_test_key_used = key

        break


if X_test is None:

    raise KeyError(
        "Could not find X_test in feature_data.pkl."
    )


# ------------------------------------------------------------
# Find y_test
# ------------------------------------------------------------

possible_y_test_keys = [

    "y_test",
    "y_test_final",
    "test_target",
    "y_test_labels"

]


y_test = None

y_test_key_used = None


for key in possible_y_test_keys:

    if key in feature_data:

        y_test = feature_data[key]

        y_test_key_used = key

        break


if y_test is None:

    raise KeyError(
        "Could not find y_test in feature_data.pkl."
    )


print(
    "X_test loaded from:",
    x_test_key_used
)

print(
    "y_test loaded from:",
    y_test_key_used
)

print(
    "Test feature shape:",
    X_test.shape
)

print(
    "Test target shape:",
    np.asarray(y_test).shape
)


# ============================================================
# 5. CREATE SHAP SAMPLE
# ============================================================

print("\n[5] Creating SHAP analysis sample...")


sample_size = min(
    SAMPLE_SIZE,
    X_test.shape[0]
)


rng = np.random.RandomState(
    RANDOM_STATE
)


sample_indices = rng.choice(

    X_test.shape[0],

    size=sample_size,

    replace=False

)


X_sample = X_test[
    sample_indices
]


if hasattr(
    y_test,
    "iloc"
):

    y_sample = y_test.iloc[
        sample_indices
    ]

else:

    y_sample = np.asarray(
        y_test
    )[sample_indices]


print(
    "SHAP sample size:",
    sample_size
)


# ============================================================
# 6. PREPARE ACTUAL FEATURE NAMES
# ============================================================

print("\n[6] Preparing feature names...")


feature_names = None


# ------------------------------------------------------------
# First check if feature names were saved
# ------------------------------------------------------------

possible_feature_name_keys = [

    "feature_names",

    "feature_names_combined",

    "combined_feature_names"

]


for key in possible_feature_name_keys:

    if key in feature_data:

        feature_names = list(
            feature_data[key]
        )

        print(
            "Feature names loaded from:",
            key
        )

        break


# ------------------------------------------------------------
# Reconstruct feature names if not saved
# ------------------------------------------------------------

if feature_names is None:

    print(
        "Feature names were not explicitly saved."
    )

    print(
        "Reconstructing feature names from TF-IDF "
        "and red-flag features..."
    )


    feature_names = []


    # --------------------------------------------------------
    # TF-IDF feature names
    # --------------------------------------------------------

    if "tfidf" in feature_data:

        tfidf = feature_data["tfidf"]


        try:

            tfidf_names = list(
                tfidf.get_feature_names_out()
            )

            feature_names.extend(
                tfidf_names
            )

            print(
                "TF-IDF feature names:",
                len(tfidf_names)
            )

        except Exception:

            print(
                "WARNING: Could not extract TF-IDF "
                "feature names."
            )


    # --------------------------------------------------------
    # Red-flag feature names
    # --------------------------------------------------------

    if "red_flag_columns" in feature_data:

        red_flag_names = list(
            feature_data["red_flag_columns"]
        )

        feature_names.extend(
            red_flag_names
        )

        print(
            "Red-flag feature names:",
            len(red_flag_names)
        )


# ------------------------------------------------------------
# Verify feature-name count
# ------------------------------------------------------------

if len(feature_names) != X_test.shape[1]:

    print(
        "\nWARNING: Reconstructed feature-name count "
        "does not match feature count."
    )

    print(
        "Expected:",
        X_test.shape[1]
    )

    print(
        "Found:",
        len(feature_names)
    )


    print(
        "Using generic feature names as fallback."
    )


    feature_names = [

        f"feature_{i}"

        for i in range(
            X_test.shape[1]
        )

    ]


print(
    "Total feature names:",
    len(feature_names)
)


# ============================================================
# 7. LOAD TRAINED MODELS
# ============================================================

print("\n[7] Loading trained models...")


models = {}


for model_name, model_path in model_files.items():

    try:

        models[model_name] = joblib.load(
            model_path
        )

        print(
            f"PASS: {model_name} loaded."
        )

    except Exception as error:

        raise RuntimeError(
            f"\nUnable to load {model_name} model.\n"
            f"File: {model_path}\n"
            f"Error: {error}"
        )


# ============================================================
# 8. VERIFY MODEL FEATURE COUNTS
# ============================================================

print("\n[8] Checking model feature counts...")


for model_name, model in models.items():

    if hasattr(
        model,
        "n_features_in_"
    ):

        model_features = model.n_features_in_

        print(
            f"{model_name} model features:",
            model_features
        )

        if model_features != X_test.shape[1]:

            raise ValueError(

                f"{model_name} expects "
                f"{model_features} features, "
                f"but X_test contains "
                f"{X_test.shape[1]} features."

            )


print(
    "PASS: Model and test feature counts match."
)


# ============================================================
# 9. PREPARE SHAP DATA
# ============================================================

print("\n[9] Preparing SHAP sample...")


def prepare_dense_data(X):

    """
    Convert sparse data to dense format.

    Only the small SHAP sample/background are converted,
    so the complete test matrix is not unnecessarily
    converted to dense format.
    """

    if hasattr(
        X,
        "toarray"
    ):

        return X.toarray()

    return np.asarray(X)


X_sample_dense = prepare_dense_data(
    X_sample
)


print(
    "Dense SHAP sample shape:",
    X_sample_dense.shape
)


# ============================================================
# 10. CREATE RANDOM FOREST SHAP BACKGROUND
# ============================================================

print(
    "\n[10] Preparing Random Forest SHAP background..."
)


background_size = min(

    BACKGROUND_SIZE,

    X_test.shape[0]

)


background_indices = rng.choice(

    X_test.shape[0],

    size=background_size,

    replace=False

)


X_background = X_test[
    background_indices
]


X_background_dense = prepare_dense_data(
    X_background
)


print(
    "Background size:",
    background_size
)

print(
    "Background shape:",
    X_background_dense.shape
)


# ============================================================
# 11. RANDOM FOREST SHAP
# ============================================================

print("\n" + "=" * 60)

print(
    "[11] RANDOM FOREST SHAP ANALYSIS"
)

print("=" * 60)


rf_model = models[
    "Random Forest"
]


print(
    "Creating interventional TreeExplainer..."
)


# ------------------------------------------------------------
# IMPORTANT:
#
# The diagnostic verification showed that:
#
# Default TreeExplainer              -> FAILED
# Tree path dependent                -> FAILED
# Default + check_additivity=False   -> FAILED
# Interventional + background        -> PASSED
#
# Therefore the Random Forest uses:
#
# feature_perturbation="interventional"
#
# with a background dataset.
#
# We intentionally DO NOT use:
#
# check_additivity=False
#
# because the diagnostic demonstrated extremely large
# additivity errors when that option was used.
# ------------------------------------------------------------

rf_explainer = shap.TreeExplainer(

    rf_model,

    data=X_background_dense,

    feature_perturbation="interventional"

)


print(
    "Explainer created successfully."
)


print(
    "Expected value:",
    rf_explainer.expected_value
)


print(
    "Calculating SHAP values..."
)


# ------------------------------------------------------------
# Calculate SHAP values WITHOUT disabling additivity checking
# ------------------------------------------------------------

rf_shap_values = rf_explainer.shap_values(

    X_sample_dense

)


# ------------------------------------------------------------
# Handle different SHAP output formats
# ------------------------------------------------------------

if isinstance(
    rf_shap_values,
    list
):

    if len(
        rf_shap_values
    ) == 2:

        rf_values = rf_shap_values[1]

    else:

        rf_values = rf_shap_values[0]

else:

    rf_values = rf_shap_values


    if rf_values.ndim == 3:

        rf_values = rf_values[:, :, 1]


rf_values = np.asarray(
    rf_values
)


print(
    "SHAP value shape:",
    rf_values.shape
)


# ------------------------------------------------------------
# Validate Random Forest SHAP values
# ------------------------------------------------------------

if np.isnan(
    rf_values
).any():

    raise ValueError(
        "Random Forest SHAP values contain NaN."
    )


if np.isinf(
    rf_values
).any():

    raise ValueError(
        "Random Forest SHAP values contain Inf."
    )


print(
    "SHAP minimum:",
    np.min(rf_values)
)

print(
    "SHAP maximum:",
    np.max(rf_values)
)

print(
    "Mean absolute SHAP:",
    np.mean(
        np.abs(rf_values)
    )
)


# ============================================================
# RANDOM FOREST PLOT
# ============================================================

print(
    "Creating Random Forest SHAP plot..."
)


plt.figure(
    figsize=(10, 8)
)


shap.summary_plot(

    rf_values,

    X_sample_dense,

    feature_names=feature_names,

    max_display=20,

    show=False

)


plt.title(
    "Random Forest - SHAP Feature Importance"
)


plt.tight_layout()


rf_plot_path = os.path.join(

    OUTPUT_DIR,

    "random_forest_shap_summary.png"

)


plt.savefig(

    rf_plot_path,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "PASS: Random Forest SHAP plot saved."
)

print(
    rf_plot_path
)


# ============================================================
# 12. XGBOOST SHAP
# ============================================================

print("\n" + "=" * 60)

print(
    "[12] XGBOOST SHAP ANALYSIS"
)

print("=" * 60)


xgb_model = models[
    "XGBoost"
]


print(
    "Creating TreeExplainer..."
)


xgb_explainer = shap.TreeExplainer(

    xgb_model,

    feature_perturbation="interventional"

)


print(
    "Calculating SHAP values..."
)


xgb_shap_values = xgb_explainer.shap_values(

    X_sample_dense,

    check_additivity=False

)


# ------------------------------------------------------------
# Handle SHAP output formats
# ------------------------------------------------------------

if isinstance(
    xgb_shap_values,
    list
):

    if len(
        xgb_shap_values
    ) == 2:

        xgb_values = xgb_shap_values[1]

    else:

        xgb_values = xgb_shap_values[0]

else:

    xgb_values = xgb_shap_values


    if xgb_values.ndim == 3:

        xgb_values = xgb_values[:, :, 1]


xgb_values = np.asarray(
    xgb_values
)


print(
    "SHAP value shape:",
    xgb_values.shape
)


# ------------------------------------------------------------
# Validate XGBoost SHAP values
# ------------------------------------------------------------

if np.isnan(
    xgb_values
).any():

    raise ValueError(
        "XGBoost SHAP values contain NaN."
    )


if np.isinf(
    xgb_values
).any():

    raise ValueError(
        "XGBoost SHAP values contain Inf."
    )


print(
    "SHAP minimum:",
    np.min(xgb_values)
)

print(
    "SHAP maximum:",
    np.max(xgb_values)
)

print(
    "Mean absolute SHAP:",
    np.mean(
        np.abs(xgb_values)
    )
)


# ============================================================
# XGBOOST PLOT
# ============================================================

print(
    "Creating XGBoost SHAP plot..."
)


plt.figure(
    figsize=(10, 8)
)


shap.summary_plot(

    xgb_values,

    X_sample_dense,

    feature_names=feature_names,

    max_display=20,

    show=False

)


plt.title(
    "XGBoost - SHAP Feature Importance"
)


plt.tight_layout()


xgb_plot_path = os.path.join(

    OUTPUT_DIR,

    "xgboost_shap_summary.png"

)


plt.savefig(

    xgb_plot_path,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "PASS: XGBoost SHAP plot saved."
)

print(
    xgb_plot_path
)


# ============================================================
# 13. SVM SHAP
# ============================================================

print("\n" + "=" * 60)

print(
    "[13] SVM SHAP ANALYSIS"
)

print("=" * 60)


svm_model = models[
    "SVM"
]


print(
    "Creating LinearExplainer..."
)


svm_background_size = min(

    BACKGROUND_SIZE,

    X_test.shape[0]

)


svm_background_indices = rng.choice(

    X_test.shape[0],

    size=svm_background_size,

    replace=False

)


X_svm_background = X_test[
    svm_background_indices
]


svm_explainer = shap.LinearExplainer(

    svm_model,

    X_svm_background

)


print(
    "Calculating SHAP values..."
)


svm_shap_values = svm_explainer.shap_values(

    X_sample

)


if isinstance(

    svm_shap_values,

    list

):

    svm_values = svm_shap_values[0]

else:

    svm_values = svm_shap_values


    if svm_values.ndim == 3:

        svm_values = svm_values[:, :, 0]


if hasattr(

    svm_values,

    "toarray"

):

    svm_values_plot = svm_values.toarray()

else:

    svm_values_plot = np.asarray(
        svm_values
    )


print(
    "SHAP value shape:",
    svm_values_plot.shape
)


# ------------------------------------------------------------
# Validate SVM SHAP values
# ------------------------------------------------------------

if np.isnan(
    svm_values_plot
).any():

    raise ValueError(
        "SVM SHAP values contain NaN."
    )


if np.isinf(
    svm_values_plot
).any():

    raise ValueError(
        "SVM SHAP values contain Inf."
    )


print(
    "SHAP minimum:",
    np.min(svm_values_plot)
)

print(
    "SHAP maximum:",
    np.max(svm_values_plot)
)

print(
    "Mean absolute SHAP:",
    np.mean(
        np.abs(svm_values_plot)
    )
)


# ============================================================
# SVM PLOT
# ============================================================

print(
    "Creating SVM SHAP plot..."
)


plt.figure(
    figsize=(10, 8)
)


shap.summary_plot(

    svm_values_plot,

    X_sample_dense,

    feature_names=feature_names,

    max_display=20,

    show=False

)


plt.title(
    "SVM - SHAP Feature Importance"
)


plt.tight_layout()


svm_plot_path = os.path.join(

    OUTPUT_DIR,

    "svm_shap_summary.png"

)


plt.savefig(

    svm_plot_path,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "PASS: SVM SHAP plot saved."
)

print(
    svm_plot_path
)


# ============================================================
# 14. SAVE SHAP VALUES
# ============================================================

print("\n" + "=" * 60)

print(
    "[14] SAVING SHAP RESULTS"
)

print("=" * 60)


shap_results = {

    "feature_names":
        feature_names,

    "sample_indices":
        sample_indices,

    "y_sample":
        np.asarray(y_sample),

    "random_forest_shap_values":
        rf_values,

    "xgboost_shap_values":
        xgb_values,

    "svm_shap_values":
        svm_values_plot

}


shap_file = os.path.join(

    OUTPUT_DIR,

    "shap_values.pkl"

)


joblib.dump(

    shap_results,

    shap_file

)


print(
    "PASS: SHAP values saved."
)

print(
    "File:",
    shap_file
)


# ============================================================
# 15. CALCULATE TOP FEATURES
# ============================================================

print("\n" + "=" * 60)

print(
    "[15] TOP SHAP FEATURES"
)

print("=" * 60)


def get_top_features(

    shap_values,

    feature_names,

    top_n=20

):

    importance = np.mean(

        np.abs(shap_values),

        axis=0

    )


    result = pd.DataFrame({

        "feature":
            feature_names,

        "mean_abs_shap":
            importance

    })


    result = result.sort_values(

        "mean_abs_shap",

        ascending=False

    )


    return result.head(
        top_n
    )


rf_top = get_top_features(

    rf_values,

    feature_names

)


xgb_top = get_top_features(

    xgb_values,

    feature_names

)


svm_top = get_top_features(

    svm_values_plot,

    feature_names

)


print(
    "\nRandom Forest:"
)

print(
    rf_top.to_string(
        index=False
    )
)


print(
    "\nXGBoost:"
)

print(
    xgb_top.to_string(
        index=False
    )
)


print(
    "\nSVM:"
)

print(
    svm_top.to_string(
        index=False
    )
)


# ============================================================
# 16. SAVE TOP FEATURE TABLES
# ============================================================

rf_top.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "random_forest_top_features.csv"

    ),

    index=False

)


xgb_top.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "xgboost_top_features.csv"

    ),

    index=False

)


svm_top.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "svm_top_features.csv"

    ),

    index=False

)


print(
    "\nPASS: Top feature tables saved."
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print(
    "SHAP ANALYSIS SUMMARY"
)

print("=" * 60)


print(
    "Test samples available :",
    X_test.shape[0]
)


print(
    "SHAP samples analyzed  :",
    sample_size
)


print(
    "Total features         :",
    X_test.shape[1]
)


print(
    "Random Forest background:",
    background_size
)


print(
    "\nModels analyzed:"
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
    " - random_forest_shap_summary.png"
)

print(
    " - xgboost_shap_summary.png"
)

print(
    " - svm_shap_summary.png"
)

print(
    " - shap_values.pkl"
)

print(
    " - random_forest_top_features.csv"
)

print(
    " - xgboost_top_features.csv"
)

print(
    " - svm_top_features.csv"
)


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print("\n" + "=" * 60)

print(
    "SHAP ANALYSIS COMPLETED"
)

print(
    "STATUS: READY FOR MODEL EVALUATION"
)

print("=" * 60)