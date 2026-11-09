import os
import numpy as np
import joblib
import shap
import sklearn


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


FEATURE_DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "features",
    "feature_data.pkl"
)


RF_MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "models",
    "random_forest_model.pkl"
)


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("RANDOM FOREST SHAP VALUE VERIFICATION")
print("=" * 70)


# ============================================================
# VERSION INFORMATION
# ============================================================

print()
print("[1] LIBRARY VERSIONS")
print("-" * 70)

print(
    "Python:",
    __import__("sys").version
)

print(
    "NumPy:",
    np.__version__
)

print(
    "scikit-learn:",
    sklearn.__version__
)

print(
    "SHAP:",
    shap.__version__
)


# ============================================================
# LOAD FEATURE DATA
# ============================================================

print()
print("[2] LOADING FEATURE DATA")
print("-" * 70)


feature_data = joblib.load(
    FEATURE_DATA_FILE
)


X_test = feature_data[
    "X_test"
]


y_test = feature_data[
    "y_test"
]


print(
    "X_test shape:",
    X_test.shape
)


print(
    "X_test type:",
    type(X_test)
)


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

print()
print("[3] LOADING RANDOM FOREST")
print("-" * 70)


rf_model = joblib.load(
    RF_MODEL_FILE
)


print(
    "Model type:",
    type(rf_model)
)


print(
    "Number of estimators:",
    rf_model.n_estimators
)


print(
    "Number of features:",
    rf_model.n_features_in_
)


# ============================================================
# SELECT SMALL SAMPLE
# ============================================================

print()
print("[4] PREPARING SMALL SHAP SAMPLE")
print("-" * 70)


sample_size = 5


X_sample_sparse = X_test[
    :sample_size
]


X_sample_dense = X_sample_sparse.toarray()


y_sample = np.asarray(
    y_test[:sample_size]
)


print(
    "Sample shape:",
    X_sample_dense.shape
)


print(
    "Sample labels:",
    y_sample
)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

print()
print("[5] RANDOM FOREST PREDICTIONS")
print("-" * 70)


predictions = rf_model.predict(
    X_sample_sparse
)


probabilities = rf_model.predict_proba(
    X_sample_sparse
)


for i in range(
    sample_size
):

    print(
        f"\nSample {i + 1}"
    )

    print(
        "Actual label:",
        y_sample[i]
    )

    print(
        "Prediction:",
        predictions[i]
    )

    print(
        "Probability genuine:",
        probabilities[i][0]
    )

    print(
        "Probability fraudulent:",
        probabilities[i][1]
    )


# ============================================================
# CREATE TREE EXPLAINER
# ============================================================

print()
print("[6] CREATING TREE EXPLAINER")
print("-" * 70)


explainer = shap.TreeExplainer(
    rf_model
)


print(
    "Explainer created successfully."
)


# ============================================================
# EXPECTED VALUE
# ============================================================

print()
print("[7] SHAP EXPECTED VALUE")
print("-" * 70)


expected_value = explainer.expected_value


print(
    "Expected value type:",
    type(expected_value)
)


print(
    "Expected value:",
    expected_value
)


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

print()
print("[8] CALCULATING SHAP VALUES")
print("-" * 70)


shap_values = explainer.shap_values(
    X_sample_dense
)


print(
    "Returned SHAP type:",
    type(shap_values)
)


if isinstance(
    shap_values,
    list
):

    print(
        "SHAP returned a list."
    )

    print(
        "Number of outputs:",
        len(shap_values)
    )

    for index, value in enumerate(
        shap_values
    ):

        print(
            f"Output {index} shape:",
            np.asarray(value).shape
        )

else:

    print(
        "SHAP returned a NumPy array."
    )

    print(
        "SHAP shape:",
        np.asarray(shap_values).shape
    )


# ============================================================
# NORMALIZE SHAP REPRESENTATION
# ============================================================

print()
print("[9] NORMALIZING SHAP OUTPUT")
print("-" * 70)


if isinstance(
    shap_values,
    list
):

    if len(shap_values) >= 2:

        # Class 1 = fraudulent
        rf_shap = np.asarray(
            shap_values[1]
        )

        print(
            "Using class-1 SHAP values."
        )

    else:

        rf_shap = np.asarray(
            shap_values[0]
        )

        print(
            "Using first SHAP output."
        )

else:

    rf_shap = np.asarray(
        shap_values
    )


print(
    "Normalized SHAP shape:",
    rf_shap.shape
)


# ============================================================
# SHAP STATISTICS
# ============================================================

print()
print("[10] SHAP VALUE STATISTICS")
print("-" * 70)


print(
    "Minimum:",
    np.min(rf_shap)
)


print(
    "Maximum:",
    np.max(rf_shap)
)


print(
    "Mean:",
    np.mean(rf_shap)
)


print(
    "Mean absolute:",
    np.mean(
        np.abs(rf_shap)
    )
)


print(
    "Maximum absolute:",
    np.max(
        np.abs(rf_shap)
    )
)


print(
    "Contains NaN:",
    np.isnan(rf_shap).any()
)


print(
    "Contains Inf:",
    np.isinf(rf_shap).any()
)


# ============================================================
# ADDITIVITY CHECK
# ============================================================

print()
print("[11] SHAP ADDITIVITY CHECK")
print("-" * 70)


print(
    "This checks whether:"
)

print(
    "expected_value + sum(SHAP values)"
)

print(
    "approximately matches the model output."
)


# ------------------------------------------------------------
# Determine base value for class 1
# ------------------------------------------------------------

if isinstance(
    expected_value,
    np.ndarray
):

    expected_array = expected_value.reshape(
        -1
    )

elif isinstance(
    expected_value,
    list
):

    expected_array = np.asarray(
        expected_value
    ).reshape(
        -1
    )

else:

    expected_array = np.asarray(
        [expected_value]
    )


print()
print(
    "Expected value array:",
    expected_array
)


if len(
    expected_array
) >= 2:

    base_value = float(
        expected_array[1]
    )

else:

    base_value = float(
        expected_array[0]
    )


print(
    "Using base value:",
    base_value
)


# ------------------------------------------------------------
# Calculate SHAP reconstructed outputs
# ------------------------------------------------------------

shap_sums = np.sum(
    rf_shap,
    axis=1
)


reconstructed = (
    base_value
    + shap_sums
)


print()
print(
    "Sample | Model probability | SHAP reconstructed | Difference"
)


print(
    "-" * 70
)


for i in range(
    sample_size
):

    model_probability = float(
        probabilities[i][1]
    )


    shap_prediction = float(
        reconstructed[i]
    )


    difference = (
        shap_prediction
        - model_probability
    )


    print(
        f"{i + 1:>6} | "
        f"{model_probability:>17.10f} | "
        f"{shap_prediction:>18.10f} | "
        f"{difference:>12.10f}"
    )


# ============================================================
# FEATURE-LEVEL INSPECTION
# ============================================================

print()
print("[12] LARGEST SHAP VALUES")
print("-" * 70)


feature_names = feature_data.get(
    "combined_feature_names"
)


if feature_names is None:

    print(
        "Feature names not available."
    )

else:

    feature_names = list(
        feature_names
    )


    mean_abs_shap = np.mean(
        np.abs(rf_shap),
        axis=0
    )


    top_indices = np.argsort(
        mean_abs_shap
    )[::-1][:20]


    print(
        "Top 20 features:"
    )


    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        print(
            f"{rank:>2}. "
            f"{feature_names[index]:<40} "
            f"{mean_abs_shap[index]:.10e}"
        )


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 70)
print("RANDOM FOREST SHAP VALUE VERIFICATION COMPLETED")
print("=" * 70)

print()
print(
    "Do NOT modify shap_analysis.py yet."
)

print(
    "Paste the complete output here."
)

print(
    "=" * 70
)