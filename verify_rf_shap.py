import os
import joblib
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


FEATURE_DATA_FILE = os.path.join(
    CURRENT_DIR,
    "dataset",
    "processed",
    "features",
    "feature_data.pkl"
)


RF_MODEL_FILE = os.path.join(
    CURRENT_DIR,
    "dataset",
    "processed",
    "models",
    "random_forest_model.pkl"
)


SHAP_FILE = os.path.join(
    CURRENT_DIR,
    "dataset",
    "processed",
    "shap",
    "shap_values.pkl"
)


# ============================================================
# HELPER
# ============================================================

def inspect_object(name, obj, level=0):

    indent = "  " * level

    print(
        f"{indent}{name}"
    )

    print(
        f"{indent}Type: {type(obj)}"
    )


    if isinstance(obj, dict):

        print(
            f"{indent}Dictionary keys:"
        )

        for key in obj.keys():

            print(
                f"{indent}  - {key}"
            )


    elif isinstance(obj, list):

        print(
            f"{indent}List length: {len(obj)}"
        )

        for index, item in enumerate(
            obj[:5]
        ):

            print(
                f"{indent}  [{index}] "
                f"type={type(item)}"
            )


    elif isinstance(obj, tuple):

        print(
            f"{indent}Tuple length: {len(obj)}"
        )

        for index, item in enumerate(
            obj[:5]
        ):

            print(
                f"{indent}  [{index}] "
                f"type={type(item)}"
            )


    elif isinstance(obj, np.ndarray):

        print(
            f"{indent}Shape: {obj.shape}"
        )

        print(
            f"{indent}Dtype: {obj.dtype}"
        )


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("RANDOM FOREST SHAP DIAGNOSTIC")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print()
print("[1] CHECKING REQUIRED FILES")
print("-" * 70)


required_files = [

    FEATURE_DATA_FILE,
    RF_MODEL_FILE,
    SHAP_FILE

]


for file_path in required_files:

    print(
        file_path
    )

    if not os.path.exists(
        file_path
    ):

        print(
            "STATUS: MISSING"
        )

        raise FileNotFoundError(
            file_path
        )

    print(
        "STATUS: EXISTS"
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


inspect_object(
    "feature_data",
    feature_data
)


if not isinstance(
    feature_data,
    dict
):

    raise ValueError(
        "feature_data.pkl does not contain a dictionary."
    )


X_test = feature_data.get(
    "X_test"
)


y_test = feature_data.get(
    "y_test"
)


tfidf = feature_data.get(
    "tfidf"
)


print()
print(
    "X_test type:",
    type(X_test)
)


print(
    "X_test shape:",
    X_test.shape
)


print(
    "X_test dtype:",
    getattr(
        X_test,
        "dtype",
        "N/A"
    )
)


print(
    "y_test shape:",
    np.shape(y_test)
)


# ============================================================
# CHECK X_TEST VALUES
# ============================================================

print()
print("[3] CHECKING TEST FEATURE MATRIX")
print("-" * 70)


if hasattr(
    X_test,
    "data"
):

    matrix_data = X_test.data

    print(
        "Sparse matrix detected."
    )

    print(
        "Stored non-zero values:",
        len(matrix_data)
    )

else:

    matrix_data = np.asarray(
        X_test
    ).ravel()

    print(
        "Dense matrix detected."
    )


print(
    "Minimum feature value:",
    np.min(matrix_data)
)


print(
    "Maximum feature value:",
    np.max(matrix_data)
)


print(
    "Mean feature value:",
    np.mean(matrix_data)
)


print(
    "Contains NaN:",
    np.isnan(matrix_data).any()
)


print(
    "Contains +Inf:",
    np.isposinf(matrix_data).any()
)


print(
    "Contains -Inf:",
    np.isneginf(matrix_data).any()
)


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

print()
print("[4] LOADING RANDOM FOREST")
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
    getattr(
        rf_model,
        "n_estimators",
        "N/A"
    )
)


print(
    "Number of input features:",
    getattr(
        rf_model,
        "n_features_in_",
        "N/A"
    )
)


# ============================================================
# RANDOM FOREST TREE VALUE CHECK
# ============================================================

print()
print("[5] CHECKING RANDOM FOREST TREE VALUES")
print("-" * 70)


all_tree_values = []


for tree_index, estimator in enumerate(
    rf_model.estimators_
):

    tree_value = estimator.tree_.value

    tree_min = np.min(
        tree_value
    )

    tree_max = np.max(
        tree_value
    )

    all_tree_values.append(
        tree_value
    )


    if tree_index < 5:

        print(
            f"Tree {tree_index + 1}:"
        )

        print(
            "  value shape:",
            tree_value.shape
        )

        print(
            "  minimum:",
            tree_min
        )

        print(
            "  maximum:",
            tree_max
        )


combined_tree_values = np.concatenate(

    [
        values.reshape(-1)
        for values in all_tree_values
    ]

)


print()
print(
    "Overall tree-value minimum:",
    np.min(
        combined_tree_values
    )
)


print(
    "Overall tree-value maximum:",
    np.max(
        combined_tree_values
    )
)


print(
    "Tree values contain NaN:",
    np.isnan(
        combined_tree_values
    ).any()
)


print(
    "Tree values contain Inf:",
    np.isinf(
        combined_tree_values
    ).any()
)


# ============================================================
# PREDICTION SANITY CHECK
# ============================================================

print()
print("[6] RANDOM FOREST PREDICTION SANITY CHECK")
print("-" * 70)


sample_count = min(
    10,
    X_test.shape[0]
)


X_sample = X_test[
    :sample_count
]


predictions = rf_model.predict(
    X_sample
)


print(
    "Sample size:",
    sample_count
)


print(
    "Predictions:",
    predictions
)


if hasattr(
    rf_model,
    "predict_proba"
):

    probabilities = (
        rf_model.predict_proba(
            X_sample
        )
    )


    print(
        "Prediction probabilities:"
    )

    print(
        probabilities
    )


# ============================================================
# LOAD SHAP ARTIFACT
# ============================================================

print()
print("[7] LOADING SHAP ARTIFACT")
print("-" * 70)


shap_data = joblib.load(
    SHAP_FILE
)


inspect_object(
    "shap_data",
    shap_data
)


# ============================================================
# RECURSIVE SHAP INSPECTION
# ============================================================

print()
print("[8] INSPECTING SHAP VALUES")
print("-" * 70)


def inspect_shap_array(
    name,
    array
):

    print()
    print(
        f"SHAP ARRAY: {name}"
    )

    print(
        "Type:",
        type(array)
    )


    if hasattr(
        array,
        "shape"
    ):

        print(
            "Shape:",
            array.shape
        )


    if isinstance(
        array,
        np.ndarray
    ):

        print(
            "Dtype:",
            array.dtype
        )


        if array.size > 0:

            print(
                "Minimum:",
                np.min(array)
            )

            print(
                "Maximum:",
                np.max(array)
            )

            print(
                "Mean:",
                np.mean(array)
            )

            print(
                "Mean absolute value:",
                np.mean(
                    np.abs(array)
                )
            )

            print(
                "Maximum absolute value:",
                np.max(
                    np.abs(array)
                )
            )

            print(
                "NaN present:",
                np.isnan(array).any()
            )

            print(
                "Inf present:",
                np.isinf(array).any()
            )


def recursive_inspect(
    name,
    obj,
    depth=0
):

    if depth > 3:

        return


    if isinstance(
        obj,
        np.ndarray
    ):

        inspect_shap_array(
            name,
            obj
        )

        return


    if isinstance(
        obj,
        dict
    ):

        for key, value in obj.items():

            recursive_inspect(
                f"{name}[{key!r}]",
                value,
                depth + 1
            )

        return


    if isinstance(
        obj,
        list
    ):

        print()
        print(
            f"{name}: list length={len(obj)}"
        )


        for index, value in enumerate(
            obj[:5]
        ):

            recursive_inspect(
                f"{name}[{index}]",
                value,
                depth + 1
            )

        return


    if isinstance(
        obj,
        tuple
    ):

        print()
        print(
            f"{name}: tuple length={len(obj)}"
        )


        for index, value in enumerate(
            obj[:5]
        ):

            recursive_inspect(
                f"{name}[{index}]",
                value,
                depth + 1
            )

        return


    print()
    print(
        f"{name}: {type(obj)}"
    )


recursive_inspect(
    "shap_data",
    shap_data
)


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 70)
print("RANDOM FOREST SHAP DIAGNOSTIC COMPLETED")
print("=" * 70)

print()
print(
    "Do NOT modify shap_analysis.py yet."
)

print(
    "Paste the complete output from this diagnostic."
)

print(
    "=" * 70
)