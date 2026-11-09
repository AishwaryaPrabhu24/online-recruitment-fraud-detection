import os
import joblib
import numpy as np
import shap


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DATA_FILE = (
    "dataset/processed/features/feature_data.pkl"
)

RF_MODEL_FILE = (
    "dataset/processed/models/random_forest_model.pkl"
)


# Number of samples used for this diagnostic
SAMPLE_SIZE = 5

# Number of background samples for interventional SHAP
BACKGROUND_SIZE = 50


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_separator():
    print("-" * 70)


def calculate_additivity_error(
    shap_values,
    expected_value,
    model_output
):
    """
    Calculate the difference between:

        expected_value + sum(SHAP values)

    and:

        model output

    Handles both single-output and multi-output SHAP arrays.
    """

    shap_values = np.asarray(shap_values)

    expected_value = np.asarray(
        expected_value
    )

    model_output = np.asarray(
        model_output
    )

    # --------------------------------------------------------
    # Multi-output case
    #
    # Shape:
    # (samples, features, outputs)
    # --------------------------------------------------------

    if shap_values.ndim == 3:

        errors = []

        for output_index in range(
            shap_values.shape[2]
        ):

            shap_sum = (
                expected_value[output_index]
                +
                shap_values[:, :, output_index].sum(
                    axis=1
                )
            )

            error = (
                shap_sum
                -
                model_output[:, output_index]
            )

            errors.append(error)

        errors = np.column_stack(
            errors
        )

    # --------------------------------------------------------
    # Single-output case
    #
    # Shape:
    # (samples, features)
    # --------------------------------------------------------

    else:

        shap_sum = (
            expected_value
            +
            shap_values.sum(
                axis=1
            )
        )

        errors = (
            shap_sum
            -
            model_output
        )

    absolute_errors = np.abs(
        errors
    )

    return {
        "errors": errors,
        "max_error": float(
            np.max(
                absolute_errors
            )
        ),
        "mean_error": float(
            np.mean(
                absolute_errors
            )
        ),
    }


def inspect_shap_values(
    name,
    shap_values,
    expected_value,
    model_output
):

    print(f"\n{name}")
    print_separator()

    shap_values = np.asarray(
        shap_values
    )

    print(
        "SHAP shape:",
        shap_values.shape
    )

    print(
        "SHAP dtype:",
        shap_values.dtype
    )

    print(
        "SHAP minimum:",
        np.min(shap_values)
    )

    print(
        "SHAP maximum:",
        np.max(shap_values)
    )

    print(
        "SHAP mean absolute value:",
        np.mean(
            np.abs(shap_values)
        )
    )

    print(
        "SHAP contains NaN:",
        np.isnan(shap_values).any()
    )

    print(
        "SHAP contains Inf:",
        np.isinf(shap_values).any()
    )

    print(
        "Expected value:",
        expected_value
    )

    result = calculate_additivity_error(
        shap_values,
        expected_value,
        model_output
    )

    print(
        "Maximum additivity error:",
        result["max_error"]
    )

    print(
        "Mean additivity error:",
        result["mean_error"]
    )

    if result["max_error"] < 1e-5:

        print(
            "ADDitivity STATUS: PASS"
        )

    elif result["max_error"] < 1e-3:

        print(
            "Additivity STATUS: ACCEPTABLE"
        )

    else:

        print(
            "Additivity STATUS: FAIL"
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RANDOM FOREST SHAP METHOD VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LIBRARY VERSIONS
    # --------------------------------------------------------

    print("\n[1] LIBRARY VERSIONS")
    print_separator()

    print(
        "NumPy:",
        np.__version__
    )

    print(
        "SHAP:",
        shap.__version__
    )

    # --------------------------------------------------------
    # 2. CHECK REQUIRED FILES
    # --------------------------------------------------------

    print("\n[2] CHECKING REQUIRED FILES")
    print_separator()

    required_files = [
        FEATURE_DATA_FILE,
        RF_MODEL_FILE,
    ]

    for file_path in required_files:

        if os.path.exists(file_path):

            print(
                "PASS:",
                file_path
            )

        else:

            print(
                "FAIL:",
                file_path
            )

            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # --------------------------------------------------------
    # 3. LOAD FEATURE DATA
    # --------------------------------------------------------

    print("\n[3] LOADING FEATURE DATA")
    print_separator()

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

    # --------------------------------------------------------
    # 4. LOAD RANDOM FOREST
    # --------------------------------------------------------

    print("\n[4] LOADING RANDOM FOREST")
    print_separator()

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

    # --------------------------------------------------------
    # 5. PREPARE SMALL TEST SAMPLE
    # --------------------------------------------------------

    print("\n[5] PREPARING TEST SAMPLE")
    print_separator()

    X_sample = X_test[
        :SAMPLE_SIZE
    ]

    y_sample = np.asarray(
        y_test[
            :SAMPLE_SIZE
        ]
    )

    # Convert only the small diagnostic sample to dense.
    X_sample_dense = X_sample.toarray()

    print(
        "Sample shape:",
        X_sample_dense.shape
    )

    print(
        "Sample labels:",
        y_sample
    )

    # --------------------------------------------------------
    # 6. RANDOM FOREST PREDICTIONS
    # --------------------------------------------------------

    print("\n[6] RANDOM FOREST PREDICTIONS")
    print_separator()

    predictions = rf_model.predict(
        X_sample_dense
    )

    probabilities = rf_model.predict_proba(
        X_sample_dense
    )

    for index in range(
        SAMPLE_SIZE
    ):

        print(
            f"\nSample {index + 1}"
        )

        print(
            "Actual label:",
            int(y_sample[index])
        )

        print(
            "Prediction:",
            int(predictions[index])
        )

        print(
            "Probability genuine:",
            probabilities[index][0]
        )

        print(
            "Probability fraudulent:",
            probabilities[index][1]
        )

    # ========================================================
    # METHOD 1
    # Default TreeExplainer
    # ========================================================

    print("\n" + "=" * 70)
    print("[7] METHOD 1 — DEFAULT TREEEXPLAINER")
    print("=" * 70)

    method1_result = None

    try:

        explainer_default = shap.TreeExplainer(
            rf_model
        )

        print(
            "Explainer created successfully."
        )

        print(
            "Expected value:",
            explainer_default.expected_value
        )

        print(
            "Calculating SHAP values..."
        )

        shap_default = (
            explainer_default.shap_values(
                X_sample_dense
            )
        )

        method1_result = inspect_shap_values(
            "DEFAULT TREEEXPLAINER RESULT",
            shap_default,
            explainer_default.expected_value,
            probabilities
        )

    except Exception as error:

        print(
            "\nDEFAULT METHOD ERROR:"
        )

        print(
            type(error).__name__,
            ":",
            error
        )

    # ========================================================
    # METHOD 2
    # Tree Path Dependent
    # ========================================================

    print("\n" + "=" * 70)
    print("[8] METHOD 2 — TREE PATH DEPENDENT")
    print("=" * 70)

    method2_result = None

    try:

        explainer_path = shap.TreeExplainer(
            rf_model,
            feature_perturbation="tree_path_dependent"
        )

        print(
            "Explainer created successfully."
        )

        print(
            "Expected value:",
            explainer_path.expected_value
        )

        print(
            "Calculating SHAP values..."
        )

        shap_path = (
            explainer_path.shap_values(
                X_sample_dense
            )
        )

        method2_result = inspect_shap_values(
            "TREE PATH DEPENDENT RESULT",
            shap_path,
            explainer_path.expected_value,
            probabilities
        )

    except Exception as error:

        print(
            "\nTREE PATH DEPENDENT ERROR:"
        )

        print(
            type(error).__name__,
            ":",
            error
        )

    # ========================================================
    # METHOD 3
    # INTERVENTIONAL WITH BACKGROUND DATA
    # ========================================================

    print("\n" + "=" * 70)
    print("[9] METHOD 3 — INTERVENTIONAL SHAP")
    print("=" * 70)

    method3_result = None

    try:

        # ----------------------------------------------------
        # Create background dataset.
        #
        # Use the first BACKGROUND_SIZE test rows only for
        # this diagnostic. This does NOT modify training,
        # testing, or model artifacts.
        # ----------------------------------------------------

        background = X_test[
            :BACKGROUND_SIZE
        ].toarray()

        print(
            "Background shape:",
            background.shape
        )

        explainer_interventional = (
            shap.TreeExplainer(
                rf_model,
                data=background,
                feature_perturbation="interventional"
            )
        )

        print(
            "Explainer created successfully."
        )

        print(
            "Expected value:",
            explainer_interventional.expected_value
        )

        print(
            "Calculating SHAP values..."
        )

        shap_interventional = (
            explainer_interventional.shap_values(
                X_sample_dense
            )
        )

        method3_result = inspect_shap_values(
            "INTERVENTIONAL RESULT",
            shap_interventional,
            explainer_interventional.expected_value,
            probabilities
        )

    except Exception as error:

        print(
            "\nINTERVENTIONAL METHOD ERROR:"
        )

        print(
            type(error).__name__,
            ":",
            error
        )

    # ========================================================
    # METHOD 4
    # DEFAULT EXPLAINER WITH ADDITIVITY CHECK DISABLED
    # ========================================================

    print("\n" + "=" * 70)
    print("[10] METHOD 4 — DEFAULT + check_additivity=False")
    print("=" * 70)

    method4_result = None

    try:

        explainer_no_check = shap.TreeExplainer(
            rf_model
        )

        print(
            "Calculating SHAP values with "
            "additivity check disabled..."
        )

        shap_no_check = (
            explainer_no_check.shap_values(
                X_sample_dense,
                check_additivity=False
            )
        )

        method4_result = inspect_shap_values(
            "CHECK-DISABLED RESULT",
            shap_no_check,
            explainer_no_check.expected_value,
            probabilities
        )

    except Exception as error:

        print(
            "\nCHECK-DISABLED METHOD ERROR:"
        )

        print(
            type(error).__name__,
            ":",
            error
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 70)
    print("[11] METHOD COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Method':<40}"
        f"{'Max Error':>20}"
        f"{'Mean Error':>20}"
    )

    print(
        "-" * 80
    )

    results = [
        (
            "Default TreeExplainer",
            method1_result
        ),
        (
            "Tree Path Dependent",
            method2_result
        ),
        (
            "Interventional",
            method3_result
        ),
        (
            "Default + check_additivity=False",
            method4_result
        ),
    ]

    for method_name, result in results:

        if result is None:

            print(
                f"{method_name:<40}"
                f"{'FAILED':>20}"
                f"{'FAILED':>20}"
            )

        else:

            print(
                f"{method_name:<40}"
                f"{result['max_error']:>20.10f}"
                f"{result['mean_error']:>20.10f}"
            )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 70)
    print("RANDOM FOREST SHAP METHOD VERIFICATION COMPLETED")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "This script is diagnostic only."
    )

    print(
        "No model, feature data, or SHAP artifact was modified."
    )

    print(
        "Do NOT modify shap_analysis.py based only on "
        "check_additivity=False."
    )

    print(
        "Use the method comparison above to determine "
        "the next correction."
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

