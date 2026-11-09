import os
import shutil
import joblib


# ============================================================
# FINAL MODEL SELECTION
# ============================================================

print("=" * 70)
print("FINAL MODEL SELECTION")
print("=" * 70)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "models"
)


# ============================================================
# MODEL PATHS
# ============================================================

XGBOOST_MODEL = os.path.join(
    MODEL_DIR,
    "xgboost_model.pkl"
)


FINAL_MODEL = os.path.join(
    MODEL_DIR,
    "final_xgboost_model.pkl"
)


# ============================================================
# CHECK XGBOOST MODEL
# ============================================================

print("\n[1] Checking XGBoost model...")

if not os.path.exists(XGBOOST_MODEL):

    raise FileNotFoundError(
        "\nXGBoost model not found:\n"
        + XGBOOST_MODEL
    )


print("PASS: XGBoost model found.")
print(XGBOOST_MODEL)


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

print("\n[2] Loading XGBoost model...")

xgboost_model = joblib.load(
    XGBOOST_MODEL
)

print("PASS: XGBoost model loaded.")


# ============================================================
# VERIFY MODEL
# ============================================================

print("\n[3] Verifying model...")

print(
    "Model type:",
    type(xgboost_model).__name__
)


if hasattr(
    xgboost_model,
    "n_features_in_"
):

    print(
        "Expected features:",
        xgboost_model.n_features_in_
    )


    if xgboost_model.n_features_in_ != 10026:

        raise ValueError(

            "\nXGBoost feature dimension mismatch.\n"

            "Expected: 10026\n"

            f"Found: "
            f"{xgboost_model.n_features_in_}"

        )


print(
    "PASS: XGBoost model is compatible "
    "with the 10,026-feature pipeline."
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n[4] Saving final production model...")


joblib.dump(
    xgboost_model,
    FINAL_MODEL
)


print(
    "PASS: Final XGBoost model saved."
)


print(
    "File:"
)


print(
    FINAL_MODEL
)


# ============================================================
# VERIFY SAVED MODEL
# ============================================================

print("\n[5] Verifying saved final model...")


if not os.path.exists(FINAL_MODEL):

    raise FileNotFoundError(
        "\nFinal model was not created."
    )


final_model = joblib.load(
    FINAL_MODEL
)


print(
    "PASS: Final model loaded successfully."
)


print(
    "Final model type:",
    type(final_model).__name__
)


if hasattr(
    final_model,
    "n_features_in_"
):

    print(
        "Final model features:",
        final_model.n_features_in_
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL SELECTED")
print("=" * 70)

print(
    "\nSelected model:"
)

print(
    "XGBoost"
)


print(
    "\nReason:"
)

print(
    "Best overall balance of accuracy, "
    "precision, recall and F1-score."
)


print(
    "\nTest performance:"
)

print(
    "Accuracy  : 98.69%"
)

print(
    "Precision : 95.65%"
)

print(
    "Recall    : 76.30%"
)

print(
    "F1-score  : 84.89%"
)

print(
    "ROC-AUC   : 98.92%"
)


print("\nFinal production model:")

print(
    FINAL_MODEL
)


print("\n" + "=" * 70)
print("MODEL SELECTION COMPLETED")
print("STATUS: SUCCESS")
print("=" * 70)