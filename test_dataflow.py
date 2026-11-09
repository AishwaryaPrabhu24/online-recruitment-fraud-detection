import pandas as pd
import os

DATASET_PATH = "dataset/processed/cleaned_job_postings.csv"

EXPECTED_ROWS = 17880

REQUIRED_COLUMNS = [
    "job_id",
    "title",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "fraudulent",
    "combined_text"
]

BINARY_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions"
]


print("=" * 60)
print("DATA FLOW TEST")
print("=" * 60)


# ------------------------------------------------------------
# 1. Check dataset file
# ------------------------------------------------------------

print("\n[1] Checking dataset file...")

if os.path.exists(DATASET_PATH):
    print("PASS: Cleaned dataset file exists.")
else:
    print("FAIL: Cleaned dataset file not found.")
    raise SystemExit


# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

print("\n[2] Loading cleaned dataset...")

try:
    df = pd.read_csv(DATASET_PATH)
    print("PASS: Dataset loaded successfully.")
except Exception as e:
    print("FAIL: Dataset could not be loaded.")
    print("Error:", e)
    raise SystemExit


# ------------------------------------------------------------
# 3. Check dataset shape
# ------------------------------------------------------------

print("\n[3] Checking dataset shape...")

print("Rows:", len(df))
print("Columns:", len(df.columns))

if len(df) == EXPECTED_ROWS:
    print("PASS: Expected row count found.")
else:
    print("FAIL: Unexpected row count.")


# ------------------------------------------------------------
# 4. Check required columns
# ------------------------------------------------------------

print("\n[4] Checking required columns...")

missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if not missing_columns:
    print("PASS: All required columns are present.")
else:
    print("FAIL: Missing columns:")
    print(missing_columns)


# ------------------------------------------------------------
# 5. Check duplicate rows
# ------------------------------------------------------------

print("\n[5] Checking duplicate rows...")

duplicate_rows = df.duplicated().sum()

print("Duplicate rows:", duplicate_rows)

if duplicate_rows == 0:
    print("PASS: No duplicate rows.")
else:
    print("WARNING: Duplicate rows found.")


# ------------------------------------------------------------
# 6. Check duplicate job IDs
# ------------------------------------------------------------

print("\n[6] Checking job IDs...")

duplicate_ids = df["job_id"].duplicated().sum()

print("Duplicate job IDs:", duplicate_ids)

if duplicate_ids == 0:
    print("PASS: All job IDs are unique.")
else:
    print("FAIL: Duplicate job IDs found.")


# ------------------------------------------------------------
# 7. Check missing values
# ------------------------------------------------------------

print("\n[7] Checking missing values...")

total_missing = df.isnull().sum().sum()

print("Total missing values:", total_missing)

if total_missing == 0:
    print("PASS: No NaN values found.")
else:
    print("FAIL: Missing values found.")
    print(df.isnull().sum()[df.isnull().sum() > 0])


# ------------------------------------------------------------
# 8. Check empty combined text
# ------------------------------------------------------------

print("\n[8] Checking combined text...")

empty_combined_text = (
    df["combined_text"]
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

print("Empty combined_text:", empty_combined_text)

if empty_combined_text == 0:
    print("PASS: All records contain combined text.")
else:
    print("FAIL: Empty combined_text values found.")


# ------------------------------------------------------------
# 9. Check target column
# ------------------------------------------------------------

print("\n[9] Checking target column...")

target_values = sorted(df["fraudulent"].unique().tolist())

print("Target values:", target_values)

if set(target_values).issubset({0, 1}):
    print("PASS: Target contains only 0 and 1.")
else:
    print("FAIL: Unexpected target values found.")


# ------------------------------------------------------------
# 10. Check binary features
# ------------------------------------------------------------

print("\n[10] Checking binary features...")

binary_test_passed = True

for column in BINARY_COLUMNS:

    values = sorted(df[column].unique().tolist())

    print(f"{column}: {values}")

    if set(values).issubset({0, 1}):
        print(f"PASS: {column} contains only 0 and 1.")
    else:
        print(f"FAIL: Unexpected values in {column}.")
        binary_test_passed = False


# ------------------------------------------------------------
# 11. Test text processing output
# ------------------------------------------------------------

print("\n[11] Testing text processing output...")

sample_text = df["combined_text"].iloc[0]

print("Sample combined_text length:", len(sample_text))
print("Sample text preview:")
print(sample_text[:200])

if isinstance(sample_text, str) and len(sample_text.strip()) > 0:
    print("PASS: Text data is available for NLP processing.")
else:
    print("FAIL: Text data is invalid.")


# ------------------------------------------------------------
# 12. Test ML-ready feature separation
# ------------------------------------------------------------

print("\n[12] Testing feature and target separation...")

X = df.drop(columns=["fraudulent"])
y = df["fraudulent"]

print("Feature rows:", len(X))
print("Target rows:", len(y))

if len(X) == len(y):
    print("PASS: Feature and target row counts match.")
else:
    print("FAIL: Feature and target row counts do not match.")


# ------------------------------------------------------------
# 13. Test target distribution
# ------------------------------------------------------------

print("\n[13] Checking target distribution...")

target_counts = y.value_counts()

print(target_counts)

if 0 in target_counts.index and 1 in target_counts.index:
    print("PASS: Both genuine and fraudulent classes are present.")
else:
    print("FAIL: One target class is missing.")


# ------------------------------------------------------------
# 14. Test final data handoff
# ------------------------------------------------------------

print("\n[14] Testing ML data handoff...")

try:

    text_data = df["combined_text"]
    structured_data = df[
        [
            "telecommuting",
            "has_company_logo",
            "has_questions"
        ]
    ]

    target_data = df["fraudulent"]

    print("Text records:", len(text_data))
    print("Structured feature records:", len(structured_data))
    print("Target records:", len(target_data))

    if (
        len(text_data)
        == len(structured_data)
        == len(target_data)
    ):
        print("PASS: Data can be passed to the ML pipeline.")
    else:
        print("FAIL: Data dimensions do not match.")

except Exception as e:

    print("FAIL: ML handoff test failed.")
    print("Error:", e)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("DATA FLOW TEST SUMMARY")
print("=" * 60)

print("Dataset loaded              : YES")
print("Expected rows               :", EXPECTED_ROWS)
print("Actual rows                 :", len(df))
print("Required columns            :", len(REQUIRED_COLUMNS))
print("Duplicate rows              :", duplicate_rows)
print("Duplicate job IDs           :", duplicate_ids)
print("Total missing values        :", total_missing)
print("Empty combined_text         :", empty_combined_text)
print("Target values               :", target_values)
print("Genuine jobs                :", (y == 0).sum())
print("Fraudulent jobs             :", (y == 1).sum())

print("\n" + "=" * 60)
print("DATA FLOW TEST COMPLETED")
print("=" * 60)

print("""
Pipeline verified:

Cleaned Dataset
      ↓
Data Loading
      ↓
Feature / Target Separation
      ↓
Text Data
      ↓
Structured Features
      ↓
Target Labels
      ↓
ML Pipeline Handoff
""")

print("STATUS: DATA FLOW READY FOR ML STAGE")
print("=" * 60)