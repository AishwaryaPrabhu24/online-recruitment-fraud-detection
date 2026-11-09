import pandas as pd
import os

# ============================================================
# FINAL DATA VALIDATION
# ============================================================

INPUT_FILE = "dataset/processed/cleaned_job_postings.csv"

print("=" * 60)
print("FINAL DATA VALIDATION")
print("=" * 60)


# ============================================================
# 1. CHECK FILE EXISTS
# ============================================================

print("\n[1] Checking cleaned dataset file...")

if not os.path.exists(INPUT_FILE):
    print("ERROR: Cleaned dataset file not found!")
    print("Expected file:", INPUT_FILE)
    exit()

print("PASS: Cleaned dataset file exists.")
print("File:", INPUT_FILE)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n[2] Loading cleaned dataset...")

try:
    df = pd.read_csv(INPUT_FILE)
    print("PASS: Dataset loaded successfully.")
except Exception as e:
    print("ERROR: Could not load dataset.")
    print("Reason:", e)
    exit()


# ============================================================
# 3. CHECK DATASET SHAPE
# ============================================================

print("\n[3] Checking dataset shape...")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

if df.shape[0] == 17880:
    print("PASS: Expected 17,880 rows found.")
else:
    print("WARNING: Expected 17,880 rows, but found", df.shape[0])


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

print("\n[4] Checking required columns...")

required_columns = [
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

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

extra_columns = [
    column for column in df.columns
    if column not in required_columns
]

if len(missing_columns) == 0:
    print("PASS: All required columns are present.")
else:
    print("FAIL: Missing columns:")
    for column in missing_columns:
        print(" -", column)

if len(extra_columns) == 0:
    print("PASS: No unexpected columns found.")
else:
    print("INFO: Extra columns found:")
    for column in extra_columns:
        print(" -", column)


# ============================================================
# 5. CHECK title.1 WAS REMOVED
# ============================================================

print("\n[5] Checking removed column...")

if "title.1" not in df.columns:
    print("PASS: 'title.1' column has been removed.")
else:
    print("FAIL: 'title.1' column still exists.")


# ============================================================
# 6. CHECK DUPLICATE ROWS
# ============================================================

print("\n[6] Checking duplicate rows...")

duplicate_rows = df.duplicated().sum()

print("Duplicate rows:", duplicate_rows)

if duplicate_rows == 0:
    print("PASS: No duplicate rows found.")
else:
    print("WARNING:", duplicate_rows, "duplicate rows found.")


# ============================================================
# 7. CHECK DUPLICATE JOB IDs
# ============================================================

print("\n[7] Checking duplicate job IDs...")

duplicate_job_ids = df["job_id"].duplicated().sum()

print("Duplicate job IDs:", duplicate_job_ids)

if duplicate_job_ids == 0:
    print("PASS: All job IDs are unique.")
else:
    print("WARNING:", duplicate_job_ids, "duplicate job IDs found.")


# ============================================================
# 8. CHECK MISSING VALUES
# ============================================================

print("\n[8] Checking missing values...")

missing_values = df.isnull().sum()

total_missing = missing_values.sum()

print("Total missing values:", total_missing)

if total_missing == 0:
    print("PASS: No NaN values found.")
else:
    print("FAIL: Missing values found:")
    print(missing_values[missing_values > 0])


# ============================================================
# 9. CHECK EMPTY TEXT VALUES
# ============================================================

print("\n[9] Checking empty text values...")

text_columns = [
    "title",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "combined_text"
]

empty_values = {}

for column in text_columns:
    empty_count = (
        df[column]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    empty_values[column] = empty_count

total_empty = sum(empty_values.values())

print("Empty text cells:", total_empty)

for column, count in empty_values.items():
    if count > 0:
        print(f" - {column}: {count}")

print(
    "\nINFO: Empty strings are allowed in some optional fields "
    "because the original dataset contains missing information."
)


# ============================================================
# 10. CHECK TARGET COLUMN
# ============================================================

print("\n[10] Checking target column...")

if "fraudulent" in df.columns:

    unique_targets = sorted(df["fraudulent"].unique())

    print("Unique target values:", unique_targets)

    if set(unique_targets).issubset({0, 1}):
        print("PASS: Target contains only 0 and 1.")
    else:
        print("FAIL: Unexpected target values found.")

    print("\nTarget distribution:")
    print(df["fraudulent"].value_counts())

else:
    print("FAIL: 'fraudulent' column not found.")


# ============================================================
# 11. CHECK TARGET CLASS BALANCE
# ============================================================

print("\n[11] Checking target class distribution...")

target_counts = df["fraudulent"].value_counts()

genuine_count = target_counts.get(0, 0)
fraudulent_count = target_counts.get(1, 0)

total_records = len(df)

genuine_percentage = (genuine_count / total_records) * 100
fraudulent_percentage = (fraudulent_count / total_records) * 100

print(f"Genuine jobs     : {genuine_count} ({genuine_percentage:.2f}%)")
print(f"Fraudulent jobs  : {fraudulent_count} ({fraudulent_percentage:.2f}%)")

print("INFO: Dataset is imbalanced because genuine jobs are much more common.")


# ============================================================
# 12. CHECK BINARY FEATURES
# ============================================================

print("\n[12] Checking binary features...")

binary_columns = [
    "telecommuting",
    "has_company_logo",
    "has_questions"
]

for column in binary_columns:

    unique_values = sorted(df[column].unique())

    print(f"\n{column}")
    print("Unique values:", unique_values)

    if set(unique_values).issubset({0, 1}):
        print("PASS: Contains only 0 and 1.")
    else:
        print("FAIL: Unexpected values found.")


# ============================================================
# 13. CHECK TEXT COLUMN
# ============================================================

print("\n[13] Checking combined_text...")

if "combined_text" in df.columns:

    empty_combined_text = (
        df["combined_text"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print("Empty combined_text values:", empty_combined_text)

    if empty_combined_text == 0:
        print("PASS: All records contain combined text.")
    else:
        print("WARNING: Empty combined_text values found.")

else:
    print("FAIL: combined_text column not found.")


# ============================================================
# 14. CHECK TEXT LENGTH
# ============================================================

print("\n[14] Checking text length...")

text_lengths = df["combined_text"].astype(str).str.len()

print("Minimum text length:", text_lengths.min())
print("Maximum text length:", text_lengths.max())
print("Average text length:", round(text_lengths.mean(), 2))

if text_lengths.min() > 0:
    print("PASS: All combined_text records contain text.")
else:
    print("WARNING: Some records have zero text length.")


# ============================================================
# 15. CHECK DATA TYPES
# ============================================================

print("\n[15] Checking important data types...")

print(df.dtypes)

if pd.api.types.is_numeric_dtype(df["job_id"]):
    print("PASS: job_id is numeric.")
else:
    print("WARNING: job_id is not numeric.")

if pd.api.types.is_numeric_dtype(df["fraudulent"]):
    print("PASS: fraudulent is numeric.")
else:
    print("WARNING: fraudulent is not numeric.")


# ============================================================
# 16. CHECK DATA CONSISTENCY
# ============================================================

print("\n[16] Checking data consistency...")

checks_passed = True

# Check number of records
if len(df) != 17880:
    checks_passed = False

# Check target values
if not set(df["fraudulent"].unique()).issubset({0, 1}):
    checks_passed = False

# Check binary columns
for column in binary_columns:
    if not set(df[column].unique()).issubset({0, 1}):
        checks_passed = False

# Check required columns
if len(missing_columns) > 0:
    checks_passed = False

# Check missing values
if total_missing > 0:
    checks_passed = False

# Check duplicates
if duplicate_rows > 0:
    checks_passed = False

# Check duplicate job IDs
if duplicate_job_ids > 0:
    checks_passed = False

# Check title.1 removal
if "title.1" in df.columns:
    checks_passed = False


if checks_passed:
    print("PASS: Main data consistency checks passed.")
else:
    print("WARNING: One or more consistency checks require attention.")


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL VALIDATION SUMMARY")
print("=" * 60)

print("Dataset file              :", INPUT_FILE)
print("Dataset shape             :", df.shape)
print("Duplicate rows            :", duplicate_rows)
print("Duplicate job IDs         :", duplicate_job_ids)
print("Total missing values     :", total_missing)
print("Genuine jobs              :", genuine_count)
print("Fraudulent jobs           :", fraudulent_count)
print("Fraud percentage          :", f"{fraudulent_percentage:.2f}%")
print("Required columns missing :", len(missing_columns))
print("Removed title.1           :", "YES" if "title.1" not in df.columns else "NO")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)

if checks_passed:
    print("FINAL STATUS: VALIDATION PASSED")
    print("The cleaned dataset is ready for the next ML stage.")
else:
    print("FINAL STATUS: VALIDATION REQUIRES ATTENTION")
    print("Check the warnings/errors shown above.")

print("=" * 60)