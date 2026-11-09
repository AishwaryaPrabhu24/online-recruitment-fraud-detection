import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "dataset/fake_job_postings.csv"
OUTPUT_PATH = "dataset/processed/cleaned_job_postings.csv"

# Explicit marker for missing/optional information
MISSING_VALUE = "[NOT_PROVIDED]"


# ============================================================
# CREATE PROCESSED DIRECTORY
# ============================================================

os.makedirs("dataset/processed", exist_ok=True)


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)

print("Original shape:", df.shape)


# ============================================================
# REMOVE UNNECESSARY COLUMN
# ============================================================

if "title.1" in df.columns:
    df = df.drop(columns=["title.1"])

print("Shape after removing title.1:", df.shape)


# ============================================================
# MISSING VALUE SUMMARY BEFORE CLEANING
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE SUMMARY BEFORE CLEANING")
print("=" * 60)

missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0]

if len(missing_before) > 0:

    missing_summary = pd.DataFrame({
        "missing_count": missing_before,
        "missing_percentage":
            (missing_before / len(df) * 100).round(2)
    })

    print(missing_summary)

else:

    print("No missing values found.")


# ============================================================
# TEXT COLUMNS
# ============================================================

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
    "function"
]


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("\nReplacing missing values with:", MISSING_VALUE)

for column in df.columns:

    if column in text_columns:

        df[column] = df[column].fillna(MISSING_VALUE)

    elif column == "fraudulent":

        # Target column must remain unchanged
        continue

    else:

        # Numeric columns
        df[column] = df[column].fillna(0)


# ============================================================
# CLEAN TEXT COLUMNS
# ============================================================

print("\nCleaning text columns...")

for column in text_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.lower()
        )

        # Convert empty strings to explicit missing-value marker
        df[column] = df[column].replace("", MISSING_VALUE)


# ============================================================
# CREATE COMBINED TEXT
# ============================================================

print("\nCreating combined text...")

df["combined_text"] = (
    "title " + df["title"] +
    " company_profile " + df["company_profile"] +
    " description " + df["description"] +
    " requirements " + df["requirements"] +
    " benefits " + df["benefits"] +
    " salary_range " + df["salary_range"]
)


# ============================================================
# CLEAN COMBINED TEXT
# ============================================================

df["combined_text"] = (
    df["combined_text"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# ============================================================
# MISSING VALUES AFTER CLEANING
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES AFTER CLEANING")
print("=" * 60)

missing_after = df.isnull().sum()

print(missing_after)

if missing_after.sum() == 0:

    print("\nPASS: No NaN values remain.")

else:

    print("\nWARNING: Missing values still exist.")


# ============================================================
# EMPTY TEXT VALUES CHECK
# ============================================================

print("\n" + "=" * 60)
print("EMPTY TEXT VALUES")
print("=" * 60)

empty_text_total = 0

for column in text_columns + ["combined_text"]:

    if column in df.columns:

        empty_count = (
            df[column]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        empty_text_total += empty_count

        print(f"{column}: {empty_count}")


if empty_text_total == 0:

    print("\nPASS: No empty text values remain.")

else:

    print("\nWARNING: Empty text values still exist.")


# ============================================================
# CLEANED SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("CLEANED SAMPLE")
print("=" * 60)

sample = df.iloc[0]

print("\nTITLE:")
print(sample["title"])

print("\nCOMPANY PROFILE:")
print(sample["company_profile"])

print("\nDESCRIPTION:")
print(sample["description"])

print("\nREQUIREMENTS:")
print(sample["requirements"])

print("\nBENEFITS:")
print(sample["benefits"])

print("\nSALARY RANGE:")
print(sample["salary_range"])


# ============================================================
# COMBINED TEXT SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("COMBINED TEXT SAMPLE")
print("=" * 60)

print(df.iloc[0]["combined_text"])


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

target_counts = df["fraudulent"].value_counts()

print(target_counts)

print("\nTarget percentage:")

target_percentage = (
    df["fraudulent"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(target_percentage)


# ============================================================
# SAVE CLEANED DATASET
# ============================================================

print("\n" + "=" * 60)
print("SAVING CLEANED DATASET")
print("=" * 60)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("File:", OUTPUT_PATH)
print("Shape:", df.shape)


# ============================================================
# VERIFY SAVED DATASET
# ============================================================

print("\n" + "=" * 60)
print("VERIFYING SAVED DATASET")
print("=" * 60)

# keep_default_na=False prevents [NOT_PROVIDED]
# or empty text fields from being interpreted as NaN
df_check = pd.read_csv(
    OUTPUT_PATH,
    keep_default_na=False
)

print("Reloaded shape:", df_check.shape)


# ============================================================
# CHECK MISSING VALUES AFTER RELOADING
# ============================================================

missing_check = df_check.isnull().sum()

print("\nMissing values after reloading:")

print(missing_check)

if missing_check.sum() == 0:

    print("\nPASS: Saved dataset contains no NaN values.")

else:

    print("\nFAIL: Saved dataset contains missing values.")


# ============================================================
# CHECK EMPTY STRINGS AFTER RELOADING
# ============================================================

print("\n" + "=" * 60)
print("CHECKING EMPTY VALUES AFTER RELOADING")
print("=" * 60)

reload_empty_total = 0

for column in text_columns + ["combined_text"]:

    if column in df_check.columns:

        empty_count = (
            df_check[column]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        reload_empty_total += empty_count

        print(f"{column}: {empty_count}")


if reload_empty_total == 0:

    print("\nPASS: No empty text values found after reloading.")

else:

    print("\nWARNING: Empty text values found after reloading.")


# ============================================================
# FINAL DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("PROCESSING COMPLETED")
print("=" * 60)

print("Final shape:", df_check.shape)

print("\nFinal columns:")
print(list(df_check.columns))

print("\nFinal missing values:")
print(df_check.isnull().sum())

print("\nTarget distribution:")
print(df_check["fraudulent"].value_counts())

print("\nDataset saved successfully at:")
print(OUTPUT_PATH)