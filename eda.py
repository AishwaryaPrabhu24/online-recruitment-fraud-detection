import pandas as pd

file_path = "dataset/processed/cleaned_job_postings.csv"

df = pd.read_csv(file_path)

print("==========================================")
print("DATASET LOADED")
print("==========================================")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\n==========================================")
print("DATASET INFORMATION")
print("==========================================")

print(df.info())

print("\n==========================================")
print("FIRST 5 ROWS")
print("==========================================")

print(df.head())

print("\n==========================================")
print("TARGET DISTRIBUTION")
print("==========================================")

print(df["fraudulent"].value_counts())

print("\nTarget percentage:")

print(
    df["fraudulent"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n==========================================")
print("DUPLICATE CHECK")
print("==========================================")

print("Duplicate rows:", df.duplicated().sum())

print("\n==========================================")
print("MISSING VALUE CHECK")
print("==========================================")

print(df.isnull().sum())

print("\n==========================================")
print("NUMERICAL COLUMNS")
print("==========================================")

print(df.select_dtypes(include="number").columns)

print("\n==========================================")
print("CATEGORICAL COLUMNS")
print("==========================================")

print(df.select_dtypes(include="object").columns)

print("\n==========================================")
print("GENUINE VS FRAUDULENT JOBS")
print("==========================================")

genuine_jobs = df[df["fraudulent"] == 0]
fraudulent_jobs = df[df["fraudulent"] == 1]

print("Genuine jobs:", len(genuine_jobs))
print("Fraudulent jobs:", len(fraudulent_jobs))

print("\n==========================================")
print("BINARY FEATURE DISTRIBUTION")
print("==========================================")

binary_columns = [
    "telecommuting",
    "has_company_logo",
    "has_questions"
]

for column in binary_columns:

    print("\n" + column)

    print(
        df[column]
        .value_counts()
        .sort_index()
    )

print("\n==========================================")
print("BINARY FEATURES VS FRAUD")
print("==========================================")

for column in binary_columns:

    print("\n------------------------------------------")
    print(column)
    print("------------------------------------------")

    print(
        pd.crosstab(
            df[column],
            df["fraudulent"],
            normalize="index"
        ).mul(100).round(2)
    )

categorical_columns = [
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

print("\n==========================================")
print("CATEGORICAL FEATURE ANALYSIS")
print("==========================================")

for column in categorical_columns:

    print("\n==========================================")
    print(column.upper())
    print("==========================================")

    print(df[column].value_counts().head(15))

print("\n==========================================")
print("FRAUDULENT JOBS BY CATEGORY")
print("==========================================")

for column in categorical_columns:

    print("\n------------------------------------------")
    print(column)
    print("------------------------------------------")

    fraud_counts = (
        df[df["fraudulent"] == 1][column]
        .value_counts()
        .head(10)
    )

    print(fraud_counts)

df["text_length"] = df["combined_text"].str.len()

print("\n==========================================")
print("TEXT LENGTH ANALYSIS")
print("==========================================")

print(
    df.groupby("fraudulent")["text_length"]
    .describe()
)

df["word_count"] = (
    df["combined_text"]
    .str.split()
    .str.len()
)

print("\n==========================================")
print("WORD COUNT ANALYSIS")
print("==========================================")

print(
    df.groupby("fraudulent")["word_count"]
    .describe()
)

original_df = pd.read_csv(
    "dataset/fake_job_postings.csv"
)

original_df = original_df.drop(
    columns=["title.1"],
    errors="ignore"
)

print("\n==========================================")
print("ORIGINAL MISSING VALUE PERCENTAGES")
print("==========================================")

missing_percentage = (
    original_df.isnull()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(missing_percentage.round(2))

original_df["salary_missing"] = (
    original_df["salary_range"].isnull()
)

print("\n==========================================")
print("SALARY MISSINGNESS VS FRAUD")
print("==========================================")

print(
    pd.crosstab(
        original_df["salary_missing"],
        original_df["fraudulent"],
        normalize="index"
    ).mul(100).round(2)
)

original_df["benefits_missing"] = (
    original_df["benefits"].isnull()
)

print("\n==========================================")
print("BENEFITS MISSINGNESS VS FRAUD")
print("==========================================")

print(
    pd.crosstab(
        original_df["benefits_missing"],
        original_df["fraudulent"],
        normalize="index"
    ).mul(100).round(2)
)

original_df["company_profile_missing"] = (
    original_df["company_profile"].isnull()
)

print("\n==========================================")
print("COMPANY PROFILE MISSINGNESS VS FRAUD")
print("==========================================")

print(
    pd.crosstab(
        original_df["company_profile_missing"],
        original_df["fraudulent"],
        normalize="index"
    ).mul(100).round(2)
)


print("\n==========================================")
print("FRAUD RATE BY CATEGORICAL FEATURES")
print("==========================================")

for column in categorical_columns:

    print("\n------------------------------------------")
    print(column.upper())
    print("------------------------------------------")

    fraud_rate = (
        df.groupby(column)["fraudulent"]
        .agg(["count", "sum", "mean"])
        .sort_values("mean", ascending=False)
    )

    fraud_rate["fraud_percentage"] = (
        fraud_rate["mean"] * 100
    )

    fraud_rate = fraud_rate.drop(columns=["mean"])

    print(
        fraud_rate.head(15).round(2)
    )