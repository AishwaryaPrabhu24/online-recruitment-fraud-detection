import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. CREATE OUTPUT FOLDER
# ============================================================

output_folder = "visualizations"

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# 2. LOAD DATASETS
# ============================================================

# Cleaned dataset - used for most EDA visualizations
df = pd.read_csv(
    "dataset/processed/cleaned_job_postings.csv"
)

# Original dataset - used only for missing-value analysis
original_df = pd.read_csv(
    "dataset/fake_job_postings.csv"
)


print("=" * 50)
print("DATASETS LOADED")
print("=" * 50)

print("Cleaned dataset shape:", df.shape)
print("Original dataset shape:", original_df.shape)


# ============================================================
# 3. SET SEABORN STYLE
# ============================================================

sns.set_theme(style="whitegrid")


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

ax = sns.countplot(
    data=df,
    x="fraudulent"
)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Genuine", "Fraudulent"])

plt.title(
    "Distribution of Genuine and Fraudulent Job Postings",
    fontsize=14
)

plt.xlabel("Job Type")
plt.ylabel("Number of Job Postings")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "01_target_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. MISSING VALUE VISUALIZATION
# ============================================================

missing = original_df.isnull().sum()

missing = (
    missing[missing > 0]
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=missing.values,
    y=missing.index
)

plt.title(
    "Missing Values in Original Job Posting Dataset",
    fontsize=14
)

plt.xlabel("Number of Missing Values")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "02_missing_values.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. BINARY FEATURE DISTRIBUTIONS
# ============================================================

binary_columns = [
    "telecommuting",
    "has_company_logo",
    "has_questions"
]


for i, column in enumerate(binary_columns, start=3):

    plt.figure(figsize=(7, 5))

    ax = sns.countplot(
        data=df,
        x=column
    )

    ax.set_xticks([0, 1])

    ax.set_xticklabels(
        ["No", "Yes"]
    )

    plt.title(
        f"Distribution of {column.replace('_', ' ').title()}",
        fontsize=14
    )

    plt.xlabel(
        column.replace("_", " ").title()
    )

    plt.ylabel(
        "Number of Job Postings"
    )

    plt.tight_layout()

    filename = f"{i:02d}_{column}_distribution.png"

    plt.savefig(
        os.path.join(
            output_folder,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 7. BINARY FEATURES VS FRAUD RATE
# ============================================================

for i, column in enumerate(binary_columns, start=6):

    fraud_rate = (
        df.groupby(column)["fraudulent"]
        .mean()
        .reset_index()
    )

    fraud_rate["fraudulent"] = (
        fraud_rate["fraudulent"] * 100
    )

    plt.figure(figsize=(7, 5))

    ax = sns.barplot(
        data=fraud_rate,
        x=column,
        y="fraudulent"
    )

    ax.set_xticks([0, 1])

    ax.set_xticklabels(
        ["No", "Yes"]
    )

    plt.title(
        f"Fraud Rate by {column.replace('_', ' ').title()}",
        fontsize=14
    )

    plt.xlabel(
        column.replace("_", " ").title()
    )

    plt.ylabel(
        "Fraud Rate (%)"
    )

    plt.tight_layout()

    filename = f"{i:02d}_fraud_rate_by_{column}.png"

    plt.savefig(
        os.path.join(
            output_folder,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 8. EMPLOYMENT TYPE - GENUINE VS FRAUDULENT
# ============================================================

plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="employment_type",
    hue="fraudulent"
)

plt.title(
    "Genuine vs Fraudulent Jobs by Employment Type",
    fontsize=14
)

plt.xlabel("Employment Type")
plt.ylabel("Number of Job Postings")

plt.xticks(rotation=30)

plt.legend(
    title="Job Type",
    labels=["Genuine", "Fraudulent"]
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "09_employment_type_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. FRAUD RATE BY EMPLOYMENT TYPE
# ============================================================

employment_fraud = (
    df.groupby("employment_type")["fraudulent"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 6))

sns.barplot(
    x=employment_fraud.values,
    y=employment_fraud.index
)

plt.title(
    "Fraud Rate by Employment Type",
    fontsize=14
)

plt.xlabel("Fraud Rate (%)")
plt.ylabel("Employment Type")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "10_fraud_rate_by_employment_type.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. FRAUD RATE BY REQUIRED EXPERIENCE
# ============================================================

experience_fraud = (
    df.groupby("required_experience")["fraudulent"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 6))

sns.barplot(
    x=experience_fraud.values,
    y=experience_fraud.index
)

plt.title(
    "Fraud Rate by Required Experience",
    fontsize=14
)

plt.xlabel("Fraud Rate (%)")
plt.ylabel("Required Experience")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "11_fraud_rate_by_experience.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. FRAUD RATE BY REQUIRED EDUCATION
# ============================================================

education_fraud = (
    df.groupby("required_education")["fraudulent"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 7))

sns.barplot(
    x=education_fraud.values,
    y=education_fraud.index
)

plt.title(
    "Fraud Rate by Required Education",
    fontsize=14
)

plt.xlabel("Fraud Rate (%)")
plt.ylabel("Required Education")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "12_fraud_rate_by_education.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. FRAUD RATE BY INDUSTRY
# ============================================================

industry_fraud = (
    df.groupby("industry")["fraudulent"]
    .agg(["count", "mean"])
)

industry_fraud["fraud_rate"] = (
    industry_fraud["mean"] * 100
)

# Ignore very small categories
industry_fraud = (
    industry_fraud[
        industry_fraud["count"] >= 20
    ]
    .sort_values(
        "fraud_rate",
        ascending=False
    )
)

top_industries = (
    industry_fraud
    .head(15)
    .reset_index()
)

plt.figure(figsize=(11, 8))

sns.barplot(
    data=top_industries,
    x="fraud_rate",
    y="industry"
)

plt.title(
    "Top Industries by Observed Fraud Rate",
    fontsize=14
)

plt.xlabel("Fraud Rate (%)")
plt.ylabel("Industry")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "13_fraud_rate_by_industry.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. FRAUD RATE BY JOB FUNCTION
# ============================================================

function_fraud = (
    df.groupby("function")["fraudulent"]
    .agg(["count", "mean"])
)

function_fraud["fraud_rate"] = (
    function_fraud["mean"] * 100
)

# Ignore very small categories
function_fraud = (
    function_fraud[
        function_fraud["count"] >= 20
    ]
    .sort_values(
        "fraud_rate",
        ascending=False
    )
)

plt.figure(figsize=(11, 8))

sns.barplot(
    data=function_fraud.reset_index(),
    x="fraud_rate",
    y="function"
)

plt.title(
    "Fraud Rate by Job Function",
    fontsize=14
)

plt.xlabel("Fraud Rate (%)")
plt.ylabel("Job Function")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "14_fraud_rate_by_job_function.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. TEXT LENGTH ANALYSIS
# ============================================================

df["text_length"] = (
    df["combined_text"]
    .astype(str)
    .str.len()
)

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="text_length",
    hue="fraudulent",
    bins=50,
    kde=True
)

plt.title(
    "Text Length Distribution of Job Postings",
    fontsize=14
)

plt.xlabel(
    "Text Length (Characters)"
)

plt.ylabel(
    "Number of Job Postings"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "15_text_length_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. WORD COUNT ANALYSIS
# ============================================================

df["word_count"] = (
    df["combined_text"]
    .astype(str)
    .str.split()
    .str.len()
)

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="word_count",
    hue="fraudulent",
    bins=50,
    kde=True
)

plt.title(
    "Word Count Distribution of Job Postings",
    fontsize=14
)

plt.xlabel(
    "Word Count"
)

plt.ylabel(
    "Number of Job Postings"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "16_word_count_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. TEXT LENGTH COMPARISON
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="fraudulent",
    y="text_length"
)

plt.title(
    "Text Length Comparison: Genuine vs Fraudulent",
    fontsize=14
)

plt.xlabel("Job Type")
plt.ylabel("Text Length")

plt.xticks(
    [0, 1],
    ["Genuine", "Fraudulent"]
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "17_text_length_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. FINISHED
# ============================================================

print()
print("=" * 60)
print("EDA VISUALIZATION COMPLETED")
print("=" * 60)

print()
print("All visualizations have been saved to:")
print(output_folder)

print()
print("Generated files:")

for file in sorted(os.listdir(output_folder)):
    print(" -", file)

print()
print("Total visualizations:", len(os.listdir(output_folder)))
print("=" * 60)