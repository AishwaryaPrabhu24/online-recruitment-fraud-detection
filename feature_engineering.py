import os
import re
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "dataset/processed/cleaned_job_postings.csv"

OUTPUT_DIR = "dataset/processed/features"

RANDOM_STATE = 42

TEST_SIZE = 0.20

# TF-IDF settings
MAX_FEATURES = 10000

NGRAM_RANGE = (1, 2)

MIN_DF = 2


# ============================================================
# SUSPICIOUS / RED-FLAG KEYWORDS
# ============================================================

SUSPICIOUS_KEYWORDS = [

    "easy money",
    "quick money",
    "make money",
    "earn money",
    "work from home",
    "no experience required",
    "no experience necessary",
    "urgent hiring",
    "immediate hiring",
    "cash bonus",
    "weekly pay",
    "daily pay",
    "guaranteed income",
    "guaranteed money",
    "financial freedom",
    "investment opportunity",
    "wire transfer",
    "bank account",
    "send money",
    "payment required",
    "processing fee",
    "registration fee",
    "training fee",
    "application fee",

]


# ============================================================
# HELPER FUNCTION
# ============================================================

def count_keyword_occurrences(text, keywords):

    """
    Count suspicious keyword/phrase occurrences.
    """

    text = str(text).lower()

    count = 0

    for keyword in keywords:

        count += text.count(keyword)

    return count


# ============================================================
# RED-FLAG FEATURE CREATION
# ============================================================

def create_red_flag_features(df):

    """
    Create numerical red-flag features.
    """

    features = pd.DataFrame(
        index=df.index
    )


    # --------------------------------------------------------
    # Existing binary features
    # --------------------------------------------------------

    features["telecommuting"] = (

        pd.to_numeric(
            df["telecommuting"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)

    )


    features["has_company_logo"] = (

        pd.to_numeric(
            df["has_company_logo"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)

    )


    features["has_questions"] = (

        pd.to_numeric(
            df["has_questions"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)

    )


    # --------------------------------------------------------
    # Missing information indicators
    # --------------------------------------------------------

    missing_marker = "[not_provided]"


    optional_columns = [

        "company_profile",
        "salary_range",
        "benefits",
        "requirements",
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function",
        "department",

    ]


    for column in optional_columns:

        if column in df.columns:

            features[
                f"missing_{column}"
            ] = (

                df[column]
                .astype(str)
                .str.lower()
                .eq(missing_marker)
                .astype(int)

            )


    # --------------------------------------------------------
    # Text features
    # --------------------------------------------------------

    text = (

        df["combined_text"]
        .fillna("")
        .astype(str)

    )


    features["text_length"] = (

        text.str.len()

    )


    features["word_count"] = (

        text.str.split()
        .str.len()

    )


    features["exclamation_count"] = (

        text.str.count("!")

    )


    features["question_mark_count"] = (

        text.str.count(r"\?")

    )


    features["dollar_sign_count"] = (

        text.str.count(r"\$")

    )


    features["percentage_symbol_count"] = (

        text.str.count("%")

    )


    features["email_count"] = (

        text.str.count(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )

    )


    features["url_count"] = (

        text.str.count(
            r"https?://|www\."
        )

    )


    features["phone_number_count"] = (

        text.str.count(
            r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
        )

    )


    # --------------------------------------------------------
    # Suspicious keywords
    # --------------------------------------------------------

    features["suspicious_keyword_count"] = text.apply(

        lambda value:

        count_keyword_occurrences(
            value,
            SUSPICIOUS_KEYWORDS
        )

    )


    features["contains_suspicious_keyword"] = (

        features["suspicious_keyword_count"] > 0

    ).astype(int)


    features["very_short_posting"] = (

        features["word_count"] < 50

    ).astype(int)


    features["long_posting"] = (

        features["word_count"] > 1000

    ).astype(int)


    # --------------------------------------------------------
    # Clean numerical values
    # --------------------------------------------------------

    features = features.replace(
        [np.inf, -np.inf],
        0
    )


    features = features.fillna(0)


    return features


# ============================================================
# MAIN
# ============================================================

print("=" * 60)

print("FEATURE ENGINEERING")

print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n[1] Loading cleaned dataset...")


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)


print(
    "PASS: Dataset loaded successfully."
)

print(
    "Shape:",
    df.shape
)


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

print("\n[2] Checking required columns...")


required_columns = [

    "combined_text",
    "fraudulent",

    "telecommuting",
    "has_company_logo",
    "has_questions",

    "company_profile",
    "salary_range",
    "benefits",
    "requirements",

    "employment_type",
    "required_experience",
    "required_education",

    "industry",
    "function",
    "department",

]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


print(
    "PASS: All required columns are present."
)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

print("\n[3] Separating features and target...")


X = df.drop(
    columns=["fraudulent"]
)


y = df["fraudulent"].astype(int)


print(
    "Feature rows:",
    len(X)
)


print(
    "Target rows:",
    len(y)
)


print(
    "\nTarget distribution:"
)


print(
    y.value_counts()
)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\n[4] Creating train/test split...")


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y

)


print(
    "Training rows:",
    len(X_train)
)


print(
    "Testing rows :",
    len(X_test)
)


print(
    "\nTraining target distribution:"
)


print(
    y_train.value_counts()
)


print(
    "\nTesting target distribution:"
)


print(
    y_test.value_counts()
)


# ============================================================
# 5. RED-FLAG FEATURES
# ============================================================

print("\n[5] Creating red-flag features...")


X_train_red_flags = create_red_flag_features(
    X_train
)


X_test_red_flags = create_red_flag_features(
    X_test
)


red_flag_columns = list(
    X_train_red_flags.columns
)


print(
    "Training red-flag feature shape:",
    X_train_red_flags.shape
)


print(
    "Testing red-flag feature shape:",
    X_test_red_flags.shape
)


print(
    "\nRed-flag features:"
)


print(
    red_flag_columns
)


# ============================================================
# 6. TF-IDF
# ============================================================

print("\n[6] Creating TF-IDF features...")


tfidf = TfidfVectorizer(

    max_features=MAX_FEATURES,

    ngram_range=NGRAM_RANGE,

    min_df=MIN_DF,

    sublinear_tf=True

)


print(
    "Fitting TF-IDF on training data..."
)


X_train_text = tfidf.fit_transform(

    X_train["combined_text"]

)


print(
    "Transforming test data..."
)


X_test_text = tfidf.transform(

    X_test["combined_text"]

)


tfidf_feature_names = list(

    tfidf.get_feature_names_out()

)


print(
    "Training TF-IDF shape:",
    X_train_text.shape
)


print(
    "Testing TF-IDF shape:",
    X_test_text.shape
)


print(
    "TF-IDF vocabulary size:",
    len(tfidf_feature_names)
)


# ============================================================
# 7. STRUCTURED FEATURES
# ============================================================

print("\n[7] Preparing structured features...")


X_train_structured = csr_matrix(

    X_train_red_flags.values.astype(float)

)


X_test_structured = csr_matrix(

    X_test_red_flags.values.astype(float)

)


print(
    "Training structured feature shape:",
    X_train_structured.shape
)


print(
    "Testing structured feature shape:",
    X_test_structured.shape
)


# ============================================================
# 8. COMBINE FEATURES
# ============================================================

print("\n[8] Combining TF-IDF + red-flag features...")


X_train_combined = hstack(

    [
        X_train_text,
        X_train_structured
    ],

    format="csr"

)


X_test_combined = hstack(

    [
        X_test_text,
        X_test_structured
    ],

    format="csr"

)


print(
    "Combined training feature shape:",
    X_train_combined.shape
)


print(
    "Combined testing feature shape:",
    X_test_combined.shape
)


# ============================================================
# 9. CREATE COMBINED FEATURE NAMES
# ============================================================

print("\n[9] Creating combined feature names...")


combined_feature_names = (

    tfidf_feature_names
    +
    red_flag_columns

)


print(
    "TF-IDF feature names:",
    len(tfidf_feature_names)
)


print(
    "Red-flag feature names:",
    len(red_flag_columns)
)


print(
    "Combined feature names:",
    len(combined_feature_names)
)


if len(combined_feature_names) != X_train_combined.shape[1]:

    raise ValueError(

        "Feature-name count does not match "
        "combined feature count."

    )


print(
    "PASS: Feature names match feature matrix."
)


# ============================================================
# 10. FEATURE / TARGET ALIGNMENT
# ============================================================

print("\n[10] Checking feature and target alignment...")


if X_train_combined.shape[0] != len(y_train):

    raise ValueError(
        "Training feature and target rows do not match."
    )


if X_test_combined.shape[0] != len(y_test):

    raise ValueError(
        "Testing feature and target rows do not match."
    )


print(
    "PASS: Training features and target rows match."
)


print(
    "PASS: Testing features and target rows match."
)


# ============================================================
# 11. SAMPLE RED-FLAG FEATURES
# ============================================================

print("\n[11] Sample red-flag feature values...")


print(
    X_train_red_flags.head()
)


# ============================================================
# 12. SAVE FEATURE DATA WITH JOBLIB
# ============================================================

print("\n[12] Saving feature-engineering artifacts...")


os.makedirs(

    OUTPUT_DIR,

    exist_ok=True

)


feature_data = {

    # --------------------------------------------------------
    # Training / testing matrices
    # --------------------------------------------------------

    "X_train":
        X_train_combined,

    "X_test":
        X_test_combined,

    "y_train":
        y_train,

    "y_test":
        y_test,


    # --------------------------------------------------------
    # Exact TF-IDF vectorizer
    # --------------------------------------------------------

    "tfidf":
        tfidf,


    # --------------------------------------------------------
    # TF-IDF feature names
    # --------------------------------------------------------

    "tfidf_feature_names":
        tfidf_feature_names,


    # --------------------------------------------------------
    # Red-flag feature names
    # --------------------------------------------------------

    "red_flag_columns":
        red_flag_columns,


    # --------------------------------------------------------
    # Complete feature names
    # --------------------------------------------------------

    "combined_feature_names":
        combined_feature_names,


    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    "max_features":
        MAX_FEATURES,

    "ngram_range":
        NGRAM_RANGE,

    "min_df":
        MIN_DF,

    "test_size":
        TEST_SIZE,

    "random_state":
        RANDOM_STATE,

}


OUTPUT_FILE = os.path.join(

    OUTPUT_DIR,

    "feature_data.pkl"

)


joblib.dump(

    feature_data,

    OUTPUT_FILE

)


print(
    "PASS: Feature data saved successfully."
)


print(
    "File:",
    OUTPUT_FILE
)


# ============================================================
# 13. SAVE TF-IDF VECTORIZER SEPARATELY
# ============================================================

print("\n[13] Saving TF-IDF vectorizer...")


TFIDF_FILE = os.path.join(

    OUTPUT_DIR,

    "tfidf_vectorizer.joblib"

)


joblib.dump(

    tfidf,

    TFIDF_FILE

)


print(
    "PASS: TF-IDF vectorizer saved."
)


print(
    "File:",
    TFIDF_FILE
)


# ============================================================
# 14. SAVE FEATURE NAMES SEPARATELY
# ============================================================

print("\n[14] Saving feature names...")


FEATURE_NAMES_FILE = os.path.join(

    OUTPUT_DIR,

    "feature_names.joblib"

)


feature_names_data = {

    "tfidf_feature_names":
        tfidf_feature_names,

    "red_flag_columns":
        red_flag_columns,

    "combined_feature_names":
        combined_feature_names,

}


joblib.dump(

    feature_names_data,

    FEATURE_NAMES_FILE

)


print(
    "PASS: Feature names saved."
)


print(
    "File:",
    FEATURE_NAMES_FILE
)


# ============================================================
# 15. VERIFY SAVED ARTIFACTS
# ============================================================

print("\n[15] Verifying saved artifacts...")


loaded_feature_data = joblib.load(

    OUTPUT_FILE

)


loaded_tfidf = joblib.load(

    TFIDF_FILE

)


loaded_feature_names = joblib.load(

    FEATURE_NAMES_FILE

)


print(
    "PASS: feature_data loaded."
)


print(
    "PASS: TF-IDF vectorizer loaded."
)


print(
    "PASS: feature names loaded."
)


# ------------------------------------------------------------
# Verify vectorizer
# ------------------------------------------------------------

if len(
    loaded_tfidf.get_feature_names_out()
) != MAX_FEATURES:

    print(
        "WARNING: TF-IDF vocabulary size differs "
        "from MAX_FEATURES because the dataset may "
        "contain fewer usable terms."
    )


# ------------------------------------------------------------
# Verify feature names
# ------------------------------------------------------------

if len(
    loaded_feature_names["combined_feature_names"]
) != X_test_combined.shape[1]:

    raise ValueError(

        "Saved feature-name count does not match "
        "test feature count."

    )


# ------------------------------------------------------------
# Verify matrix
# ------------------------------------------------------------

if (

    loaded_feature_data["X_test"].shape

    !=

    X_test_combined.shape

):

    raise ValueError(
        "Saved X_test shape does not match original."
    )


print(
    "PASS: Saved artifacts verified."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("FEATURE ENGINEERING SUMMARY")

print("=" * 60)


print(
    "Original dataset rows     :",
    len(df)
)


print(
    "Training rows             :",
    X_train_combined.shape[0]
)


print(
    "Testing rows              :",
    X_test_combined.shape[0]
)


print(
    "TF-IDF features           :",
    X_train_text.shape[1]
)


print(
    "Red-flag features         :",
    X_train_structured.shape[1]
)


print(
    "Total combined features   :",
    X_train_combined.shape[1]
)


print(
    "Combined feature names    :",
    len(combined_feature_names)
)


print(
    "Training target rows      :",
    len(y_train)
)


print(
    "Testing target rows       :",
    len(y_test)
)


print(
    "Genuine training samples  :",
    int((y_train == 0).sum())
)


print(
    "Fraudulent training samples:",
    int((y_train == 1).sum())
)


print(
    "Genuine testing samples   :",
    int((y_test == 0).sum())
)


print(
    "Fraudulent testing samples:",
    int((y_test == 1).sum())
)


print("\nSaved artifacts:")


print(
    "1. feature_data.pkl"
)


print(
    "2. tfidf_vectorizer.joblib"
)


print(
    "3. feature_names.joblib"
)


print("\nFeature pipeline:")


print("""
Cleaned Dataset
      |
      v
Train / Test Split
      |
      +--------------------+
      |                    |
      v                    v
combined_text       Red-Flag Features
      |                    |
      v                    |
    TF-IDF                 |
      |                    |
      +---------+----------+
                |
                v
        Combined Features
                |
                v
        Ready for SMOTE
                |
                v
          ML Training
""")


print("=" * 60)

print(
    "FEATURE ENGINEERING COMPLETED"
)

print(
    "STATUS: ARTIFACTS READY FOR PREDICTOR"
)

print("=" * 60)  