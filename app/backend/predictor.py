import os

import re

import joblib

import numpy as np

import pandas as pd

import xgboost as xgb

from scipy.sparse import hstack, csr_matrix

# ============================================================

# PROJECT PATHS

# ============================================================



CURRENT_DIR = os.path.dirname(

    os.path.abspath(__file__)

)



PROJECT_ROOT = os.path.abspath(

    os.path.join(

        CURRENT_DIR,

        "..",

        ".."

    )

)





FEATURE_DIR = os.path.join(

    PROJECT_ROOT,

    "dataset",

    "processed",

    "features"

)



MODEL_DIR = os.path.join(

    PROJECT_ROOT,

    "dataset",

    "processed",

    "models"

)





FEATURE_DATA_FILE = os.path.join(

    FEATURE_DIR,

    "feature_data.pkl"

)



TFIDF_FILE = os.path.join(

    FEATURE_DIR,

    "tfidf_vectorizer.joblib"

)



FEATURE_NAMES_FILE = os.path.join(

    FEATURE_DIR,

    "feature_names.joblib"

)



MODEL_FILE = os.path.join(

    MODEL_DIR,

    "final_xgboost_model.pkl"

)





# ============================================================

# SUSPICIOUS KEYWORDS

# MUST MATCH feature_engineering.py

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

# HELPER

# ============================================================



def count_keyword_occurrences(text, keywords):



    text = str(text).lower()



    count = 0



    for keyword in keywords:



        count += text.count(keyword)



    return count





# ============================================================

# BUILD COMBINED TEXT

# MUST MATCH TRAINING PIPELINE

# ============================================================



def build_combined_text(job):



    fields = [



        job.get("title", ""),



        job.get(

            "company_profile",

            ""

        ),



        job.get(

            "description",

            ""

        ),



        job.get(

            "requirements",

            ""

        ),



        job.get(

            "benefits",

            ""

        ),



        job.get(

            "salary_range",

            ""

        ),



        job.get(

            "employment_type",

            ""

        ),



        job.get(

            "required_experience",

            ""

        ),



        job.get(

            "required_education",

            ""

        ),



        job.get(

            "industry",

            ""

        ),



        job.get(

            "function",

            ""

        ),



        job.get(

            "department",

            ""

        )



    ]



    text_parts = []



    for value in fields:



        if value is None:



            value = ""



        value = str(value).strip()



        if value:



            text_parts.append(value)



    return " ".join(text_parts)





# ============================================================

# RED-FLAG FEATURE CREATION

# EXACTLY 26 FEATURES

# ============================================================



def create_red_flag_features(job):



    # --------------------------------------------------------

    # Basic values

    # --------------------------------------------------------



    telecommuting = int(

        pd.to_numeric(

            job.get(

                "telecommuting",

                0

            ),

            errors="coerce"

        )

        if str(

            job.get(

                "telecommuting",

                0

            )

        ).strip()

        else 0

    )



    has_company_logo = int(

        pd.to_numeric(

            job.get(

                "has_company_logo",

                0

            ),

            errors="coerce"

        )

        if str(

            job.get(

                "has_company_logo",

                0

            )

        ).strip()

        else 0

    )



    has_questions = int(

        pd.to_numeric(

            job.get(

                "has_questions",

                0

            ),

            errors="coerce"

        )

        if str(

            job.get(

                "has_questions",

                0

            )

        ).strip()

        else 0

    )





    # --------------------------------------------------------

    # Text

    # --------------------------------------------------------



    combined_text = build_combined_text(

        job

    )



    text_lower = combined_text.lower()



    word_count = len(

        combined_text.split()

    )





    # --------------------------------------------------------

    # Missing information

    #

    # Training preprocessing uses:

    # [not_provided]

    # --------------------------------------------------------



    def is_missing(value):



        if value is None:



            return 1



        value = str(value).strip().lower()



        return int(

            value == "[not_provided]"

            or value == ""

        )





    missing_company_profile = is_missing(

        job.get(

            "company_profile",

            ""

        )

    )



    missing_salary_range = is_missing(

        job.get(

            "salary_range",

            ""

        )

    )



    missing_benefits = is_missing(

        job.get(

            "benefits",

            ""

        )

    )



    missing_requirements = is_missing(

        job.get(

            "requirements",

            ""

        )

    )



    missing_employment_type = is_missing(

        job.get(

            "employment_type",

            ""

        )

    )



    missing_required_experience = is_missing(

        job.get(

            "required_experience",

            ""

        )

    )



    missing_required_education = is_missing(

        job.get(

            "required_education",

            ""

        )

    )



    missing_industry = is_missing(

        job.get(

            "industry",

            ""

        )

    )



    missing_function = is_missing(

        job.get(

            "function",

            ""

        )

    )



    missing_department = is_missing(

        job.get(

            "department",

            ""

        )

    )





    # --------------------------------------------------------

    # Text statistics

    # --------------------------------------------------------



    text_length = len(

        combined_text

    )



    exclamation_count = combined_text.count(

        "!"

    )



    question_mark_count = combined_text.count(

        "?"

    )



    dollar_sign_count = combined_text.count(

        "$"

    )



    percentage_symbol_count = combined_text.count(

        "%"

    )





    email_count = len(

        re.findall(

            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\b",

            combined_text

        )

    )





    url_count = len(

        re.findall(

            r"https?://|www\\.",

            combined_text,

            flags=re.IGNORECASE

        )

    )





    phone_number_count = len(

        re.findall(

            r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",

            combined_text

        )

    )





    # --------------------------------------------------------

    # Suspicious keywords

    # --------------------------------------------------------



    suspicious_keyword_count = count_keyword_occurrences(

        combined_text,

        SUSPICIOUS_KEYWORDS

    )



    contains_suspicious_keyword = int(

        suspicious_keyword_count > 0

    )





    # --------------------------------------------------------

    # Posting length

    # MUST MATCH TRAINING

    # --------------------------------------------------------



    very_short_posting = int(

        word_count < 50

    )



    long_posting = int(

        word_count > 1000

    )





    # --------------------------------------------------------

    # EXACT FEATURE ORDER

    # --------------------------------------------------------



    feature_names = [



        "telecommuting",

        "has_company_logo",

        "has_questions",



        "missing_company_profile",

        "missing_salary_range",

        "missing_benefits",

        "missing_requirements",

        "missing_employment_type",

        "missing_required_experience",

        "missing_required_education",

        "missing_industry",

        "missing_function",

        "missing_department",



        "text_length",

        "word_count",

        "exclamation_count",

        "question_mark_count",

        "dollar_sign_count",

        "percentage_symbol_count",

        "email_count",

        "url_count",

        "phone_number_count",



        "suspicious_keyword_count",

        "contains_suspicious_keyword",



        "very_short_posting",

        "long_posting"



    ]





    values = [



        telecommuting,

        has_company_logo,

        has_questions,



        missing_company_profile,

        missing_salary_range,

        missing_benefits,

        missing_requirements,

        missing_employment_type,

        missing_required_experience,

        missing_required_education,

        missing_industry,

        missing_function,

        missing_department,



        text_length,

        word_count,

        exclamation_count,

        question_mark_count,

        dollar_sign_count,

        percentage_symbol_count,

        email_count,

        url_count,

        phone_number_count,



        suspicious_keyword_count,

        contains_suspicious_keyword,



        very_short_posting,

        long_posting



    ]





    features = pd.DataFrame(

        [values],

        columns=feature_names

    )





    features = features.replace(

        [np.inf, -np.inf],

        0

    )



    features = features.fillna(0)





    return features





# ============================================================

# LOAD ARTIFACTS

# REQUIRED BY app.py

# ============================================================



def load_artifacts():



    print("=" * 70)

    print("LOADING FINAL XGBOOST ARTIFACTS")

    print("=" * 70)



    print("\nProject root:")

    print(PROJECT_ROOT)



    print("\nFeature directory:")

    print(FEATURE_DIR)



    print("\nModel directory:")

    print(MODEL_DIR)





    # --------------------------------------------------------

    # Check files

    # --------------------------------------------------------



    print("\n[1] Checking required files...")



    required_files = [



        FEATURE_DATA_FILE,

        TFIDF_FILE,

        FEATURE_NAMES_FILE,

        MODEL_FILE



    ]





    for file_path in required_files:



        if not os.path.exists(file_path):



            raise FileNotFoundError(

                f"Required file not found: {file_path}"

            )



        print(

            "PASS:",

            file_path

        )





    # --------------------------------------------------------

    # Feature data

    # --------------------------------------------------------



    print("\n[2] Loading feature data...")



    feature_data = joblib.load(

        FEATURE_DATA_FILE

    )



    print(

        "PASS: Feature data loaded."

    )



    print("\nAvailable keys:")



    for key in feature_data.keys():



        print(

            " -",

            key

        )





    # --------------------------------------------------------

    # TF-IDF

    # --------------------------------------------------------



    print("\n[3] Loading TF-IDF vectorizer...")



    tfidf = joblib.load(

        TFIDF_FILE

    )



    print(

        "PASS: TF-IDF vectorizer loaded."

    )



    print(

        "TF-IDF vocabulary size:",

        len(

            tfidf.get_feature_names_out()

        )

    )





    # --------------------------------------------------------

    # Feature names

    # --------------------------------------------------------



    print("\n[4] Loading feature names...")



    feature_names_data = joblib.load(

        FEATURE_NAMES_FILE

    )





    tfidf_feature_names = feature_names_data[

        "tfidf_feature_names"

    ]



    red_flag_columns = feature_names_data[

        "red_flag_columns"

    ]



    combined_feature_names = feature_names_data[

        "combined_feature_names"

    ]





    print(

        "PASS: Combined feature names loaded."

    )



    print(

        "TF-IDF features:",

        len(tfidf_feature_names)

    )



    print(

        "Red-flag feature count:",

        len(red_flag_columns)

    )



    print(

        "Total feature count:",

        len(combined_feature_names)

    )





    if len(red_flag_columns) != 26:



        raise ValueError(

            "Expected 26 red-flag features."

        )





    if len(combined_feature_names) != 10026:



        raise ValueError(

            "Expected 10026 combined features."

        )





    # --------------------------------------------------------

    # Model

    # --------------------------------------------------------



    print("\n[5] Loading final XGBoost model...")



    model = joblib.load(

        MODEL_FILE

    )



    print(

        "PASS: Final XGBoost model loaded."

    )



    print(

        "Model type:",

        type(model).__name__

    )





    model_features = getattr(

        model,

        "n_features_in_",

        None

    )





    print(

        "Model expected features:",

        model_features

    )





    if model_features != len(

        combined_feature_names

    ):



        raise ValueError(

            "Model feature count does not match "

            "combined feature count."

        )





    print(

        "PASS: Final XGBoost feature dimensions verified."

    )





    print("\n" + "=" * 70)

    print("FINAL XGBOOST ARTIFACTS LOADED")

    print("=" * 70)



    print(

        "TF-IDF features   :",

        len(tfidf_feature_names)

    )



    print(

        "Red-flag features :",

        len(red_flag_columns)

    )



    print(

        "Total features    :",

        len(combined_feature_names)

    )



    print(

        "Production model  :",

        type(model).__name__

    )



    print("=" * 70)





    return (

        feature_data,

        tfidf,

        combined_feature_names,

        red_flag_columns,

        model

    )





# ============================================================

# TOP WORD ANALYSIS

# ============================================================



def get_top_words(

    text,

    tfidf,

    model,

    prediction,

    top_n=10

):



    try:



        vector = tfidf.transform(

            [text]

        )



        feature_names = (

            tfidf.get_feature_names_out()

        )



        non_zero_indices = (

            vector.nonzero()[1]

        )





        if len(non_zero_indices) == 0:



            return []





        # ----------------------------------------------------

        # XGBoost global feature importance

        # ----------------------------------------------------



        importances = model.feature_importances_



        results = []





        for index in non_zero_indices:



            if index >= len(importances):



                continue





            importance = float(

                importances[index]

                * abs(

                    vector[0, index]

                )

            )





            if importance <= 0:



                continue





            results.append({



                "word":

                    str(

                        feature_names[index]

                    ),



                "importance":

                    importance,



                "direction":

                    "fraud"

                    if prediction == 1

                    else "legitimate"



            })





        results.sort(

            key=lambda item:

                item["importance"],

            reverse=True

        )





        return results[:top_n]





    except Exception as error:



        print(

            "WARNING: Top-word analysis failed:",

            str(error)

        )



        return []





# ============================================================

# PREDICT JOB

# REQUIRED BY app.py

# ============================================================



def predict_job(

    job,

    tfidf,

    feature_names,

    red_flag_columns,

    model

):



    print("\n" + "=" * 70)

    print("GENERATING FINAL XGBOOST PREDICTION")

    print("=" * 70)





    # --------------------------------------------------------

    # Combined text

    # --------------------------------------------------------



    combined_text = build_combined_text(

        job

    )





    print("\nCreating prediction features...")





    # --------------------------------------------------------

    # TF-IDF

    # --------------------------------------------------------



    tfidf_features = tfidf.transform(

        [combined_text]

    )





    print(

        "TF-IDF feature shape:",

        tfidf_features.shape

    )





    # --------------------------------------------------------

    # Red flags

    # --------------------------------------------------------



    red_flags_df = create_red_flag_features(

        job

    )





    # Ensure exact order

    red_flags_df = red_flags_df[

        red_flag_columns

    ]





    red_flag_features = csr_matrix(

        red_flags_df.values.astype(float)

    )





    print(

        "Red-flag feature shape:",

        red_flag_features.shape

    )





    # --------------------------------------------------------

    # Combine

    # --------------------------------------------------------



    combined_features = hstack(

        [

            tfidf_features,

            red_flag_features

        ],

        format="csr"

    )





    print(

        "Combined feature shape:",

        combined_features.shape

    )





    # --------------------------------------------------------

    # Dimension verification

    # --------------------------------------------------------



    expected_features = len(

        feature_names

    )



    actual_features = (

        combined_features.shape[1]

    )





    if actual_features != expected_features:



        raise ValueError(

            f"Feature dimension mismatch. "

            f"Expected {expected_features}, "

            f"got {actual_features}."

        )





    model_features = getattr(

        model,

        "n_features_in_",

        None

    )





    if model_features != actual_features:



        raise ValueError(

            f"Model expects {model_features} "

            f"features but input contains "

            f"{actual_features}."

        )





    print(

        "PASS: Prediction feature dimensions verified."

    )





    # --------------------------------------------------------

    # Prediction

    # --------------------------------------------------------



    print(

        "\nPredicting with final XGBoost model..."

    )





    prediction = int(

        model.predict(

            combined_features

        )[0]

    )





    probabilities = model.predict_proba(

        combined_features

    )[0]





    probability_legitimate = float(

        probabilities[0]

    )



    probability_fraud = float(

        probabilities[1]

    )





    label = (

        "Fraudulent"

        if prediction == 1

        else "Legitimate"

    )





    # --------------------------------------------------------

    # Red flags

    # --------------------------------------------------------



    # --------------------------------------------------------
    # Red flags - reporting only
    # Keep reporting separate from model feature extraction.
    # --------------------------------------------------------

    red_flag_values = red_flags_df.iloc[0]
    detected_red_flags = []

    # Stronger warning indicators. These are still not proof of fraud.
    high_risk_features = {
        "suspicious_keyword_count",
    }

    # Missing company/job details are warnings, not proof of fraud.
    medium_risk_features = {
        "missing_company_profile",
        "missing_salary_range",
        "missing_benefits",
        "missing_requirements",
    }

    # Lower-priority completeness/length warnings.
    low_risk_features = {
        "missing_employment_type",
        "missing_required_experience",
        "missing_required_education",
        "missing_industry",
        "missing_function",
        "missing_department",
        "very_short_posting",
        "long_posting",
    }

    for column in red_flag_columns:
        if column not in (
            high_risk_features
            | medium_risk_features
            | low_risk_features
        ):
            # Ignore descriptive statistics such as text_length,
            # word_count and dollar_sign_count in the red-flag list.
            continue

        value = float(red_flag_values.get(column, 0))

        if value <= 0:
            continue

        if column in high_risk_features:
            level = "high"
        elif column in medium_risk_features:
            level = "medium"
        else:
            level = "low"

        detected_red_flags.append({
            "feature": column,
            "value": value,
            "level": level,
        })

    high_flags = sum(
        1 for item in detected_red_flags
        if item["level"] == "high"
    )
    medium_flags = sum(
        1 for item in detected_red_flags
        if item["level"] == "medium"
    )
    low_flags = sum(
        1 for item in detected_red_flags
        if item["level"] == "low"
    )

    # Top words

    # --------------------------------------------------------



    top_words = get_top_words(



        combined_text,



        tfidf,



        model,



        prediction,



        top_n=10



    )





    # --------------------------------------------------------

    # Console output

    # --------------------------------------------------------



    print("\n" + "=" * 70)

    print("FINAL XGBOOST PREDICTION")

    print("=" * 70)



    print(

        "Prediction:",

        label

    )



    print(

        "Fraud probability:",

        f"{probability_fraud * 100:.2f}%"

    )



    print(

        "Legitimate probability:",

        f"{probability_legitimate * 100:.2f}%"

    )





    print("\n" + "=" * 70)

    print("DETECTED RED FLAGS")

    print("=" * 70)





    if not detected_red_flags:



        print(

            "No configured red flags detected."

        )



    else:



        for flag in detected_red_flags:



            print(

                "-",

                flag["feature"],

                ":",

                flag["value"],

                f'[{flag["level"]}]'

            )





    print("\n" + "=" * 70)

    print("TOP INFLUENTIAL WORDS")

    print("=" * 70)





    if not top_words:



        print(

            "No influential words found."

        )



    else:



        for item in top_words:



            print(

                "-",

                item["word"],

                ":",

                f"{item['importance']:.8f}",

                f"({item['direction']})"

            )





    # --------------------------------------------------------

    # Final conclusion

    # --------------------------------------------------------



    if prediction == 1:



        conclusion_message = (

            "The selected XGBoost model classifies "

            "this job posting as likely fraudulent."

        )



    else:



        conclusion_message = (

            "The selected XGBoost model classifies "

            "this job posting as likely legitimate."

        )





    print("\n" + "=" * 70)

    print("FINAL CONCLUSION")

    print("=" * 70)



    print(

        "Prediction:",

        label

    )



    print(

        conclusion_message

    )





    print("=" * 70)

    print("PREDICTION COMPLETED")

    print("STATUS: SUCCESS")

    print("MODEL: XGBoost")

    print("=" * 70)





    # --------------------------------------------------------

    # API RESULT

    # --------------------------------------------------------



    return {



        "model": {



            "name":

                "XGBoost",



            "prediction":

                prediction,



            "probability_fraud":

                probability_fraud,



            "probability_legitimate":

                probability_legitimate,



            "top_words":

                top_words



        },





        "red_flags":

            detected_red_flags,





        "red_flag_summary": {



            "high":

                high_flags,



            "medium":

                medium_flags,



            "low":

                low_flags,



            "total_detected":

                len(

                    detected_red_flags

                )



        },





        "top_words":

            top_words,





        "final_conclusion": {



            "prediction":

                prediction,



            "label":

                label,



            "probability_fraud":

                probability_fraud,



            "probability_legitimate":

                probability_legitimate,



            "message":

                conclusion_message



        }



    }





# ============================================================

# DIRECT TEST

# ============================================================



if __name__ == "__main__":



    print("=" * 70)

    print("JOB FRAUD DETECTION - FINAL XGBOOST PREDICTOR")

    print("=" * 70)





    (

        feature_data,

        tfidf,

        feature_names,

        red_flag_columns,

        model

    ) = load_artifacts()





    # --------------------------------------------------------

    # Sample job

    # --------------------------------------------------------



    sample_job = {



        "title":

            "Software Developer",



        "company_profile":

            "We are a technology company developing "

            "software solutions for businesses.",



        "description":

            "We are looking for a software developer "

            "to join our engineering team. The candidate "

            "will work on web applications and backend services.",



        "requirements":

            "Experience with JavaScript, React, Node.js "

            "and database technologies.",



        "benefits":

            "Competitive salary, health insurance and "

            "professional development opportunities.",



        "salary_range":

            "$60000-$80000",



        "employment_type":

            "Full-time",



        "required_experience":

            "2 years",



        "required_education":

            "Bachelor's degree",



        "industry":

            "Information Technology",



        "function":

            "Engineering",



        "department":

            "Software Development",



        "telecommuting":

            0,



        "has_company_logo":

            1,



        "has_questions":

            1



    }





    print("\n" + "=" * 70)

    print("SAMPLE JOB POSTING")

    print("=" * 70)



    print(

        "Title:",

        sample_job["title"]

    )



    print(

        "Company profile:",

        sample_job["company_profile"]

    )



    print(

        "Description:",

        sample_job["description"]

    )



    print(

        "Requirements:",

        sample_job["requirements"]

    )



    print(

        "Benefits:",

        sample_job["benefits"]

    )





    predict_job(



        sample_job,



        tfidf,



        feature_names,



        red_flag_columns,



        model



    )