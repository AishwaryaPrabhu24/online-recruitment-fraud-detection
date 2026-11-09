
from flask import Flask, request, jsonify
from flask_cors import CORS

from predictor import load_artifacts, predict_job


# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

MAX_MESSAGE_LENGTH = 50000


# --------------------------------------------------
# Load your existing trained model and artifacts
# --------------------------------------------------

try:
    (
        feature_data,
        tfidf,
        feature_names,
        red_flag_columns,
        model,
    ) = load_artifacts()

    print("FraudShield backend initialized successfully.")

except Exception:
    app.logger.exception("Failed to load prediction artifacts.")
    raise


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "FraudShield prediction API is running.",
        "model": "XGBoost",
        "endpoint": "/predict",
        "method": "POST",
    }), 200


# --------------------------------------------------
# Analyze a complete job advertisement
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Please send the job message as JSON."
            }), 400

        # Accept the complete advertisement as one string.
        job_text = data.get("job_text", "")

        if not isinstance(job_text, str):
            return jsonify({
                "status": "error",
                "message": "job_text must be a string."
            }), 400

        job_text = job_text.strip()

        if not job_text:
            return jsonify({
                "status": "error",
                "message": "Please paste a job advertisement before analyzing."
            }), 400

        if len(job_text) > MAX_MESSAGE_LENGTH:
            return jsonify({
                "status": "error",
                "message": (
                    f"The message is too long. "
                    f"Maximum length is {MAX_MESSAGE_LENGTH} characters."
                )
            }), 413

        print("\n" + "=" * 60)
        print("RECEIVED COMPLETE JOB MESSAGE")
        print("=" * 60)
        print("Characters:", len(job_text))
        print("Words:", len(job_text.split()))

        # Preserve the existing predictor's input format.
        # The full advertisement is passed as description so
        # build_combined_text() and TF-IDF process the full text.
        job = {
            "title": "",
            "company_profile": "",
            "description": job_text,
            "requirements": "",
            "benefits": "",
            "salary_range": "",
            "employment_type": "",
            "required_experience": "",
            "required_education": "",
            "industry": "",
            "function": "",
            "department": "",
            "telecommuting": 0,
            "has_company_logo": 0,
            "has_questions": 0,
        }

        # Use the existing trained XGBoost model and feature pipeline.
        prediction_result = predict_job(
            job,
            tfidf,
            feature_names,
            red_flag_columns,
            model,
        )

        conclusion = prediction_result.get("final_conclusion")

        if not conclusion:
            raise ValueError(
                "The prediction pipeline did not return a final conclusion."
            )

        print("=" * 60)
        print("API PREDICTION COMPLETED")
        print("Label:", conclusion.get("label"))
        print(
            "Fraud probability:",
            conclusion.get("probability_fraud")
        )
        print("=" * 60)

        # Keep the response structure compatible with Results.jsx.
        return jsonify({
            "status": "success",
            "model": "XGBoost",
            "prediction": prediction_result,
        }), 200

    except Exception:
        app.logger.exception("Job prediction failed.")

        return jsonify({
            "status": "error",
            "message": (
                "The job could not be analyzed. "
                "Check the Flask terminal for the detailed error."
            )
        }), 500


# --------------------------------------------------
# Start the development server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
