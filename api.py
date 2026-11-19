"""
api.py

A Flask API that serves FOUR machine learning models our group saved in the
earlier labs:

    1. Regression       -- predicts the rental price of a property (ElasticNet)
    2. Classification   -- predicts if an employee will leave (Logistic Regression)
    3. Clustering       -- assigns a store to a performance segment (K-Means)
    4. Association Rule -- recommends the next music tracks for a listening
       Mining              session (Apriori rules)

FIRST VERSION (Member 1): all four models are loaded, and the regression and
classification endpoints work. Member 2 adds the clustering and recommender
endpoints.

The models are loaded from the ./model folder ONCE when the server starts.
Nothing is retrained in this file.

Every endpoint checks the incoming data first. If something is wrong
(missing field, wrong type, value out of range, unknown category) it returns
HTTP 400 with a message that says exactly what is wrong.

Run locally with the Flask development server:

    python api.py
"""

import math
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# -----------------------------------------------------------------------
# CORS lets a web page on another port (for example VS Code Live Server on
# port 5500) call this API from the browser.
# -----------------------------------------------------------------------
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # VS Code Live Server extension
    "null",  # allows opening a frontend .html file directly (file://)
    # Pages hosted online: a Hugging Face Static Space, or a page on Render.
    # These two are patterns, so any Space or Render address matches.
    r"https://[a-zA-Z0-9.-]+\.hf\.space",
    r"https://[a-zA-Z0-9-]+\.onrender\.com",
]}})

# -----------------------------------------------------------------------
# Load the four saved models ONCE, when the server starts.
# BASE_DIR makes the paths work no matter which folder the server is started
# from (important for Docker and Gunicorn).
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# 1. Regression: a full pipeline (preprocessing + ElasticNet)
regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "elasticnet_regressor_for_rental_price.joblib")
)

# 2. Classification: a full pipeline (preprocessing + SMOTE + Logistic Regression)
classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "logisticregression_classifier_for_employee_attrition.joblib")
)

# 3. Clustering: a dictionary holding the imputer, scaler, K-Means model and
#    the list of features the model was trained on
clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "kmeans_store_performance_segmentation.joblib")
)
clustering_imputer = clustering_artifact["imputer"]
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
clustering_features = clustering_artifact["clustering_features"]

# 4. Association rules: a dictionary holding the precomputed recommendation
#    lookup and the track catalogue (track id -> name, genre)
arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib")
)
recommendation_lookup = arm_artifact["recommendation_lookup"]
track_lookup = arm_artifact["track_lookup"]

print("All four models have been loaded. The API is ready.")


# =======================================================================
# Helpers for checking the incoming data
# =======================================================================

class ValidationError(Exception):
    """Raised when the incoming data is wrong. The message goes to the client."""


def error_response(message, status_code=400):
    # Every error from this API has the same shape: {"error": "..."}
    return jsonify({"error": message}), status_code


def get_json_object():
    """Read the JSON body and make sure it is a JSON object like {"a": 1}."""
    payload = request.get_json(silent=True)  # silent=True: no crash on bad JSON
    if payload is None:
        raise ValidationError(
            "Send a valid JSON body with the header Content-Type: application/json."
        )
    if not isinstance(payload, dict):
        raise ValidationError("The request body must be a JSON object.")
    return payload


def is_number(value):
    # True for int and float. bool is excluded because True/False are not
    # real numbers for our purpose (in Python, bool is a kind of int).
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def check_fields(payload, number_fields, choice_fields, optional_numbers=()):
    """
    Check every field the model needs and return a clean dictionary.

    number_fields   : {"field name": (smallest allowed, biggest allowed)}
    choice_fields   : {"field name": [allowed values]}
    optional_numbers: number fields that may be left out or sent as null
                      (the saved pipeline fills them in by itself)
    """
    all_fields = list(number_fields) + list(choice_fields)

    # Step 1: which required fields are missing?
    missing = [
        name for name in all_fields
        if name not in optional_numbers and name not in payload
    ]
    if missing:
        raise ValidationError(f"Missing required field(s): {', '.join(missing)}")

    clean = {}

    # Step 2: number fields must be numbers inside the allowed range
    for name, (low, high) in number_fields.items():
        value = payload.get(name)
        if name in optional_numbers and value is None:
            clean[name] = np.nan  # the pipeline's imputer fills this in
            continue
        if not is_number(value):
            raise ValidationError(f"'{name}' must be a number.")
        if not low <= value <= high:
            raise ValidationError(f"'{name}' must be between {low} and {high}.")
        clean[name] = float(value)

    # Step 3: choice fields must be one of the categories the model knows
    for name, allowed in choice_fields.items():
        value = payload[name]
        if not isinstance(value, str) or value not in allowed:
            raise ValidationError(
                f"'{name}' must be one of: {', '.join(allowed)}."
            )
        clean[name] = value

    return clean


# =======================================================================
# Error handlers: make sure errors come back as JSON, not as HTML pages
# =======================================================================

@app.errorhandler(ValidationError)
def handle_validation_error(error):
    return error_response(str(error), 400)


@app.errorhandler(404)
def handle_not_found(error):
    return error_response("This URL does not exist.", 404)


@app.errorhandler(405)
def handle_method_not_allowed(error):
    return error_response("Wrong HTTP method. Use POST for predictions.", 405)


@app.errorhandler(500)
def handle_server_error(error):
    return error_response("Something went wrong on the server.", 500)


# A simple URL to check that the API is running.
# It starts with /api/ on purpose: Nginx only forwards /api/... URLs to Flask.
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# =======================================================================
# 1. REGRESSION -- predict the rental price of a property
# =======================================================================

# Allowed values the saved pipeline was trained on
PROPERTY_TYPES = ["Apartment", "Bungalow", "Maisonette", "Studio", "Townhouse"]
NEIGHBORHOODS = ["Karen", "Kilimani", "Ngong Road", "Ruaka", "South B", "Westlands"]
FURNISHING_STATUSES = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/rental-price-regressor/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"property_age_years": 5, "size_sqm": 90, "num_bedrooms": 2,
#             "num_bathrooms": 2, "distance_to_cbd_km": 6,
#             "service_charge_kes": 8000, "garage_size_sqm": null,
#             "property_type": "Apartment", "neighborhood": "Kilimani",
#             "furnishing_status": "Semi-Furnished"}'
#
# garage_size_sqm is optional: send null (or leave it out) if the property has
# no garage. The pipeline fills it in by itself.
@app.route("/api/v1/models/rental-price-regressor/predictions", methods=["POST"])
def predict_rental_price():
    payload = get_json_object()

    clean = check_fields(
        payload,
        number_fields={
            "property_age_years": (0, 150),
            "size_sqm": (1, 5000),
            "num_bedrooms": (0, 20),
            "num_bathrooms": (0, 20),
            "distance_to_cbd_km": (0, 200),
            "service_charge_kes": (0, 10_000_000),
            "garage_size_sqm": (0, 1000),
        },
        choice_fields={
            "property_type": PROPERTY_TYPES,
            "neighborhood": NEIGHBORHOODS,
            "furnishing_status": FURNISHING_STATUSES,
        },
        optional_numbers=("garage_size_sqm",),
    )

    # The saved pipeline does the encoding, imputing and scaling by itself
    input_row = pd.DataFrame([clean])
    predicted_log_price = regression_pipeline.predict(input_row)[0]

    # The model was trained on log1p(price), so we undo it with expm1
    # to get the price back in KES.
    predicted_price_kes = float(np.expm1(predicted_log_price))

    return jsonify({
        "predicted_rental_price_kes": round(predicted_price_kes, 2),
        "model": "ElasticNet Regressor",
    })


# =======================================================================
# 2. CLASSIFICATION -- predict if an employee will leave the company
# =======================================================================

DEPARTMENTS = ["Engineering", "Finance", "HR", "Operations", "Sales"]
JOB_ROLE_LEVELS = ["Junior", "Mid", "Senior", "Lead"]
EDUCATION_LEVELS = ["Diploma", "Bachelor's", "Master's", "PhD"]
PERFORMANCE_RATINGS = ["Low", "Average", "High", "Outstanding"]

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/employee-attrition-classifier/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"department": "Sales", "job_role_level": "Mid",
#             "education_level": "Bachelor'"'"'s", "performance_rating": "Average",
#             "years_at_company": 3, "years_since_last_promotion": 2,
#             "monthly_salary_kes": 85000, "overtime_hours_per_month": 20,
#             "job_satisfaction_score": 3, "work_life_balance_score": 3,
#             "distance_from_home_km": 12, "num_companies_worked_previously": 2,
#             "remote_work_days_per_week": 1, "bonus_amount_kes": null}'
#
# bonus_amount_kes is optional: send null (or leave it out) if there is no bonus.
@app.route("/api/v1/models/employee-attrition-classifier/predictions", methods=["POST"])
def predict_attrition():
    payload = get_json_object()

    clean = check_fields(
        payload,
        number_fields={
            "years_at_company": (0, 60),
            "years_since_last_promotion": (0, 60),
            "monthly_salary_kes": (0, 10_000_000),
            "overtime_hours_per_month": (0, 300),
            "job_satisfaction_score": (1, 5),
            "work_life_balance_score": (1, 5),
            "distance_from_home_km": (0, 500),
            "num_companies_worked_previously": (0, 30),
            "remote_work_days_per_week": (0, 7),
            "bonus_amount_kes": (0, 100_000_000),
        },
        choice_fields={
            "department": DEPARTMENTS,
            "job_role_level": JOB_ROLE_LEVELS,
            "education_level": EDUCATION_LEVELS,
            "performance_rating": PERFORMANCE_RATINGS,
        },
        optional_numbers=("bonus_amount_kes",),
    )

    input_row = pd.DataFrame([clean])

    predicted_class = str(classification_pipeline.predict(input_row)[0])  # "Yes" or "No"
    probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = classification_pipeline.named_steps["model"].classes_
    class_probabilities = {
        str(label): round(float(prob), 4)
        for label, prob in zip(class_labels, probabilities)
    }

    return jsonify({
        "predicted_attrition": predicted_class,
        "probability_of_leaving": class_probabilities.get("Yes"),
        "class_probabilities": class_probabilities,
        "model": "Logistic Regression (SMOTE used at training only)",
    })


# =======================================================================
# Entry point
# =======================================================================
if __name__ == "__main__":
    # Development server only. debug=True must NEVER be used on a public
    # server. In Docker and on Render, Gunicorn starts the app instead.
    app.run(debug=True)
