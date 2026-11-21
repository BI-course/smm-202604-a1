"""
model_service.py

Shared helper for the two demo apps (Gradio on Hugging Face, and Streamlit).
The file is IDENTICAL in both folders, so both apps give the same answers.

What it does:
  1. Loads our four saved models from disk ONCE (nothing is retrained).
  2. Checks the input, using the same rules as api.py.
  3. Gives one simple function per model, which returns a dictionary.

If the input is wrong, a function raises InputError with a clear message.
The apps catch it and show the message to the user.
"""

import math
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

# The four saved model files. They are the same files that api.py uses.
MODEL_FILES = {
    "regression": "elasticnet_regressor_for_rental_price.joblib",
    "classification": "logisticregression_classifier_for_employee_attrition.joblib",
    "clustering": "kmeans_store_performance_segmentation.joblib",
    "rules": "apriori_recommendation_rules.joblib",
}


# ----------------------------------------------------------------------
# Finding the model files
# ----------------------------------------------------------------------
def _find_model_dir():
    """
    Look for the model files in these places, in this order:
      1. next to this file               (Hugging Face Space: all files in one folder)
      2. a 'model' folder next to it
      3. a 'model' folder in the folder above (our team repo: model/ at the root)
      4. a 'model' folder where the app was started
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        here,
        os.path.join(here, "model"),
        os.path.join(os.path.dirname(here), "model"),
        os.path.join(os.getcwd(), "model"),
    ]
    for folder in candidates:
        if all(os.path.exists(os.path.join(folder, name)) for name in MODEL_FILES.values()):
            return folder
    raise FileNotFoundError(
        "Could not find the four .joblib model files. Looked in: " + ", ".join(candidates)
    )


@lru_cache(maxsize=1)
def load_models():
    """Load the four models once. lru_cache means later calls reuse the same objects."""
    folder = _find_model_dir()

    def load(key):
        return joblib.load(os.path.join(folder, MODEL_FILES[key]))

    clustering = load("clustering")
    rules = load("rules")
    return {
        "regression": load("regression"),
        "classification": load("classification"),
        "imputer": clustering["imputer"],
        "scaler": clustering["scaler"],
        "kmeans": clustering["kmeans"],
        "cluster_features": clustering["clustering_features"],
        "lookup": rules["recommendation_lookup"],
        "tracks": rules["track_lookup"],
    }


# ----------------------------------------------------------------------
# Allowed values (the same as in api.py)
# ----------------------------------------------------------------------
PROPERTY_TYPES = ["Apartment", "Bungalow", "Maisonette", "Studio", "Townhouse"]
NEIGHBORHOODS = ["Karen", "Kilimani", "Ngong Road", "Ruaka", "South B", "Westlands"]
FURNISHING_STATUSES = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]

DEPARTMENTS = ["Engineering", "Finance", "HR", "Operations", "Sales"]
JOB_ROLE_LEVELS = ["Junior", "Mid", "Senior", "Lead"]
EDUCATION_LEVELS = ["Diploma", "Bachelor's", "Master's", "PhD"]
PERFORMANCE_RATINGS = ["Low", "Average", "High", "Outstanding"]

# field name -> (smallest allowed, biggest allowed)
RENT_NUMBERS = {
    "property_age_years": (0, 150),
    "size_sqm": (1, 5000),
    "num_bedrooms": (0, 20),
    "num_bathrooms": (0, 20),
    "distance_to_cbd_km": (0, 200),
    "service_charge_kes": (0, 10_000_000),
    "garage_size_sqm": (0, 1000),
}
RENT_OPTIONAL = ("garage_size_sqm",)

ATTRITION_NUMBERS = {
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
}
ATTRITION_OPTIONAL = ("bonus_amount_kes",)

STORE_FIELDS = [
    "monthly_revenue_kes",
    "daily_foot_traffic",
    "avg_basket_size_kes",
    "revenue_per_sqm",
    "revenue_per_staff",
]


class InputError(ValueError):
    """The user's input is wrong. The message says what to fix."""


# Friendly names used in the error messages (the screens show these names too).
FIELD_LABELS = {
    "property_age_years": "Property age", "size_sqm": "Size",
    "num_bedrooms": "Bedrooms", "num_bathrooms": "Bathrooms",
    "distance_to_cbd_km": "Distance to the city centre",
    "service_charge_kes": "Service charge", "garage_size_sqm": "Garage size",
    "property_type": "Property type", "neighborhood": "Neighbourhood",
    "furnishing_status": "Furnishing",
    "department": "Department", "job_role_level": "Job level",
    "education_level": "Education", "performance_rating": "Performance rating",
    "years_at_company": "Years at the company",
    "years_since_last_promotion": "Years since last promotion",
    "monthly_salary_kes": "Monthly salary",
    "overtime_hours_per_month": "Overtime hours per month",
    "job_satisfaction_score": "Job satisfaction",
    "work_life_balance_score": "Work-life balance",
    "distance_from_home_km": "Distance from home",
    "num_companies_worked_previously": "Companies worked for before",
    "remote_work_days_per_week": "Remote work days per week",
    "bonus_amount_kes": "Bonus",
    "monthly_revenue_kes": "Monthly revenue", "daily_foot_traffic": "Daily foot traffic",
    "avg_basket_size_kes": "Average basket size",
    "revenue_per_sqm": "Revenue per square metre",
    "revenue_per_staff": "Revenue per staff member",
}


def _label(name):
    return FIELD_LABELS.get(name, name)


# ----------------------------------------------------------------------
# Checking the input
# ----------------------------------------------------------------------
def _is_number(value):
    # bool is excluded because True/False are not real numbers for our purpose
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _is_blank(value):
    return value is None or (isinstance(value, str) and value.strip() == "")


def _check(values, number_fields, choice_fields, optional=()):
    """Check every field. Returns a clean dictionary or raises InputError."""
    all_fields = list(number_fields) + list(choice_fields)

    # Step 1: required fields must be filled in
    missing = [
        name for name in all_fields
        if name not in optional and _is_blank(values.get(name))
    ]
    if missing:
        raise InputError("Please fill in: " + ", ".join(_label(n) for n in missing) + ".")

    clean = {}

    # Step 2: numbers must be numbers inside the allowed range
    for name, (low, high) in number_fields.items():
        value = values.get(name)
        if name in optional and _is_blank(value):
            clean[name] = np.nan  # the saved pipeline fills this in by itself
            continue
        if not _is_number(value):
            raise InputError(f"{_label(name)} must be a number.")
        if not low <= value <= high:
            raise InputError(f"{_label(name)} must be between {low:,} and {high:,}.")
        clean[name] = float(value)

    # Step 3: choices must be one of the categories the model knows
    for name, allowed in choice_fields.items():
        value = values.get(name)
        if value not in allowed:
            raise InputError(f"{_label(name)} must be one of: {', '.join(allowed)}.")
        clean[name] = value

    return clean


# ----------------------------------------------------------------------
# 1. Regression: rental price
# ----------------------------------------------------------------------
def predict_rental_price(values):
    models = load_models()
    clean = _check(
        values,
        RENT_NUMBERS,
        {
            "property_type": PROPERTY_TYPES,
            "neighborhood": NEIGHBORHOODS,
            "furnishing_status": FURNISHING_STATUSES,
        },
        optional=RENT_OPTIONAL,
    )
    predicted_log_price = models["regression"].predict(pd.DataFrame([clean]))[0]
    # The model was trained on log1p(price), so expm1 turns it back into KES
    price_kes = float(np.expm1(predicted_log_price))
    return {
        "predicted_rental_price_kes": round(price_kes, 2),
        "model": "ElasticNet Regressor",
    }


# ----------------------------------------------------------------------
# 2. Classification: employee attrition
# ----------------------------------------------------------------------
def predict_attrition(values):
    models = load_models()
    clean = _check(
        values,
        ATTRITION_NUMBERS,
        {
            "department": DEPARTMENTS,
            "job_role_level": JOB_ROLE_LEVELS,
            "education_level": EDUCATION_LEVELS,
            "performance_rating": PERFORMANCE_RATINGS,
        },
        optional=ATTRITION_OPTIONAL,
    )
    pipeline = models["classification"]
    row = pd.DataFrame([clean])
    predicted_class = str(pipeline.predict(row)[0])  # "Yes" or "No"
    probabilities = pipeline.predict_proba(row)[0]
    labels = pipeline.named_steps["model"].classes_
    by_class = {str(l): round(float(p), 4) for l, p in zip(labels, probabilities)}
    return {
        "predicted_attrition": predicted_class,
        "probability_of_leaving": by_class.get("Yes"),
        "class_probabilities": by_class,
        "model": "Logistic Regression (SMOTE used at training only)",
    }


# ----------------------------------------------------------------------
# 3. Clustering: store performance segment
# ----------------------------------------------------------------------
# Short names for the five clusters. They were chosen by reading the typical
# values below, and they only fit THIS saved K-Means model. If the model is
# retrained, the cluster numbers can change, so check them again.
SEGMENT_NAMES = {
    0: "Small, quiet stores",
    1: "Few shoppers, big baskets",
    2: "Mid-sized, steady stores",
    3: "Top-revenue, high-traffic stores",
    4: "Busy stores, small baskets",
}


def cluster_profiles():
    """
    Typical values for each cluster, worked out from the K-Means cluster centres.
    The centres live in 'log and scaled' space, so we undo both steps.
    The result describes each cluster in general; it is not a single store.
    """
    models = load_models()
    kmeans, scaler = models["kmeans"], models["scaler"]
    typical = np.expm1(kmeans.cluster_centers_ * scaler.scale_ + scaler.mean_)
    sizes = np.bincount(kmeans.labels_, minlength=kmeans.n_clusters)
    profiles = []
    for cluster_id in range(kmeans.n_clusters):
        row = {"cluster": cluster_id,
               "name": SEGMENT_NAMES.get(cluster_id, f"Cluster {cluster_id}"),
               "stores_in_training_data": int(sizes[cluster_id])}
        for feature, value in zip(models["cluster_features"], typical[cluster_id]):
            row[feature] = round(float(value), 1)
        profiles.append(row)
    return profiles


def predict_store_segment(values):
    models = load_models()
    features = models["cluster_features"]
    clean = _check(values, {name: (0, 1_000_000_000) for name in features}, {})

    # Keep the columns in the training order
    row = pd.DataFrame([clean])[features]

    # The same steps as in training: fill gaps -> log1p -> scale -> K-Means
    filled = models["imputer"].transform(row)
    scaled = models["scaler"].transform(np.log1p(filled))
    cluster_id = int(models["kmeans"].predict(scaled)[0])

    profile = next(p for p in cluster_profiles() if p["cluster"] == cluster_id)
    return {
        "assigned_cluster": cluster_id,
        "segment_name": profile["name"],
        "profile": profile,
        "model": f"K-Means (k={models['kmeans'].n_clusters}, log-transformed and standardized features)",
    }


# ----------------------------------------------------------------------
# 4. Association rules: music recommender
# ----------------------------------------------------------------------
def track_choices():
    """List of (label, track id) for drop-downs, for example ('M09 - Sunday Worship 2', 'M09')."""
    tracks = load_models()["tracks"]
    return [(f"{tid} - {row['item_name']} ({row['category']})", tid)
            for tid, row in tracks.sort_index().iterrows()]


def recommend_tracks(session_tracks, top_n=3):
    models = load_models()
    tracks = models["tracks"]

    # Step 1: a non-empty list of track ids
    if not isinstance(session_tracks, (list, tuple, set)) or len(session_tracks) == 0:
        raise InputError("Pick at least one track that was already played in the session.")
    if not all(isinstance(t, str) for t in session_tracks):
        raise InputError("Every track id must be text, such as 'M09'.")

    # Step 2: every track must exist (ignore spaces and capital letters)
    session = {t.strip().upper() for t in session_tracks}
    unknown = sorted(t for t in session if t not in tracks.index)
    if unknown:
        raise InputError("Unknown track id(s): " + ", ".join(unknown))

    # Step 3: top_n is a whole number from 1 to 10
    if _is_blank(top_n):
        top_n = 3
    if not _is_number(top_n) or float(top_n) != int(top_n) or not 1 <= int(top_n) <= 10:
        raise InputError("'top_n' must be a whole number between 1 and 10.")
    top_n = int(top_n)

    # Step 4: use the saved rules. A rule applies when ALL its antecedent tracks
    # are already in the session. For each track we keep its best confidence.
    candidates = {}
    for antecedent, consequents in models["lookup"].items():
        if antecedent.issubset(session):
            for track_id, confidence, lift in consequents:
                if track_id not in session:
                    if track_id not in candidates or confidence > candidates[track_id][0]:
                        candidates[track_id] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
    recommendations = [
        {
            "track_id": str(track_id),
            "item_name": tracks.loc[track_id, "item_name"],
            "category": tracks.loc[track_id, "category"],
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        }
        for track_id, (confidence, lift) in ranked
    ]
    return {
        "session_tracks": sorted(session),
        "recommendations": recommendations,
        "model": "Apriori association rules (cleaned, precomputed lookup)",
    }
