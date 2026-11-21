"""
app.py  --  Streamlit app for Streamlit Community Cloud

One app with FOUR tabs, one tab per model:
    1. Rental price       (regression)
    2. Employee attrition (classification)
    3. Store segment      (clustering)
    4. Music recommender  (association rules)

The models are loaded from the same .joblib files that api.py uses.
All the loading, input checking and predicting is done in model_service.py,
so this file only builds the screen.

Run locally from the ROOT of the repository (not from inside this folder):
    streamlit run streamlit-sharing-using-streamlit/app.py
"""

import pandas as pd
import streamlit as st

import model_service as ms

st.set_page_config(page_title="Group A: model validation demos", layout="wide")


# Streamlit runs this whole file again on every click. Python only imports
# model_service once, and load_models() remembers its result, so the four model
# files are read from disk only the first time. This line just makes that first
# read happen now, behind a spinner.
@st.cache_resource(show_spinner="Loading the four saved models...")
def get_models():
    return ms.load_models()


get_models()

st.title("Group A: model validation demos")
st.write("Choose a tab, change the values if you like, and press the button. "
         "The four models are the same saved files that our API uses.")

tab_rent, tab_attrition, tab_store, tab_music = st.tabs(
    ["Rental price", "Employee attrition", "Store segment", "Music recommender"])


def optional_number(label, key, minimum, maximum):
    """A number box that can be left empty (the saved pipeline fills it in)."""
    return st.number_input(label, min_value=float(minimum), max_value=float(maximum),
                           value=None, step=1.0, key=key)


# -----------------------------------------------------------------------
# 1. Regression: rental price
# -----------------------------------------------------------------------
with tab_rent:
    with st.form("rent_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Property age (years)", 0.0, 150.0, 5.0, 1.0, key="rent_age")
            size = st.number_input("Size (square metres)", 1.0, 5000.0, 90.0, 1.0, key="rent_size")
            beds = st.number_input("Bedrooms", 0.0, 20.0, 2.0, 1.0, key="rent_beds")
            baths = st.number_input("Bathrooms", 0.0, 20.0, 2.0, 1.0, key="rent_baths")
            cbd = st.number_input("Distance to the city centre (km)", 0.0, 200.0, 6.0, 0.5,
                                  key="rent_cbd")
        with right:
            service = st.number_input("Service charge (KES per month)", 0.0, 10_000_000.0, 8000.0,
                                      500.0, key="rent_service")
            garage = optional_number("Garage size (square metres), leave empty if none",
                                     "rent_garage", 0, 1000)
            prop_type = st.selectbox("Property type", ms.PROPERTY_TYPES, key="rent_type")
            area = st.selectbox("Neighbourhood", ms.NEIGHBORHOODS, index=1, key="rent_area")
            furnish = st.selectbox("Furnishing", ms.FURNISHING_STATUSES, index=1, key="rent_furnish")
        rent_submit = st.form_submit_button("Predict rent", type="primary")

    if rent_submit:
        try:
            result = ms.predict_rental_price({
                "property_age_years": age, "size_sqm": size, "num_bedrooms": beds,
                "num_bathrooms": baths, "distance_to_cbd_km": cbd,
                "service_charge_kes": service, "garage_size_sqm": garage,
                "property_type": prop_type, "neighborhood": area, "furnishing_status": furnish,
            })
            st.metric("Predicted rent per month",
                      f"KES {result['predicted_rental_price_kes']:,.0f}")
            st.caption(f"Model: {result['model']}")
        except ms.InputError as error:
            st.error(f"Check your input: {error}")

# -----------------------------------------------------------------------
# 2. Classification: employee attrition
# -----------------------------------------------------------------------
with tab_attrition:
    with st.form("attrition_form"):
        left, right = st.columns(2)
        with left:
            dept = st.selectbox("Department", ms.DEPARTMENTS, index=4, key="att_dept")
            level = st.selectbox("Job level", ms.JOB_ROLE_LEVELS, index=1, key="att_level")
            edu = st.selectbox("Education", ms.EDUCATION_LEVELS, index=1, key="att_edu")
            perf = st.selectbox("Performance rating", ms.PERFORMANCE_RATINGS, index=1, key="att_perf")
            years = st.number_input("Years at the company", 0.0, 60.0, 3.0, 1.0, key="att_years")
            promo = st.number_input("Years since last promotion", 0.0, 60.0, 2.0, 1.0, key="att_promo")
            salary = st.number_input("Monthly salary (KES)", 0.0, 10_000_000.0, 85000.0, 1000.0,
                                     key="att_salary")
        with right:
            overtime = st.number_input("Overtime hours per month", 0.0, 300.0, 20.0, 1.0,
                                       key="att_overtime")
            satisfaction = st.number_input("Job satisfaction (1 to 5)", 1.0, 5.0, 3.0, 0.5, key="att_sat")
            balance = st.number_input("Work-life balance (1 to 5)", 1.0, 5.0, 3.0, 0.5, key="att_wlb")
            distance = st.number_input("Distance from home (km)", 0.0, 500.0, 12.0, 1.0, key="att_dist")
            previous = st.number_input("Companies worked for before", 0.0, 30.0, 2.0, 1.0,
                                       key="att_prev")
            remote = st.number_input("Remote work days per week", 0.0, 7.0, 1.0, 1.0, key="att_remote")
            bonus = optional_number("Bonus (KES), leave empty if none", "att_bonus", 0, 100_000_000)
        attrition_submit = st.form_submit_button("Predict attrition", type="primary")

    if attrition_submit:
        try:
            result = ms.predict_attrition({
                "department": dept, "job_role_level": level, "education_level": edu,
                "performance_rating": perf, "years_at_company": years,
                "years_since_last_promotion": promo, "monthly_salary_kes": salary,
                "overtime_hours_per_month": overtime, "job_satisfaction_score": satisfaction,
                "work_life_balance_score": balance, "distance_from_home_km": distance,
                "num_companies_worked_previously": previous,
                "remote_work_days_per_week": remote, "bonus_amount_kes": bonus,
            })
            st.metric("Likely to leave the company", result["predicted_attrition"])
            st.progress(result["probability_of_leaving"],
                        text=f"Probability of leaving: {result['probability_of_leaving']:.1%}")
            st.caption(f"Model: {result['model']}")
        except ms.InputError as error:
            st.error(f"Check your input: {error}")

# -----------------------------------------------------------------------
# 3. Clustering: store segment
# -----------------------------------------------------------------------
with tab_store:
    with st.form("store_form"):
        revenue = st.number_input("Monthly revenue (KES)", 0.0, 1_000_000_000.0, 1_200_000.0,
                                  10_000.0, key="store_revenue")
        traffic = st.number_input("Daily foot traffic (shoppers)", 0.0, 1_000_000_000.0, 400.0,
                                  10.0, key="store_traffic")
        basket = st.number_input("Average basket size (KES)", 0.0, 1_000_000_000.0, 120.0, 5.0,
                                 key="store_basket")
        per_sqm = st.number_input("Revenue per square metre (KES)", 0.0, 1_000_000_000.0, 5000.0,
                                  100.0, key="store_sqm")
        per_staff = st.number_input("Revenue per staff member (KES)", 0.0, 1_000_000_000.0,
                                    140_000.0, 1000.0, key="store_staff")
        store_submit = st.form_submit_button("Find the segment", type="primary")

    if store_submit:
        try:
            result = ms.predict_store_segment({
                "monthly_revenue_kes": revenue, "daily_foot_traffic": traffic,
                "avg_basket_size_kes": basket, "revenue_per_sqm": per_sqm,
                "revenue_per_staff": per_staff,
            })
            st.success(f"Cluster {result['assigned_cluster']}: {result['segment_name']}")
            # A table of all five clusters, with the assigned one marked
            rows = []
            for p in ms.cluster_profiles():
                rows.append({
                    "Assigned": "yes" if p["cluster"] == result["assigned_cluster"] else "",
                    "Cluster": p["cluster"], "Name": p["name"],
                    "Stores": p["stores_in_training_data"],
                    "Monthly revenue (KES)": round(p["monthly_revenue_kes"]),
                    "Daily foot traffic": round(p["daily_foot_traffic"]),
                    "Basket size (KES)": round(p["avg_basket_size_kes"]),
                })
            st.caption("Typical store in each cluster (worked out from the K-Means cluster centres)")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(f"Model: {result['model']}")
        except ms.InputError as error:
            st.error(f"Check your input: {error}")

# -----------------------------------------------------------------------
# 4. Association rules: music recommender
# -----------------------------------------------------------------------
with tab_music:
    choices = dict(ms.track_choices())            # label -> track id
    labels_by_id = {track_id: label for label, track_id in choices.items()}
    with st.form("music_form"):
        st.write("Pick the tracks already played in the listening session. "
                 "Try M09 + M10, or M01 + M03, or M13 + M15.")
        picked = st.multiselect("Tracks played so far", options=list(choices),
                                default=[labels_by_id["M09"], labels_by_id["M10"]],
                                key="music_tracks")
        top_n = st.slider("How many tracks to recommend", 1, 10, 3, key="music_top")
        music_submit = st.form_submit_button("Recommend next tracks", type="primary")

    if music_submit:
        try:
            result = ms.recommend_tracks([choices[label] for label in picked], top_n)
            if not result["recommendations"]:
                st.warning("No saved rule matches these tracks. "
                           "Try a pair that is often played together, such as M09 + M10.")
            else:
                table = pd.DataFrame([{
                    "Track": r["track_id"], "Name": r["item_name"], "Genre": r["category"],
                    "Confidence": f"{r['confidence']:.1%}", "Lift": r["lift"],
                } for r in result["recommendations"]])
                st.subheader("Play next")
                st.dataframe(table, hide_index=True, width="stretch")
                st.caption(f"Model: {result['model']}")
        except ms.InputError as error:
            st.error(f"Check your input: {error}")
