"""
app.py  --  Gradio app for Hugging Face Spaces

One app with FOUR tabs, one tab per model:
    1. Rental price      (regression)
    2. Employee attrition (classification)
    3. Store segment     (clustering)
    4. Music recommender (association rules)

The models are loaded from the same .joblib files that api.py uses.
All the loading, input checking and predicting is done in model_service.py,
so this file only builds the screen.

Run locally:   python app.py        (then open http://127.0.0.1:7860)
"""

import gradio as gr

import model_service as ms

# -----------------------------------------------------------------------
# Hugging Face's free ZeroGPU hardware expects at least one function marked
# with @spaces.GPU. Our models do NOT use a GPU, so this function does
# nothing. It only stops the "No @spaces.GPU function detected" start-up
# message. On a normal computer the 'spaces' package is not installed, so
# the try/except simply skips this part.
# -----------------------------------------------------------------------
try:
    import spaces

    @spaces.GPU
    def _warm_zerogpu():
        return None

    _warm_zerogpu()
except ImportError:
    pass

# Load the models once, when the app starts (not on every click)
ms.load_models()


# -----------------------------------------------------------------------
# Small helpers that turn results into text (Markdown) for the screen
# -----------------------------------------------------------------------
def problem(message):
    """What the user sees when the input is wrong."""
    return f"**Check your input:** {message}"


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


# -----------------------------------------------------------------------
# 1. Regression: rental price
# -----------------------------------------------------------------------
RENT_FIELDS = [
    "property_age_years", "size_sqm", "num_bedrooms", "num_bathrooms",
    "distance_to_cbd_km", "service_charge_kes", "garage_size_sqm",
    "property_type", "neighborhood", "furnishing_status",
]


def run_rent(*values):
    try:
        result = ms.predict_rental_price(dict(zip(RENT_FIELDS, values)))
    except ms.InputError as error:
        return problem(error)
    return (f"### Predicted rent: KES {result['predicted_rental_price_kes']:,.0f} per month\n\n"
            f"Model: {result['model']}")


# -----------------------------------------------------------------------
# 2. Classification: employee attrition
# -----------------------------------------------------------------------
ATTRITION_FIELDS = [
    "department", "job_role_level", "education_level", "performance_rating",
    "years_at_company", "years_since_last_promotion", "monthly_salary_kes",
    "overtime_hours_per_month", "job_satisfaction_score", "work_life_balance_score",
    "distance_from_home_km", "num_companies_worked_previously",
    "remote_work_days_per_week", "bonus_amount_kes",
]


def run_attrition(*values):
    try:
        result = ms.predict_attrition(dict(zip(ATTRITION_FIELDS, values)))
    except ms.InputError as error:
        return problem(error)
    leaving = "Yes" if result["predicted_attrition"] == "Yes" else "No"
    rows = [(label, f"{prob:.1%}") for label, prob in result["class_probabilities"].items()]
    return (f"### Likely to leave the company: {leaving}\n\n"
            + markdown_table(["Leaves the company?", "Probability"], rows)
            + f"\n\nModel: {result['model']}")


# -----------------------------------------------------------------------
# 3. Clustering: store segment
# -----------------------------------------------------------------------
def run_store(*values):
    try:
        result = ms.predict_store_segment(dict(zip(ms.STORE_FIELDS, values)))
    except ms.InputError as error:
        return problem(error)

    # A table of all five clusters, with the assigned one marked
    rows = []
    for p in ms.cluster_profiles():
        mark = "**assigned**" if p["cluster"] == result["assigned_cluster"] else ""
        rows.append((p["cluster"], p["name"], p["stores_in_training_data"],
                     f"{p['monthly_revenue_kes']:,.0f}", f"{p['daily_foot_traffic']:,.0f}",
                     f"{p['avg_basket_size_kes']:,.0f}", mark))
    return (f"### Cluster {result['assigned_cluster']}: {result['segment_name']}\n\n"
            "Typical store in each cluster (worked out from the K-Means cluster centres):\n\n"
            + markdown_table(
                ["Cluster", "Name", "Stores", "Monthly revenue (KES)",
                 "Daily foot traffic", "Basket size (KES)", ""], rows)
            + f"\n\nModel: {result['model']}")


# -----------------------------------------------------------------------
# 4. Association rules: music recommender
# -----------------------------------------------------------------------
def run_music(tracks, top_n):
    try:
        result = ms.recommend_tracks(tracks, top_n)
    except ms.InputError as error:
        return problem(error)
    if not result["recommendations"]:
        return ("### No recommendation\n\nNo saved rule matches these tracks. "
                "Try a pair that is often played together, such as M09 + M10.")
    rows = [(r["track_id"], r["item_name"], r["category"],
             f"{r['confidence']:.1%}", r["lift"]) for r in result["recommendations"]]
    return ("### Play next\n\n"
            + markdown_table(["Track", "Name", "Genre", "Confidence", "Lift"], rows)
            + f"\n\nModel: {result['model']}")


# -----------------------------------------------------------------------
# The screen: one gr.Blocks app with four tabs
# -----------------------------------------------------------------------
with gr.Blocks(title="Group A: Model validation demos") as demo:
    gr.Markdown("# Group A: model validation demos\n"
                "Choose a tab, change the values if you like, and press the button. "
                "The four models are the same saved files that our API uses.")

    # ---------------- Tab 1: rent ----------------
    with gr.Tab("Rental price"):
        with gr.Row():
            with gr.Column():
                rent_age = gr.Number(label="Property age (years)", value=5, minimum=0, maximum=150)
                rent_size = gr.Number(label="Size (square metres)", value=90, minimum=1, maximum=5000)
                rent_beds = gr.Number(label="Bedrooms", value=2, minimum=0, maximum=20)
                rent_baths = gr.Number(label="Bathrooms", value=2, minimum=0, maximum=20)
                rent_cbd = gr.Number(label="Distance to the city centre (km)", value=6, minimum=0, maximum=200)
            with gr.Column():
                rent_service = gr.Number(label="Service charge (KES per month)", value=8000,
                                         minimum=0, maximum=10_000_000)
                rent_garage = gr.Number(label="Garage size (square metres), leave empty if none",
                                        value=None, minimum=0, maximum=1000)
                rent_type = gr.Dropdown(label="Property type", choices=ms.PROPERTY_TYPES, value="Apartment")
                rent_area = gr.Dropdown(label="Neighbourhood", choices=ms.NEIGHBORHOODS, value="Kilimani")
                rent_furnish = gr.Dropdown(label="Furnishing", choices=ms.FURNISHING_STATUSES,
                                           value="Semi-Furnished")
        rent_button = gr.Button("Predict rent", variant="primary")
        rent_output = gr.Markdown()
        rent_button.click(
            run_rent,
            [rent_age, rent_size, rent_beds, rent_baths, rent_cbd, rent_service, rent_garage,
             rent_type, rent_area, rent_furnish],
            rent_output, api_name="rental_price")

    # ---------------- Tab 2: attrition ----------------
    with gr.Tab("Employee attrition"):
        with gr.Row():
            with gr.Column():
                att_dept = gr.Dropdown(label="Department", choices=ms.DEPARTMENTS, value="Sales")
                att_level = gr.Dropdown(label="Job level", choices=ms.JOB_ROLE_LEVELS, value="Mid")
                att_edu = gr.Dropdown(label="Education", choices=ms.EDUCATION_LEVELS, value="Bachelor's")
                att_perf = gr.Dropdown(label="Performance rating", choices=ms.PERFORMANCE_RATINGS,
                                       value="Average")
                att_years = gr.Number(label="Years at the company", value=3, minimum=0, maximum=60)
                att_promo = gr.Number(label="Years since last promotion", value=2, minimum=0, maximum=60)
                att_salary = gr.Number(label="Monthly salary (KES)", value=85000,
                                       minimum=0, maximum=10_000_000)
            with gr.Column():
                att_overtime = gr.Number(label="Overtime hours per month", value=20, minimum=0, maximum=300)
                att_sat = gr.Number(label="Job satisfaction (1 to 5)", value=3, minimum=1, maximum=5)
                att_wlb = gr.Number(label="Work-life balance (1 to 5)", value=3, minimum=1, maximum=5)
                att_dist = gr.Number(label="Distance from home (km)", value=12, minimum=0, maximum=500)
                att_prev = gr.Number(label="Companies worked for before", value=2, minimum=0, maximum=30)
                att_remote = gr.Number(label="Remote work days per week", value=1, minimum=0, maximum=7)
                att_bonus = gr.Number(label="Bonus (KES), leave empty if none", value=None,
                                      minimum=0, maximum=100_000_000)
        att_button = gr.Button("Predict attrition", variant="primary")
        att_output = gr.Markdown()
        att_button.click(
            run_attrition,
            [att_dept, att_level, att_edu, att_perf, att_years, att_promo, att_salary,
             att_overtime, att_sat, att_wlb, att_dist, att_prev, att_remote, att_bonus],
            att_output, api_name="employee_attrition")

    # ---------------- Tab 3: store segment ----------------
    with gr.Tab("Store segment"):
        store_revenue = gr.Number(label="Monthly revenue (KES)", value=1200000, minimum=0)
        store_traffic = gr.Number(label="Daily foot traffic (shoppers)", value=400, minimum=0)
        store_basket = gr.Number(label="Average basket size (KES)", value=120, minimum=0)
        store_sqm = gr.Number(label="Revenue per square metre (KES)", value=5000, minimum=0)
        store_staff = gr.Number(label="Revenue per staff member (KES)", value=140000, minimum=0)
        store_button = gr.Button("Find the segment", variant="primary")
        store_output = gr.Markdown()
        store_button.click(
            run_store,
            [store_revenue, store_traffic, store_basket, store_sqm, store_staff],
            store_output, api_name="store_segment")

    # ---------------- Tab 4: music recommender ----------------
    with gr.Tab("Music recommender"):
        gr.Markdown("Tick the tracks already played in the listening session. "
                    "Try M09 + M10, or M01 + M03, or M13 + M15.")
        music_tracks = gr.CheckboxGroup(label="Tracks played so far",
                                        choices=ms.track_choices(), value=["M09", "M10"])
        music_top = gr.Slider(label="How many tracks to recommend", minimum=1, maximum=10,
                              step=1, value=3)
        music_button = gr.Button("Recommend next tracks", variant="primary")
        music_output = gr.Markdown()
        music_button.click(run_music, [music_tracks, music_top], music_output,
                           api_name="music_recommender")

if __name__ == "__main__":
    demo.launch()
