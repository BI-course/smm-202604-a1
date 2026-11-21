---
title: Group A Model Validation Demos
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: 3.12.12
app_file: app.py
pinned: false
---

# Group A: model validation demos

One Gradio app with four tabs, one for each of our saved models:

1. Rental price (regression)
2. Employee attrition (classification)
3. Store segment (clustering)
4. Music recommender (association rules)

The models are the same `.joblib` files used by our Flask API. Nothing is retrained here.
