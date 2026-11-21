# Lab Submission Instructions

## Student Details and Individual Member Contributions

**Name of the team on GitHub Classroom:**

**Member 1:**

| **Details**                                                                                                                           | **Comment**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|:--------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Student ID**                                                                                                                        | 153359                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Name**                                                                                                                              | Salim Rashid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | I built the API core: I added the four saved models and wrote the first `api.py`, which loads them once at startup and serves the regression (rental price) and classification (employee attrition) endpoints with input checks that return HTTP 400 on bad data. I also added `requirements/serving.txt` and the `Procfile`. I tested both endpoints in Postman. I learned to load models once and never retrain them in the API, and to pin library versions (scikit-learn 1.6.1) so the saved models load correctly. Branch: https://github.com/BI-course/smm-202604-a1/tree/feature/api-core-and-models            |

**Member 2:**

| **Details**                                                                                                                           | **Comment**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|:--------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Student ID**                                                                                                                        | 167273                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Name**                                                                                                                              | Mbatha, Amanda Koki                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | I handled clustering and recommender routes, as well as error handling. I updated the api.py file to incorporate the store performance segmentation and music session recommender endpoints, while retaining the previous member's existing endpoints. I also added the tests folder to the repository and ran the api locally to verify that all four ML models loaded successfully. Finally, I used Postman to test the new endpoints with both valid and invalid inputs. I tested store segmentation, generating music recommendations, submitting invalid store data, and submitting empty request bodies. The tests helped verify that the api returned the expected predictions and rejected the invalid requests with HTTP 400 error responses. I learned how unsupervised ML models can be integrated into a REST api to provide predictions and recommendation. Link to branch:https://github.com/BI-course/smm-202604-a1/tree/feature/unsupervised-endpoints |

**Member 3:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        |   155858          |
| **Name**                                                                                                                              | Mvurya Hassan Wanini            |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** |   I set up Docker, Gunicorn and Nginx: I added a Dockerfile that runs the API under Gunicorn, created a self-signed HTTPS certificate, and ran two copies of the API behind an Nginx reverse proxy that redirects http to https and forwards /api/ requests to Flask. I tested it through https://localhost/api and checked the Gunicorn log and docker compose ps (two API containers plus Nginx). I learned that Gunicorn is the production server while Flask's built-in server is for development only, that a Docker image bundles the code, models and exact library versions so it runs the same everywhere, and that a reverse proxy gives one secure public entry point and can share requests across several containers. Branch: https://github.com/BI-course/smm-202604-a1/tree/feature/docker-nginx-setup          |

**Member 4:**

| **Details**                                                                                                                           | **Comment**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|:--------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Student ID**                                                                                                                        | 167142                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Name**                                                                                                                              | Fiona Nduu                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | I deployed the finished API to Render by connecting the team's GitHub repo to a new Web Service and waiting until all four models loaded and the status showed Live. I then used Postman to test the rental price, employee attrition, store segment and music recommender routes, and each returned the expected prediction. I also sent an empty request body to the store route and got an HTTP 400 error with a message, confirming the API rejects invalid input.I learned that an API lets other programs use a machine learning model over the internet. I used a GET request on /api/health to check the service was running, and POST requests to send JSON inputs to the model routes and get predictions back. In Postman, I also saw that a 200 status means success, while a 400 means invalid input, as when I sent an empty {} body.   link to branch https://github.com/BI-course/smm-202604-a1/tree/feature/render-deployment |

**Member 5:**

| **Details**                                                                                                                           | **Comment**  |
|:--------------------------------------------------------------------------------------------------------------------------------------|:-------------|
| **Student ID**                                                                                                                        | 169640       |
| **Name**                                                                                                                              | Brian Kiptoo |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | I built the web pages, the Streamlit app and the Hugging Face Space. The pages (HTML, CSS and JS in separate files) call the API for all four models, check the input before sending it, and show clear messages for errors or when the API is not running. I deployed the Streamlit app on Streamlit Community Cloud and published the pages as a Hugging Face Static Space that calls the Render API. I learned that a browser page can only call an API on another address when the server allows it (CORS), that a Static Space has no Python so it must call a hosted API, that Streamlit reruns the script on every click so models must be loaded once and cached, and that input mistakes should become readable messages, not crashes. Branch: https://github.com/BI-course/smm-202604-a1/tree/feature/hf-streamlit-frontend             |

## Chosen Level of Difficulty

**Specify the chosen level of difficulty** (baseline, intermediate, or advanced):

## Video Demonstration

Submit the link to a short video (**not more than 10 minutes**) demonstrating your complete solution at whichever difficulty level you chose. Please ensure that the lecturer has rights to view the video.

At minimum, the video must show, in one continuous walkthrough:

1. `api.py` running **locally**, with a live request/response against each model endpoint you implemented.
2. If you attempted the Intermediate or Advanced tier: the same API running inside its **Gunicorn + Nginx Docker container**, reached through the reverse proxy, not the Flask development server.
3. If you attempted the Advanced tier: your **Hugging Face Space** and your **Streamlit Community Cloud app**, each opened live in a browser, with at least one prediction made on each tab, and your **Render**-hosted API answering a request from Postman or `curl`.

**Why this matters for grading:** free-tier public deployments are not guaranteed to still exist by the time this is graded. The video is treated as the authoritative record of what was actually working at submission time; a broken link at grading time is not penalized if the video clearly shows that deployment working.

Note that you are required to submit the link to the video and NOT the video itself. The video should NOT be uploaded to your repository—that would be a misuse of GitHub.

**Link to the video:**

## Public URLs

Fill in only the rows that apply to your chosen difficulty level (see Part C below). Leave a row blank if your tier does not require that deployment target.

| **Deployment target**                             | **URL**                                                                                                                                                                                                                                                                                                                                                    |
|:--------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Render (the full `api.py`, all endpoints)         | https://smm-202604-a1.onrender.com/api/v1/models/rental-price-regressor/predictions                                                                                                                                                                                                                                                                        | 
|                                                   | https://smm-202604-a1.onrender.com/api/v1/models/employee-attrition-classifier/predictions                                                                                                                                                                                                                                                                 |
|                                                   | https://smm-202604-a1.onrender.com/api/v1/models/store-performance-segmenter/predictions                                                                                                                                                                                                                                                                   |
|                                                   | https://smm-202604-a1.onrender.com/api/v1/models/music-session-recommender/predictions                                                                                                                                                                                                                                                                     |
| Hugging Face Space (Gradio, one tab per model)    |                                                                                                                                                                                                                                                                                                                                                            |
| Streamlit Community Cloud app (one tab per model) |                                                                                                                                                                                                                                                                                                                                                            |

---

## Scenario

Across earlier labs, **your group** was assigned its own dataset and trained your own models for four distinct tasks:

1. A **regression** task
2. A **classification** task
3. A **k-Means clustering** task
4. An **Apriori association-rule mining** task

Different groups worked with different datasets and, in some cases, different algorithms for each task -- there is no single shared reference repository or model list for the whole class. Wherever this document refers to "your regressor," "your classifier," "your clustering model," or "your recommender," it means **whichever model your own group already trained and saved for that task**, not a specific named algorithm.

You are now required to build a single Flask API (`api.py`) that serves **your own group's four models**, loaded from disk (never retrained inside `api.py`), and to demonstrate that API running correctly across a growing number of deployment surfaces depending on the difficulty tier you choose.

### Part A: Regression and Classification

- Locate the trained regression and classification model files your group produced and saved in your earlier labs on Regression and Classification.
- Update [api.py](api.py) to include end-points that load these two models from disk and serve predictions from them.

### Part B: Clustering and Association Rule Mining

- Locate the trained clustering model and the association rules your group produced and saved in your earlier labs on Clustering and Association Rule Mining.
- Update [api.py](api.py) to include end-points that:
  - Load the clustering model from disk and predict which cluster a new client belongs to
  - Load the association rules from disk and recommend products based on them

**Note 1:** **`api.py` is NOT production-grade as it is.** It is only meant for demonstration purposes. Scalability and security must be taken into consideration before deploying an API in a production environment.

**Note 2:** Some students often treat the API as an afterthought, focusing only on training ML models. In practice, **the API is the product** -- it is how others interact with your model. The "hidden" learning here is that the delivery mechanism (API design, usability, error handling, and even documentation) often matters more to stakeholders in the industry than the models themselves.

### Part C: Deployment Breadth (Read Before Choosing Your Tier)

All three tiers below ask you to serve **the same four models** (your regressor, your classifier, your clustering model, your recommender) from **one, unmodified `api.py`**. The model files and the Flask routes do not change between tiers -- only how many different places that one working API is shown to run correctly.

**Baseline (Required):**

- Update `api.py` to serve **at least three of your four** models, loaded from disk.
- Run and demonstrate `api.py` **locally** with Flask's development server, with a successful request against each implemented endpoint.

**Intermediate (Recommended):**

- Update `api.py` to serve **all four** of your models, loaded from disk.
- Dockerize your Flask API using **Gunicorn**, and place it behind an **Nginx** reverse proxy (see the reverse-proxy lab notes for the full Docker Compose setup).
- Deploy the same, unmodified `api.py` to **Render**, and demonstrate a real request against it from outside your own machine (cURL or Postman).

**Advanced (Optional):**

- Everything required for Intermediate, plus:
  - Create a web page (or pages) using Basic HTML, CSS, and Vanilla JavaScript that demonstrates the use of the API
  - Implement basic error handling (e.g., missing inputs)
  - Build **one Gradio app with a tab per model** (all four of your models) and publish it as a **Hugging Face Space**
  - Build **one Streamlit app with a tab per model**, covering the same four models, and publish it on **Streamlit Community Cloud**
  - All deployment targets -- local, Dockerized/Nginx, Render, Hugging Face, and Streamlit -- must serve models loaded **from the same trained artifacts**; retraining different models per platform does not satisfy this requirement

Why this is production-relevant beyond the model itself:

- **Portability:** the same container image and the same trained artifacts run unmodified on your laptop, inside Docker, and on a managed cloud host.
- **Consistency:** no "but it works on my machine" gap between what you demo locally and what a domain expert sees on a public URL.
- **Separation of concerns:** a REST API (Render), a no-code demo UI (Hugging Face/Streamlit), and your own frontend (the Advanced-tier HTML pages) are three different ways of exposing the *same* underlying models to three different kinds of user -- a developer integrating your API, a non-technical domain expert validating it, and an end customer.

---

## Marking Guide

Refer to the marking guide available here: [lab_submission_marking_guide.md](lab_submission_marking_guide.md)