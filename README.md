# Titanic Survival Predictor (Streamlit App)

**Project 05 | Data Science & AI | InternCircle**

An interactive web app that wraps the decision tree from Project 04. Choose a passenger's ticket class, sex, age and fare, press **Predict**, and the app shows the model's estimated chance of survival.

**Live app:** _paste your Streamlit link here_

![Screenshot of a prediction](screenshot.png)

## Features
- Inputs with sensible limits: ticket class (select box), sex (radio), age 1 to 80 and fare 0 to 500 (sliders)
- Model loaded once from `model.joblib` with `@st.cache_resource`, so the app stays fast
- Prediction shown with `st.metric`, a progress bar and a plain-language message
- Friendly messages (no error traces) if the model file is missing or cannot be loaded, or if the input is invalid
- Caption and "About this model" section: what the model is, what data it learned from, and that it is an educational example
- No personal data is collected or stored

## About the model
- Decision tree (scikit-learn, `max_depth=4`) trained on the Titanic dataset (714 passengers with a known age)
- Accuracy on unseen test data: about 79.7%, compared with a 59.4% baseline
- Limitations: small dataset, only 4 inputs, misses about 28% of survivors. It finds patterns in historical data and is not advice.

## Files
| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `model.joblib` | The saved trained model |
| `train_model.py` | Script that trains and saves the model |
| `requirements.txt` | Packages needed to run the app (versions pinned) |

## Run it on your computer
```
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model (this also needs seaborn):
```
pip install seaborn
python train_model.py
```
Keep the feature order (`pclass, sex, age, fare`) the same in `train_model.py` and `app.py`.

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Go to streamlit.io/cloud, sign in with GitHub and click **Create app**.
3. Choose the repository, branch `main` and main file `app.py`, then click **Deploy**.

**Author:** Sankari
