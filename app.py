"""Streamlit app: would this passenger have survived the Titanic?

A small educational app. It loads a saved decision tree (model.joblib),
takes inputs from the user, and shows the estimated chance of survival.
"""
import joblib
import pandas as pd
import streamlit as st

# Same column order as train_model.py (very important!)
FEATURES = ["pclass", "sex", "age", "fare"]
MODEL_FILE = "model.joblib"

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")


@st.cache_resource
def load_model():
    """Load the model once, so it is not reloaded on every click."""
    return joblib.load(MODEL_FILE)


def validate(age, fare):
    """Return an error message if the input is invalid, otherwise None."""
    if age is None or not (1 <= age <= 80):
        return "Age must be between 1 and 80."
    if fare is None or not (0 <= fare <= 500):
        return "Fare must be between 0 and 500."
    return None


st.title("🚢 Would this passenger have survived?")
st.write("An educational model trained on the Titanic passenger list. "
         "Choose a passenger's details and press **Predict**.")

# ---- Inputs ----
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Ticket class", [1, 2, 3], index=2,
                          help="1 = upper, 2 = middle, 3 = lower class")
    sex = st.radio("Sex", ["female", "male"], horizontal=True)
with col2:
    age = st.slider("Age (years)", 1, 80, 30)
    fare = st.slider("Fare paid (pounds)", 0, 500, 30)

# ---- Prediction ----
if st.button("Predict", type="primary"):
    problem = validate(age, fare)
    if problem:
        st.warning(problem)
        st.stop()

    try:
        model = load_model()
    except FileNotFoundError:
        st.error("The model file (model.joblib) is missing. "
                 "Run `python train_model.py` to create it.")
        st.stop()
    except Exception:
        st.error("The model file could not be loaded. It may have been saved "
                 "with a different scikit-learn version. Run "
                 "`python train_model.py` again.")
        st.stop()

    # The model must see columns in the same order it was trained with
    if list(getattr(model, "feature_names_in_", FEATURES)) != FEATURES:
        st.error("The model's feature order does not match this app. "
                 "Run `python train_model.py` again.")
        st.stop()

    row = pd.DataFrame([[pclass, 1 if sex == "female" else 0, age, fare]],
                       columns=FEATURES)
    chance = float(model.predict_proba(row)[0][1])

    st.divider()
    st.metric("Estimated chance of survival", f"{chance:.0%}")
    st.progress(chance)
    if chance >= 0.5:
        st.success("The model predicts this passenger **would have survived**.")
    else:
        st.info("The model predicts this passenger **would not have survived**.")

    st.caption("A learning exercise. It describes a historical dataset "
               "and is not advice.")

# ---- About ----
with st.expander("About this model"):
    st.markdown(
        """
- **Model:** Decision Tree classifier (scikit-learn, `max_depth=4`)
- **Data:** Titanic passengers (714 with a known age), from the seaborn library
- **Accuracy on unseen test data:** about 79.7%, compared with 59.4% for always guessing "did not survive"
- **Most important features:** sex, then ticket class, then age and fare
- **Limitations:** small dataset, only 4 inputs, and it misses about 28% of survivors. 
It finds patterns in 1912 data and does not explain *why* people survived.
- No personal data is collected or stored by this app.
        """
    )
