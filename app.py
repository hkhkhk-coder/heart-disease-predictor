
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

@st.cache_resource
def load_model():
    with open("heart_disease_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.title("Heart Disease Prediction")
st.write("Enter the four patient attributes below.")

age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox(
    "Chest Pain Type",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "Typical angina",
        2: "Atypical angina",
        3: "Non-anginal pain",
        4: "Asymptomatic"
    }[x]
)
thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50,
    max_value=250,
    value=150
)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "age": float(age),
        "sex": 1.0 if sex == "Male" else 0.0,
        "cp": float(cp),
        "thalach": float(thalach)
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.warning("The model predicts: Heart Disease Present")
    else:
        st.success("The model predicts: No Heart Disease")

    st.caption(
        "Educational demonstration only. This prediction is not a medical diagnosis."
    )
