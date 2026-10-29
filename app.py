import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")
st.title("Iris Flower Classification")
st.write("Enter flower measurements to predict the Iris species using a trained machine-learning model.")

@st.cache_resource
def load_artifacts():
    model = joblib.load("iris_model.joblib")
    scaler = joblib.load("iris_scaler.joblib")
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.warning("Model files not found. Run `python train_model.py` first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal length (cm)", min_value=0.1, max_value=10.0, value=5.1, step=0.1)
    sepal_width = st.number_input("Sepal width (cm)", min_value=0.1, max_value=10.0, value=3.5, step=0.1)
with col2:
    petal_length = st.number_input("Petal length (cm)", min_value=0.1, max_value=10.0, value=1.4, step=0.1)
    petal_width = st.number_input("Petal width (cm)", min_value=0.1, max_value=10.0, value=0.2, step=0.1)

if st.button("Predict species", type="primary"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(scaler.transform(features))[0]
    probabilities = model.predict_proba(scaler.transform(features))[0]
    st.success(f"Predicted species: **{prediction}**")
    st.write("Model confidence (estimated probability):")
    st.bar_chart({name: float(prob) for name, prob in zip(model.classes_, probabilities)})
