import streamlit as st
import pandas as pd
import joblib

# load model
model = joblib.load("final_model.pkl")

st.title("Happiness Score Predictor")

# take inputs
gdp = st.number_input("GDP per capita", value=1.0)
social = st.number_input("Social support", value=1.0)
life = st.number_input("Healthy life expectancy", value=1.0)

# button to predict
if st.button("Predict"):
    data = pd.DataFrame([[gdp, social, life]], 
                        columns=["GDP per capita", "Social support", "Healthy life expectancy"])
    pred = model.predict(data)[0]
    st.write("Predicted Happiness Score:", round(pred, 2))
