# app.py
import streamlit as st
import numpy as np
import joblib

# Charger modèle, scaler et features
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Prix Maison Californie", page_icon="🏡")
st.title("🏡 Estimation du prix d'une maison en Californie")

st.write("Saisissez les caractéristiques de la maison ci-dessous :")

# Interface utilisateur
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature}", value=0.0, step=0.1)

if st.button("Prédire le prix"):
    input_array = np.array([list(user_input.values())])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    st.success(f"💰 Prix estimé : {prediction:.2f} (en centaines de milliers de dollars)")
