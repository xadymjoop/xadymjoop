import streamlit as st
import pandas as pd
import joblib
import gdown
import os


# Charger le modèle sauvegardé
model = joblib.load('covid_risk_model.pkl')

# Interface Streamlit
st.title('Prédiction du risque de décès dû au COVID-19')

# Entrées utilisateur
age = st.slider('Âge', 0, 100, 50)
pneumonia = st.selectbox('Pneumonie', [0, 1])
diabetes = st.selectbox('Diabète', [0, 1])
obesity = st.selectbox('Obésité', [0, 1])

# Prédiction
if st.button('Prédire'):
    input_data = [[age, pneumonia, diabetes, obesity]]
    prediction = model.predict(input_data)
    st.write(f'Risque de décès : {"Élevé" if prediction[0] == 1 else "Faible"}')
