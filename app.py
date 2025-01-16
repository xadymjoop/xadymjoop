import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime



# Titre de l'application
st.title("Entraînement et prédiction avec RandomForest")




# Charger le modèle pour faire des prédictions
if st.button("Charger le modèle et faire une prédiction"):
    try:
        model = joblib.load('covid_risk_model.pkl')
        st.success("Modèle chargé avec succès !")

        # Formulaire pour saisir les données
        st.write("## Saisissez les informations du patient")
        age = st.number_input('Âge', min_value=0, max_value=120, value=50)
        diabetes = st.selectbox('Diabète', [0, 1], help="0 = Non, 1 = Oui")
        hypertension = st.selectbox('Hypertension', [0, 1], help="0 = Non, 1 = Oui")
        obesity = st.selectbox('Obésité', [0, 1], help="0 = Non, 1 = Oui")
        tobacco = st.selectbox('Tabagisme', [0, 1], help="0 = Non, 1 = Oui")

        # Bouton pour faire une prédiction
        if st.button('Prédire'):
            input_data = pd.DataFrame({
                'AGE': [age],
                'DIABETES': [diabetes],
                'HIPERTENSION': [hypertension],
                'OBESITY': [obesity],
                'TOBACCO': [tobacco]
            })

            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.error("Le patient est susceptible d'être admis en soins intensifs.")
            else:
                st.success("Le patient n'est pas susceptible d'être admis en soins intensifs.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
