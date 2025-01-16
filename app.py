import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os


# Titre de l'application
st.title("Prédiction avec RandomForestClassifier")

# Charger le modèle
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None, f"Le fichier du modèle '{path}' est introuvable."
    try:
        model = joblib.load(path)
        return model, "Modèle chargé avec succès !"
    except Exception as e:
        return None, f"Erreur lors du chargement du modèle : {e}"

# Charger le modèle
model_path = 'covid_risk_model.pkl'
model, message = load_model(model_path)

if model is None:
    st.error(message)
else:
    st.success(message)

    # Collecte des données utilisateur
    st.write("## Saisissez les informations du patient")
    usmer = st.selectbox('Soins médicaux en milieu rural/urbain (USMER)', [0, 1])
    medical_unit = st.number_input('Unité médicale (code)', min_value=0, max_value=99, value=1)
    sex = st.selectbox('Sexe', [0, 1], help="0 = Femme, 1 = Homme")
    patient_type = st.selectbox('Type de patient', [0, 1], help="0 = Ambulatoire, 1 = Hospitalisé")
    day = st.number_input('Jour depuis la date de diagnostic', min_value=0, max_value=365, value=0)
    intubed = st.selectbox('Intubé', [0, 1], help="0 = Non, 1 = Oui")
    pneumonia = st.selectbox('Pneumonie', [0, 1], help="0 = Non, 1 = Oui")
    age = st.number_input('Âge', min_value=0, max_value=120, value=50)
    pregnant = st.selectbox('Enceinte', [0, 1], help="0 = Non, 1 = Oui")
    diabetes = st.selectbox('Diabète', [0, 1], help="0 = Non, 1 = Oui")
    copd = st.selectbox('Maladie pulmonaire obstructive chronique (COPD)', [0, 1])
    asthma = st.selectbox('Asthme', [0, 1])
    inmsupr = st.selectbox('Immunosuppression', [0, 1])
    hipertension = st.selectbox('Hypertension', [0, 1])
    other_disease = st.selectbox('Autre maladie', [0, 1])
    cardiovascular = st.selectbox('Maladie cardiovasculaire', [0, 1])
    obesity = st.selectbox('Obésité', [0, 1])
    renal_chronic = st.selectbox('Insuffisance rénale chronique', [0, 1])
    tobacco = st.selectbox('Tabagisme', [0, 1])
    clasif_final = st.number_input('Classification finale (code)', min_value=1, max_value=7, value=1)
    icu = st.selectbox('Admis en soins intensifs (ICU)', [0, 1])

    # Bouton pour faire une prédiction
    if st.button('Prédire'):
        try:
            # Créez un DataFrame avec les 21 caractéristiques attendues
            input_data = pd.DataFrame({
                'USMER': [usmer],
                'MEDICAL_UNIT': [medical_unit],
                'SEX': [sex],
                'PATIENT_TYPE': [patient_type],
                'DAY_DIED': [day],  # Jour extrait de la date
                'INTUBED': [intubed],
                'PNEUMONIA': [pneumonia],
                'AGE': [age],
                'PREGNANT': [pregnant],
                'DIABETES': [diabetes],
                'COPD': [copd],
                'ASTHMA': [asthma],
                'INMSUPR': [inmsupr],
                'HIPERTENSION': [hipertension],
                'OTHER_DISEASE': [other_disease],
                'CARDIOVASCULAR': [cardiovascular],
                'OBESITY': [obesity],
                'RENAL_CHRONIC': [renal_chronic],
                'TOBACCO': [tobacco],
                'CLASIFFICATION_FINAL': [clasif_final],
                'ICU': [icu]
            })

            # Faire une prédiction
            prediction = model.predict(input_data)

            # Afficher le résultat de la prédiction
            if prediction[0] == 1:
                st.error("Le patient est susceptible d'être admis en soins intensifs.")
            else:
                st.success("Le patient n'est pas susceptible d'être admis en soins intensifs.")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
