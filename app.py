import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime



# Charger le modèle sauvegardé
model = joblib.load('covid_risk_model.pkl')

# Interface Streamlit
st.title('Prédiction du risque de décès dû au COVID-19')

# Entrées utilisateur pour les 22 features
usmer = st.selectbox('USMER', [1, 2])
medical_unit = st.selectbox('Unité médicale', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
sex = st.selectbox('Sexe', [1, 2])
patient_type = st.selectbox('Type de patient', [1, 2])
date_died = st.text_input('Date de décès (format JJ/MM/AAAA)', '01/01/2020')
intubed = st.selectbox('Intubé', [1, 2])
pneumonia = st.selectbox('Pneumonie', [1, 2])
age = st.slider('Âge', 0, 100, 50)
pregnant = st.selectbox('Enceinte', [1, 2])
diabetes = st.selectbox('Diabète', [1, 2])
copd = st.selectbox('BPCO', [1, 2])
asthma = st.selectbox('Asthme', [1, 2])
inmsupr = st.selectbox('Immunosuppression', [1, 2])
hipertension = st.selectbox('Hypertension', [1, 2])
other_disease = st.selectbox('Autre maladie', [1, 2])
cardiovascular = st.selectbox('Maladie cardiovasculaire', [1, 2])
obesity = st.selectbox('Obésité', [1, 2])
renal_chronic = st.selectbox('Maladie rénale chronique', [1, 2])
tobacco = st.selectbox('Tabagisme', [1, 2])
clasif_final = st.selectbox('Classification finale', [1, 2, 3, 4, 5, 6, 7])
icu = st.selectbox('Unité de soins intensifs', [1, 2])

# Prédiction
if st.button('Prédire'):
    # Convertir la date en jour, mois et année
    try:
        date_obj = datetime.strptime(date_died, '%d/%m/%Y')
        day = date_obj.day
        month = date_obj.month
        year = date_obj.year
    except ValueError:
        st.error("Format de date invalide. Utilisez le format JJ/MM/AAAA.")
        st.stop()

    # Créer un DataFrame avec les 22 features dans le même ordre que lors de l'entraînement
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

    # Faire la prédiction
    try:
        prediction = model.predict(input_data)
        st.write(f'Risque de décès : {"Élevé" if prediction[0] == 1 else "Faible"}')
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
