import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime


# Vérifier si le modèle existe avant de le charger
model_path = 'covid_risk_model.pkl'
if not os.path.exists(model_path):
    st.error("Le fichier du modèle 'covid_risk_model.pkl' est introuvable. Assurez-vous qu'il est bien dans le répertoire.")
    st.stop()

# Charger le modèle sauvegardé
model = joblib.load(model_path)

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

    # Gérer la valeur de `DAY_DIED` si aucune date n'est fournie (par exemple, date par défaut pour les vivants)
    if date_died.strip() == '01/01/2020':  # Exemple de date par défaut
        day_died = 0  # Indiquer que la personne est vivante
    else:
        day_died = year  # Sinon, utiliser l'année

    # Créer un DataFrame avec les 22 features dans le même ordre que lors de l'entraînement
    input_data = pd.DataFrame({
        'USMER': [usmer],
        'MEDICAL_UNIT': [medical_unit],
        'SEX': [sex],
        'PATIENT_TYPE': [patient_type],
        'DAY_DIED': [day_died],
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

    # Vérifiez que le DataFrame a bien 22 colonnes
    expected_features = 22
    if input_data.shape[1] != expected_features:
        st.error(f"Erreur : le modèle attend {expected_features} caractéristiques, mais {input_data.shape[1]} ont été fournies.")
        st.stop()

    # Faire la prédiction
    try:
        prediction = model.predict(input_data)
        st.write(f'Risque de décès : {"Élevé" if prediction[0] == 1 else "Faible"}')
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

