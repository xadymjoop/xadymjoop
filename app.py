import streamlit as st
import joblib
import pandas as pd
import gdown



# Télécharger le modèle depuis Google Drive
url = 'https://drive.google.com/uc?id=1X_aSkREb2TRXOHLmqzr8_neAy9OJEuTb'
output = 'covid_icu_model.pkl'
gdown.download(url, output, quiet=False)

# Titre de l'application
st.title("Prédiction d'admission en soins intensifs (COVID-19)")

# Formulaire pour saisir les données
age = st.number_input('Âge', min_value=0, max_value=120, value=50)
diabetes = st.selectbox('Diabète', [0, 1])
hypertension = st.selectbox('Hypertension', [0, 1])
obesity = st.selectbox('Obésité', [0, 1])
tobacco = st.selectbox('Tabagisme', [0, 1])

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
    st.write(f"Prédiction : {'Admis en soins intensifs' if prediction[0] == 1 else 'Non admis en soins intensifs'}")