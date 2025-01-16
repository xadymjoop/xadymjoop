import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import gdown
import os

# Titre de l'application
st.title("Prédiction d'admission en soins intensifs (COVID-19)")

# Télécharger et charger le modèle
@st.cache_resource
def load_model():
    url = 'https://drive.google.com/uc?id=1X_aSkREb2TRXOHLmqzr8_neAy9OJEuTb'
    output = 'covid_icu_model.pkl'

    # Télécharger le modèle s'il n'existe pas déjà
    if not os.path.exists(output):
        gdown.download(url, output, quiet=True)

    # Vérifier que le fichier a bien été téléchargé
    if os.path.exists(output):
        try:
            return joblib.load(output)
        except Exception as e:
            raise RuntimeError(f"Impossible de charger le modèle : {e}")
    else:
        raise FileNotFoundError(f"Le fichier du modèle {output} n'a pas pu être téléchargé.")

# Charger le modèle avec vérification
try:
    model = load_model()
    st.success("Modèle chargé avec succès !")
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()  # Arrête l'exécution de l'application si le modèle ne peut pas être chargé

# Formulaire pour saisir les données
st.sidebar.header("Saisissez les informations du patient")
age = st.sidebar.number_input('Âge', min_value=0, max_value=120, value=50)
diabetes = st.sidebar.selectbox('Diabète', [0, 1], help="0 = Non, 1 = Oui")
hypertension = st.sidebar.selectbox('Hypertension', [0, 1], help="0 = Non, 1 = Oui")
obesity = st.sidebar.selectbox('Obésité', [0, 1], help="0 = Non, 1 = Oui")
tobacco = st.sidebar.selectbox('Tabagisme', [0, 1], help="0 = Non, 1 = Oui")

# Bouton pour faire une prédiction
if st.sidebar.button('Prédire'):
    input_data = pd.DataFrame({
        'AGE': [age],
        'DIABETES': [diabetes],
        'HIPERTENSION': [hypertension],
        'OBESITY': [obesity],
        'TOBACCO': [tobacco]
    })

    # Vérifier si les colonnes d'entrée sont correctes
    expected_columns = ['AGE', 'DIABETES', 'HIPERTENSION', 'OBESITY', 'TOBACCO']
    if list(input_data.columns) != expected_columns:
        st.error(f"Les colonnes d'entrée ne correspondent pas au modèle. Attendu : {expected_columns}")
    else:
        # Convertir les colonnes au bon type si nécessaire
        input_data['AGE'] = input_data['AGE'].astype(float)
        input_data['DIABETES'] = input_data['DIABETES'].astype(int)
        input_data['HIPERTENSION'] = input_data['HIPERTENSION'].astype(int)
        input_data['OBESITY'] = input_data['OBESITY'].astype(int)
        input_data['TOBACCO'] = input_data['TOBACCO'].astype(int)

        # Vérifier les valeurs manquantes
        if input_data.isnull().any().any():
            st.warning("Il y a des valeurs manquantes dans les données d'entrée. Elles doivent être traitées avant la prédiction.")
            input_data = input_data.fillna(input_data.mean())

        # Faire la prédiction
        try:
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.error("Le patient est susceptible d'être admis en soins intensifs.")
            else:
                st.success("Le patient n'est pas susceptible d'être admis en soins intensifs.")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
