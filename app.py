import streamlit as st
import joblib
import pandas as pd
import os

# Vérifier les dépendances nécessaires
try:
    import sklearn
except ImportError:
    st.error("Erreur : La bibliothèque 'scikit-learn' n'est pas installée. Veuillez l'installer avec la commande : `pip install scikit-learn`")
    st.stop()

# Titre de l'application
st.title("Prédiction d'admission en soins intensifs (COVID-19)")

# Fonction pour charger le modèle localement
@st.cache_resource  # Cache le modèle pour éviter de le recharger à chaque interaction
def load_model():
    model_path = 'covid_icu_model (1).pkl'  # Chemin relatif au même dossier que app.py
    
    # Vérifier que le fichier existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier du modèle '{model_path}' n'existe pas dans le dossier courant.")
    
    try:
        return joblib.load(model_path)  # Charger le modèle
    except Exception as e:
        raise RuntimeError(f"Impossible de charger le modèle : {e}")

# Charger le modèle avec vérification
try:
    model = load_model()
    st.success("Modèle chargé avec succès !")
except FileNotFoundError as e:
    st.error(f"Erreur : {e}")
    st.stop()
except RuntimeError as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# Formulaire pour saisir les données
st.sidebar.header("Saisissez les informations du patient")
age = st.sidebar.number_input('Âge', min_value=0, max_value=120, value=50)
diabetes = st.sidebar.selectbox('Diabète', [0, 1], help="0 = Non, 1 = Oui")
hypertension = st.sidebar.selectbox('Hypertension', [0, 1], help="0 = Non, 1 = Oui")
obesity = st.sidebar.selectbox('Obésité', [0, 1], help="0 = Non, 1 = Oui")
tobacco = st.sidebar.selectbox('Tabagisme', [0, 1], help="0 = Non, 1 = Oui")

# Bouton pour faire une prédiction
if st.sidebar.button('Prédire'):
    # Préparer les données d'entrée
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
            input_data = input_data.fillna(input_data.mean())  # Remplacer les valeurs manquantes par la moyenne
        
        # Faire la prédiction
        try:
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.error("Le patient est susceptible d'être admis en soins intensifs.")
            else:
                st.success("Le patient n'est pas susceptible d'être admis en soins intensifs.")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
