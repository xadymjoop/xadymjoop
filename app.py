import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Titre de l'application
st.title("Entraînement et prédiction avec RandomForest")

# Charger les données
@st.cache_data
def load_data():
    # Remplacez ceci par votre propre chargement de données
    # Exemple : df = pd.read_csv('votre_fichier.csv')
    df = pd.DataFrame({
        'AGE': [50, 60, 70, 80, 90],
        'DIABETES': [0, 1, 0, 1, 0],
        'HIPERTENSION': [1, 0, 1, 0, 1],
        'OBESITY': [0, 1, 0, 1, 0],
        'TOBACCO': [1, 0, 1, 0, 1],
        'ICU': [0, 1, 0, 1, 0]  # Cible
    })
    return df

df = load_data()

# Afficher les données
st.write("## Données utilisées pour l'entraînement")
st.write(df)

# Sélectionner les caractéristiques et la cible
features = ['AGE', 'DIABETES', 'HIPERTENSION', 'OBESITY', 'TOBACCO']
X = df[features]
y = df['ICU']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle RandomForest
if st.button("Entraîner le modèle"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluer le modèle
    st.write("## Performance du modèle")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Classification Report:\n", classification_report(y_test, y_pred))

    # Sauvegarder le modèle
    joblib.dump(model, 'covid_icu_model.pkl')
    st.success("Modèle entraîné et sauvegardé avec succès !")

# Charger le modèle pour faire des prédictions
if st.button("Charger le modèle et faire une prédiction"):
    try:
        model = joblib.load('covid_icu_model.pkl')
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
