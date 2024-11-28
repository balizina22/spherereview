import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk
from transformers import pipeline, TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Télécharger les stopwords une seule fois
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Liste des stopwords en français
stop_words = set(stopwords.words('french'))
custom_stop_words = {"!", ".", ",", ":", ";", "?", "a", "deux", "4", "plus", "si", "peu", "tout", "sans", "vin", "trop"}
stop_words.update(custom_stop_words)

# Initialisation de l'analyseur VADER
sia = SentimentIntensityAnalyzer()

def analyze_sentiment_textblob(standardized_text):
    if not isinstance(standardized_text, str): 
        return 'Neutre'  # Si ce n'est pas une chaîne, renvoyer un sentiment par défaut
    analysis = TextBlob(standardized_text)
    if analysis.sentiment.polarity > 0:
        return 'Positif'
    elif analysis.sentiment.polarity == 0:
        return 'Neutre'
    else:
        return 'Négatif'


def analyze_sentiment_vader(text):
    # Vérifier si le texte est un nombre ou NaN, et le convertir en chaîne vide
    if not isinstance(text, str):
        text = str(text) if text is not None else ""  # Convertir en chaîne vide si text est None ou NaN

    # Utiliser l'analyseur VADER
    scores = sia.polarity_scores(text)  # Obtenir les scores de sentiment
    if scores['compound'] >= 0.05:
        return "Positif"
    elif scores['compound'] <= -0.05:
        return "Négatif"
    else:
        return "Neutre"


# Pipeline Hugging Face pour l'analyse des sentiments
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    truncation=True  # Activer la troncation
)

# Fonction pour normaliser les labels TensorFlow
def normalize_tensorflow(label):
    if label == "LABEL_5":
        return "Positif"
    elif label == "LABEL_4":
        return "Positif"
    elif label == "LABEL_3":
        return "Neutre"
    elif label == "LABEL_2":
        return "Négatif"
    elif label == "LABEL_1":
        return "Négatif"
    
# modèle BERT pré-entraîné et le tokenizer
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME)

# Fonction pour analyser les sentiments avec BERT-TensorFlow
def analyze_sentiment_tensorflow(text):
    # Vérifier et convertir en chaîne, gérer les valeurs manquantes
    if not isinstance(text, str):
        text = str(text) if text is not None else ""  # Convertir en chaîne vide si text est None ou NaN

    # Tokeniser et tronquer le texte
    inputs = tokenizer(
        text,
        return_tensors="tf",
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    # Prédiction avec le modèle BERT
    outputs = model(inputs)
    scores = outputs.logits[0].numpy()
    # Convertir les scores en probabilités
    probabilities = tf.nn.softmax(scores).numpy()
    # Identifier le label avec la probabilité la plus élevée
    sentiment_label = probabilities.argmax() + 1  # Les labels commencent à 1
    sentiment_score = probabilities.max()  # Score de confiance
    return f"LABEL_{sentiment_label}", sentiment_score


# Titre de l'application
st.title("Analyse des commentaires du restaurant Sphere")

# Chargement des données
uploaded_file = st.file_uploader("Téléchargez le fichier CSV", type="csv")
if uploaded_file:
    # Lire le CSV
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données :", data.head())

    # Analyse descriptive
    st.subheader("Analyse descriptive")
    st.write("Statistiques descriptives :", data.describe())

    # Visualisation 1 : Histogramme des notes
    st.subheader("Histogramme des notes")
    fig, ax = plt.subplots()
    data['rating'].plot(kind='hist', bins=20, ax=ax, title='Répartition des notes')
    st.pyplot(fig)

    # Visualisation 2 : Répartition des types de clients
    st.subheader("Répartition des types de clients")
    fig, ax = plt.subplots()
    type_distribution = data['type'].value_counts()
    type_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='Pastel1', ax=ax)
    ax.set_ylabel("")  # Supprimer le label sur l'axe Y
    st.pyplot(fig)

    # Visualisation 3 : Distribution des notes (histogramme simple)
    st.subheader("Distribution des notes attribuées")
    fig, ax = plt.subplots()
    ax.hist(data['rating'], bins=5, range=(1, 6))
    ax.set_xlabel('Notes')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution des notes attribuées')
    st.pyplot(fig)

    # Visualisation 4 : Heatmap Langue vs Type
    st.subheader("Heatmap : Langue vs Type")
    fig, ax = plt.subplots(figsize=(8, 8))
    df_2dhist = pd.DataFrame({
        x_label: grp['type'].value_counts()
        for x_label, grp in data.groupby('langue')
    })
    sns.heatmap(df_2dhist, cmap='viridis', ax=ax)
    ax.set_xlabel('Langue')
    ax.set_ylabel('Type')
    st.pyplot(fig)

    # Visualisation 5 : Évolution des notes au fil du temps
    st.subheader("Évolution des notes au fil du temps")
    fig, ax = plt.subplots()
    data.groupby('date_publication')['rating'].mean().plot(ax=ax)
    ax.set_xlabel('Date de publication')
    ax.set_ylabel('Note moyenne')
    ax.set_title('Évolution des notes au fil du temps')
    st.pyplot(fig)

    # Visualisation 6 : Heatmap des corrélations
    st.subheader("Corrélations entre les aspects")
    selected_columns = ["Qualité/prix", "Service", "Cuisine", "Ambiance"]
    if set(selected_columns).issubset(data.columns):
        filtered_df = data[selected_columns]
        correlation_matrix = filtered_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=0.8, vmax=1, square=True, ax=ax)
        ax.set_title("Corrélation entre les Aspects")
        st.pyplot(fig)

    # Visualisation 7 : Moyenne des notes par langue
    st.subheader("Moyenne des Notes par Langue")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=data, x='langue', y='rating', palette='Set2', ci=None, ax=ax)
    ax.set_title('Moyenne des Notes par Langue')
    ax.set_xlabel('Langue')
    ax.set_ylabel('Moyenne des Notes')
    st.pyplot(fig)

    # Visualisation 8 : Distribution des notes par type de visite
    st.subheader("Distribution des Notes par Type de Visite")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x='type', y='rating', palette='muted', ax=ax)
    ax.set_title('Distribution des Notes par Type de Visite')
    ax.set_xlabel('Type de Visite')
    ax.set_ylabel('Notes')
    st.pyplot(fig)

    # Visualisation 9 : Moyennes des sous-catégories par type de client
    st.subheader("Moyennes des sous-catégories par type de client")
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_by_type = data.groupby('type')[['Qualité/prix', 'Service', 'Cuisine', 'Ambiance']].mean()
    avg_by_type.plot(kind='bar', stacked=True, ax=ax, cmap='viridis')
    ax.set_title("Moyennes des sous-catégories par type de client")
    ax.set_xlabel("Type de client")
    ax.set_ylabel("Moyenne")
    ax.legend(title="Sous-catégories", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

    # Visualisation 10 : Nuage de mots
    st.subheader("Nuage de mots (Standardized Text)")
    if 'standardized_text' in data.columns:
        # Combiner toutes les données de la colonne 'standardized_text'
        all_reviews = " ".join(data['standardized_text'].astype(str))

        # Générer le nuage de mots
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(all_reviews)

        # Afficher le nuage de mots
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    # Analyse des sentiments
st.subheader("Analyse des Sentiments")

# Sélection de la méthode d'analyse
method = st.radio("Choisir la méthode", ("TextBlob", "VADER", "BERT"))

if method == "TextBlob":
    sentiment_results = data['standardized_text'].apply(analyze_sentiment_textblob)
elif method == "VADER":
    sentiment_results = data['standardized_text'].apply(analyze_sentiment_vader)
elif method == "BERT":
    sentiment_results = data['standardized_text'].apply(lambda x: analyze_sentiment_tensorflow(x)[0])

# S'afficher seulement si sentiment_results est défini
if 'sentiment_results' in locals():
    st.write("Analyse des sentiments :", sentiment_results)

    # Pourcentage de sentiments dans les données
    sentiment_count = sentiment_results.value_counts()
    st.write("Répartition des sentiments :", sentiment_count)

    # Afficher la matrice de confusion si les sentiments sont disponibles
    if 'sentiment' in data.columns:
        st.subheader("Matrice de confusion")
        
        # Vérification des valeurs dans y_true
        y_true = data['sentiment'].dropna()  # Enlever les NaN dans y_true
        y_pred = sentiment_results

        # Vérifier si y_true contient des valeurs attendues
        valid_labels = ["Positif", "Neutre", "Négatif"]
        y_true = y_true[y_true.isin(valid_labels)]  # Filtrer y_true pour ne garder que les labels valides
        y_pred = y_pred[y_true.index]  # Aligner y_pred avec les indices valides de y_true

        # Si y_true contient au moins un label valide, afficher la matrice de confusion
        if not y_true.empty:
            cm = confusion_matrix(y_true, y_pred, labels=valid_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_labels)
            disp.plot(cmap='Blues')
            st.pyplot()
        else:
            st.warning("La matrice de confusion ne peut être calculée car les labels de sentiment ne sont pas présents dans les données.")
    
    # Classification report
    st.subheader("Rapport de classification")
    if 'sentiment' in data.columns:
        y_true = data['sentiment'].dropna()
        y_pred = sentiment_results

        # Filtrer pour ne garder que les labels valides
        y_true = y_true[y_true.isin(valid_labels)]
        y_pred = y_pred[y_true.index]

        # Afficher le rapport de classification si y_true contient des valeurs valides
        if not y_true.empty:
            report = classification_report(y_true, y_pred)
            st.text(report)
        else:
            st.warning("Le rapport de classification ne peut être généré car les labels de sentiment ne sont pas présents dans les données.")
