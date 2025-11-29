import pandas as pd
import os
import re
import spacy
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import platform  # New import for robust path handling

# --- CRITICAL FIX: Set NLTK Data Path Explicitly ---
# This ensures NLTK looks where the manual downloads typically go (C:\Users\User\AppData\Roaming\nltk_data)
# and prevents the persistent LookupError for 'punkt_tab'.
if platform.system() == "Windows":
    # Construct the path to the AppData\Roaming folder
    appdata_path = os.path.join(os.environ['APPDATA'], 'nltk_data')
    if os.path.exists(appdata_path) and appdata_path not in nltk.data.path:
        nltk.data.path.append(appdata_path)
# ---------------------------------------------------


# --- NLTK Resource Check and Download ---
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    # Added explicit checks for the resources
    nltk.data.find('tokenizers/punkt_tab/english/')
except LookupError:
    print("NLTK resources not found. Downloading...")
    # Using 'all' ensures everything is covered if initial download failed
    nltk.download('all', quiet=True)
    print("Download complete.")

# Load spaCy model for keyword/n-gram extraction
# NOTE: This line requires 'python -m spacy download en_core_web_sm' to be run once.
nlp_spacy = spacy.load("en_core_web_sm")

# --- Configuration ---
INPUT_FILE = os.path.join('data', 'cleaned_reviews.csv')
OUTPUT_FILE = os.path.join('data', 'analyzed_reviews.csv')
OUTPUT_THEMES_FILE = os.path.join('data', 'themes_summary.csv')


def preprocess_for_theme(text):
    """Basic cleaning for TF-IDF/Keyword extraction."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)  # Remove URLs, mentions, hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    # Filter stopwords and common fintech/bank words
    stop_words = set(
        stopwords.words('english') + ['bank', 'app', 'mobile', 'banking', 'cbe', 'boa', 'dashen', 'it', 'use', 'get',
                                      'can', 'one', 'would', 'like', 'good'])
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


def analyze_sentiment(df):
    """Use Hugging Face model for sentiment analysis."""
    print("Running Hugging Face Sentiment Analysis...")

    # Initialize pipeline. Using device=-1 for CPU.
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

    # Get results, converting to list first for batch processing
    results = sentiment_pipeline(df['review'].tolist())

    # Extract label and score
    df['sentiment_label'] = [r['label'] for r in results]
    df['sentiment_score'] = [r['score'] for r in results]

    # Map scores to a simple -1 to 1 scale for easier interpretation
    def map_score(row):
        score = row['sentiment_score']
        if row['sentiment_label'] == 'POSITIVE':
            return score
        elif row['sentiment_label'] == 'NEGATIVE':
            return -score
        return 0

    df['normalized_sentiment'] = df.apply(map_score, axis=1)

    # Use a simple threshold to also get a 'Neutral' label
    NEUTRAL_THRESHOLD = 0.5
    df['final_sentiment'] = np.select(
        [df['normalized_sentiment'] > NEUTRAL_THRESHOLD, df['normalized_sentiment'] < -NEUTRAL_THRESHOLD],
        ['Positive', 'Negative'],
        default='Neutral'
    )

    print("Sentiment analysis complete.")
    return df


def extract_themes_and_keywords(df):
    """Extract keywords and group them into themes using TF-IDF and N-Grams."""
    print("Starting Thematic Analysis...")

    # 1. Preprocess reviews for keyword extraction
    df['processed_review'] = df['review'].apply(preprocess_for_theme)

    # 2. Extract Keywords (Unigrams and Bigrams) using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    # Fit only on reviews that actually have content after preprocessing
    df_with_content = df[df['processed_review'].str.strip() != ""]
    if df_with_content.empty:
        print("Warning: No content left after preprocessing for thematic analysis.")
        return df.drop(columns=['processed_review'], errors='ignore')

    tfidf_matrix = vectorizer.fit_transform(df_with_content['processed_review'])
    feature_names = vectorizer.get_feature_names_out()

    # 3. Identify top keywords per bank for manual/rule-based grouping
    theme_summary = []

    # Rule-Based Theme Mapping for the challenge
    THEME_MAPPING_RULES = {
        'Account Access Issues': ['login', 'fingerprint', 'password', 'otp', 'access', 'username'],
        'Transaction Performance': ['slow', 'transfer', 'fast', 'loading', 'takes time', 'delay', 'transaction'],
        'User Interface & Experience': ['ui', 'interface', 'design', 'easy', 'confusing', 'complicated'],
        'Reliability & Bugs': ['crash', 'bug', 'error', 'reliable', 'stable', 'not working'],
        'Customer Support': ['support', 'customer care', 'help', 'call center']
    }

    def assign_themes(review_text):
        """Assign primary themes based on keyword presence."""
        themes = []
        text = review_text.lower()
        for theme, keywords in THEME_MAPPING_RULES.items():
            if any(k in text for k in keywords):
                themes.append(theme)
        return ", ".join(themes) if themes else 'General/Other'

    # Apply the rule-based theme assignment
    df['identified_theme'] = df['review'].apply(assign_themes)

    # Summarize themes per bank (for the report)
    for bank in df['bank'].unique():
        bank_df = df[df['bank'] == bank]

        # --- Adjusted Keyword Extraction Logic for safety ---
        bank_content = bank_df[bank_df['processed_review'].str.strip() != '']['processed_review']
        if not bank_content.empty:
            # Transform only the bank's non-empty content using the overall vectorizer
            bank_tfidf = vectorizer.transform(bank_content)

            # Sum TF-IDF scores for all features across all reviews for this bank
            feature_scores = bank_tfidf.sum(axis=0).A1

            # Get top 15 indices
            top_indices = np.argsort(feature_scores)[::-1][:15]
            top_keywords = [feature_names[i] for i in top_indices]
        else:
            top_keywords = []

        # Get theme distribution
        theme_counts = bank_df['identified_theme'].value_counts(normalize=True).mul(100).round(1).to_dict()

        theme_summary.append({
            'bank': bank,
            'theme_distribution': theme_counts,
            'top_keywords': top_keywords
        })

    pd.DataFrame(theme_summary).to_csv(OUTPUT_THEMES_FILE, index=False)
    print(f"Theme summary saved to {OUTPUT_THEMES_FILE}.")
    return df.drop(columns=['processed_review'])


if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Run T1_scrape_preprocess.py first.")
    else:
        df = pd.read_csv(INPUT_FILE)

        # 1. Sentiment Analysis
        df = analyze_sentiment(df)

        # 2. Thematic Analysis
        df = extract_themes_and_keywords(df)

        # Save results
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Final analyzed data saved to {OUTPUT_FILE}.")