#!/usr/bin/env python3
"""
Quick training script for sentiment analysis model to integrate with Streamlit
"""

import pandas as pd
import numpy as np
import re
import string
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

def clean_text(text):
    """Clean text for sentiment analysis"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def main():
    print("Loading dataset...")
    
    # Try to load the dataset with different possible names
    dataset_files = [
        "review_product_di_shoppe_dan_tokopedia.csv",
        "list-ecommerce-for-delivery-and-review-prediction.csv"
    ]
    
    df = None
    for file in dataset_files:
        if os.path.exists(file):
            try:
                # Try different delimiters and encodings
                df = pd.read_csv(file, delimiter=';', encoding='latin1')
                if 'rating' not in df.columns and 'review_score' in df.columns:
                    df = pd.read_csv(file, encoding='latin1')
                print(f"Successfully loaded: {file}")
                break
            except:
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                    print(f"Successfully loaded: {file}")
                    break
                except:
                    continue
    
    if df is None:
        print("Error: Could not find or load dataset file")
        print(f"Looking for: {dataset_files}")
        return
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Identify text and rating columns
    text_col = None
    rating_col = None
    
    # Look for text column
    text_candidates = ['review', 'review_text', 'text', 'comment', 'ulasan']
    for col in text_candidates:
        if col in df.columns:
            text_col = col
            break
    
    # Look for rating column  
    rating_candidates = ['rating', 'review_score', 'score', 'nilai']
    for col in rating_candidates:
        if col in df.columns:
            rating_col = col
            break
    
    if text_col is None or rating_col is None:
        print(f"Could not find required columns. Available: {list(df.columns)}")
        return
    
    print(f"Using text column: {text_col}")
    print(f"Using rating column: {rating_col}")
    
    # Prepare data
    df_clean = df[[text_col, rating_col]].dropna()
    df_clean = df_clean[df_clean[text_col].str.len() > 10]  # Filter very short reviews
    
    # Ensure ratings are 1-5
    df_clean[rating_col] = pd.to_numeric(df_clean[rating_col], errors='coerce')
    df_clean = df_clean[df_clean[rating_col].between(1, 5)]
    
    print(f"Clean dataset shape: {df_clean.shape}")
    print(f"Rating distribution:\n{df_clean[rating_col].value_counts().sort_index()}")
    
    if len(df_clean) < 100:
        print("Error: Not enough clean data for training")
        return
    
    # Clean text
    print("Cleaning text...")
    df_clean['clean_text'] = df_clean[text_col].apply(clean_text)
    df_clean = df_clean[df_clean['clean_text'].str.len() > 5]
    
    # Split data
    X = df_clean['clean_text']
    y = df_clean[rating_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # TF-IDF Vectorization
    print("Creating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': MultinomialNB()
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} with accuracy: {best_accuracy:.4f}")
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Save models
    print("Saving models...")
    joblib.dump(best_model, 'model/sentiment_model.pkl')
    joblib.dump(tfidf_vectorizer, 'model/tfidf_vectorizer.pkl')
    joblib.dump(clean_text, 'model/text_cleaner.pkl')
    
    # Save model info
    model_info = {
        'best_model_name': best_name,
        'accuracy': best_accuracy,
        'features': X_train_tfidf.shape[1],
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'text_column': text_col,
        'rating_column': rating_col
    }
    joblib.dump(model_info, 'model/model_info.pkl')
    
    print(f"\nModels saved successfully!")
    print(f"- Sentiment Model: model/sentiment_model.pkl")
    print(f"- TF-IDF Vectorizer: model/tfidf_vectorizer.pkl")
    print(f"- Text Cleaner: model/text_cleaner.pkl")
    print(f"- Model Info: model/model_info.pkl")
    
    # Test with sample reviews
    test_reviews = [
        "This product is amazing! Great quality and fast delivery.",
        "Terrible product, very disappointed. Bad quality.",
        "It's okay, nothing special but acceptable.",
        "Excellent! Highly recommend this product.",
        "Poor quality, not worth the money."
    ]
    
    print("\nTesting with sample reviews:")
    for i, review in enumerate(test_reviews, 1):
        cleaned = clean_text(review)
        review_tfidf = tfidf_vectorizer.transform([cleaned])
        prediction = best_model.predict(review_tfidf)[0]
        try:
            confidence = max(best_model.predict_proba(review_tfidf)[0])
        except:
            confidence = 1.0
        
        print(f"{i}. '{review}' -> Rating: {prediction}, Confidence: {confidence:.3f}")

if __name__ == "__main__":
    main()
