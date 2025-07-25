import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import joblib
import re
import string

st.set_page_config(page_title="E-Commerce Analysis Dashboard", layout="wide")
st.title("🛍️ E-Commerce Review Analysis Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["Sentiment Analysis", "Rating Regression", "Combined Analysis"]
)

# Sidebar for navigation
st.sidebar.title("Navigation")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["Sentiment Analysis", "Rating Regression", "Combined Analysis"]
)

# Text preprocessing function for sentiment analysis
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

# Load sentiment analysis model function
@st.cache_resource
def load_sentiment_model():
    """Load sentiment analysis model and vectorizer"""
    try:
        model = joblib.load("model/sentiment_model.pkl")
        vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("Sentiment analysis model files not found. Please train the model first.")
        return None, None

# Load regression model function
@st.cache_resource
def load_regression_model():
    """Load regression model and preprocessing objects"""
    try:
        scaler = joblib.load("model/minmax_scaler.pkl")
        label_encoders = {
            "customer_state": joblib.load("model/label_encoder_customer_state.pkl"),
            "delivery_delay_range": joblib.load("model/label_encoder_delivery_delay_range.pkl"),
            "freight_range": joblib.load("model/label_encoder_freight_range.pkl"),
            "price_range": joblib.load("model/label_encoder_price_range.pkl"),
            "product_category_name_english": joblib.load("model/label_encoder_product_category_name_english.pkl"),
            "same_state": joblib.load("model/label_encoder_same_state.pkl"),
            "seller_grade": joblib.load("model/label_encoder_seller_grade.pkl"),
            "seller_state": joblib.load("model/label_encoder_seller_state.pkl"),
        }
        model = joblib.load("model/best_model_XGBoost.pkl")
        return model, scaler, label_encoders
    except FileNotFoundError:
        st.error("Regression model files not found. Please train the model first.")
        return None, None, None

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment rating from text"""
    if model is None or vectorizer is None:
        return None, None
    
    cleaned_text = clean_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)[0]
    prediction_proba = model.predict_proba(text_vectorized)[0]
    confidence = max(prediction_proba)
    
    return prediction, confidence

# Sentiment Analysis Section
if analysis_type == "Sentiment Analysis":
    st.header("📝 Sentiment Analysis")
    st.write("Enter a review text to predict its rating (1-5 stars)")
    
    # Load sentiment model
    sentiment_model, tfidf_vectorizer = load_sentiment_model()
    
    # Text input
    review_text = st.text_area(
        "Enter review text:",
        placeholder="Type your review here...",
        height=150
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if review_text.strip():
            if sentiment_model is not None and tfidf_vectorizer is not None:
                rating, confidence = predict_sentiment(review_text, sentiment_model, tfidf_vectorizer)
                
                if rating is not None:
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Rating", f"{rating}/5 ⭐")
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Rating visualization
                    stars = "⭐" * int(rating) + "☆" * (5 - int(rating))
                    st.write(f"### {stars}")
                    
                    # Sentiment interpretation
                    if rating >= 4:
                        st.success("😊 Positive sentiment detected!")
                    elif rating >= 3:
                        st.warning("😐 Neutral sentiment detected!")
                    else:
                        st.error("😞 Negative sentiment detected!")
                        
                    # Display cleaned text
                    with st.expander("Processed Text"):
                        cleaned = clean_text(review_text)
                        st.write(cleaned)
            else:
                st.error("Please train the sentiment analysis model first.")
        else:
            st.warning("Please enter some review text.")

# Rating Regression Section
elif analysis_type == "Rating Regression":
    st.header("📊 Rating Regression Analysis")
    st.write("Upload e-commerce dataset to predict review scores based on features")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview Data")
    st.dataframe(df.head())

    # Drop duplicates
    df_cleaned = df.drop_duplicates(keep='first')

    # Target: avg_review_score
    if 'avg_review_score' not in df_cleaned.columns:
        st.error("Dataset must contain 'avg_review_score' column.")
    else:
        # Drop columns not used as features
        drop_cols = [
            'review_score', 'avg_review_score', 'sentiment_score',
            'order_id', 'product_id', 'customer_id', 'seller_id', 'customer_unique_id',
            'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                'order_delivered_customer_date', 'order_estimated_delivery_date'
        ]
        drop_cols = [col for col in drop_cols if col in df_cleaned.columns]
        X = df_cleaned.drop(columns=drop_cols)
        y = df_cleaned['avg_review_score']

        # Load regression model
        reg_model, scaler, label_encoders = load_regression_model()
        
        if reg_model is not None:
            # Replace encoding and scaling with preloaded objects
            for feature, encoder in label_encoders.items():
                if feature in X.columns:
                    X[feature] = encoder.transform(X[feature].astype(str))

            X_scaled = scaler.transform(X)

            # Use the loaded model for predictions
            y_pred = reg_model.predict(X_scaled)

            # Display predictions
            st.subheader("Predictions")
            st.write(y_pred)

            # Evaluation
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            st.subheader(f"Model Evaluation")
            st.write(f"MAE: {mean_absolute_error(y, y_pred):.4f}")
            st.write(f"RMSE: {mean_squared_error(y, y_pred, squared=False):.4f}")
            st.write(f"R2 Score: {r2_score(y, y_pred):.4f}")

            # Visualization
            st.subheader("Actual vs Predicted avg_review_score")
            chart_df = pd.DataFrame({"Actual": y, "Predicted": y_pred})
            st.line_chart(chart_df.reset_index(drop=True))

            st.subheader("Feature Importances (if available)")
            if hasattr(reg_model, "feature_importances_"):
                importances = reg_model.feature_importances_
                feat_names = X.columns
                imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
                st.bar_chart(imp_df.set_index("Feature"))
            elif hasattr(reg_model, "coef_"):
                coef = reg_model.coef_
                feat_names = X.columns
                coef_df = pd.DataFrame({"Feature": feat_names, "Coefficient": coef})
                st.bar_chart(coef_df.set_index("Feature"))
            else:
                st.write("Feature importances not available for this model.")

# Combined Analysis Section
elif analysis_type == "Combined Analysis":
    st.header("🔄 Combined Analysis")
    st.write("Get both sentiment analysis from text and feature-based predictions")
    
    # Load both models
    sentiment_model, tfidf_vectorizer = load_sentiment_model()
    reg_model, scaler, label_encoders = load_regression_model()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Text Sentiment Analysis")
        review_text = st.text_area(
            "Enter review text:",
            placeholder="Type your review here...",
            height=150
        )
        
        if st.button("Analyze Text Sentiment", type="primary"):
            if review_text.strip():
                if sentiment_model is not None and tfidf_vectorizer is not None:
                    rating, confidence = predict_sentiment(review_text, sentiment_model, tfidf_vectorizer)
                    
                    if rating is not None:
                        st.metric("Text-based Rating", f"{rating}/5 ⭐")
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Rating visualization
                        stars = "⭐" * int(rating) + "☆" * (5 - int(rating))
                        st.write(f"### {stars}")
    
    with col2:
        st.subheader("📊 Feature-based Prediction")
        uploaded_file_combined = st.file_uploader(
            "Upload CSV for feature-based prediction:", 
            type=["csv"], 
            key="combined_upload"
        )
        
        if uploaded_file_combined:
            df = pd.read_csv(uploaded_file_combined)
            st.write("Data preview:")
            st.dataframe(df.head(3))
            
            if 'avg_review_score' in df.columns and reg_model is not None:
                # Process data for regression
                df_cleaned = df.drop_duplicates(keep='first')
                drop_cols = [
                    'review_score', 'avg_review_score', 'sentiment_score',
                    'order_id', 'product_id', 'customer_id', 'seller_id', 'customer_unique_id',
                    'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                    'order_delivered_customer_date', 'order_estimated_delivery_date'
                ]
                drop_cols = [col for col in drop_cols if col in df_cleaned.columns]
                X = df_cleaned.drop(columns=drop_cols)
                y = df_cleaned['avg_review_score']
                
                # Apply preprocessing
                for feature, encoder in label_encoders.items():
                    if feature in X.columns:
                        X[feature] = encoder.transform(X[feature].astype(str))
                
                X_scaled = scaler.transform(X)
                y_pred = reg_model.predict(X_scaled)
                
                # Show sample predictions
                st.write("Sample feature-based predictions:")
                sample_df = pd.DataFrame({
                    'Actual': y.head(5).values,
                    'Predicted': y_pred[:5]
                })
                st.dataframe(sample_df)
                
                # Overall metrics
                from sklearn.metrics import mean_absolute_error, r2_score
                st.metric("MAE", f"{mean_absolute_error(y, y_pred):.3f}")
                st.metric("R² Score", f"{r2_score(y, y_pred):.3f}")

else:
    st.info("Please upload a CSV file to start the analysis.")
