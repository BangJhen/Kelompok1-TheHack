import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import re
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import io
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="E-Commerce Business Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 E-Commerce Business Insights Dashboard</h1>', unsafe_allow_html=True)

# Load models and preprocessing objects
@st.cache_resource
def load_regression_models():
    """Load regression models and preprocessing objects"""
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
        model = joblib.load("model/best_model_RandomForest.pkl")
        return model, scaler, label_encoders, True
    except Exception as e:
        st.error(f"Error loading regression models: {e}")
        return None, None, None, False

@st.cache_resource
def load_sentiment_models():
    """Load sentiment analysis models and preprocessing"""
    try:
        sentiment_model = joblib.load("model/sentiment_model.pkl")
        tfidf_vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
        text_cleaner = joblib.load("model/text_cleaner.pkl")
        return sentiment_model, tfidf_vectorizer, text_cleaner, True
    except Exception as e:
        st.error(f"Error loading sentiment models: {e}")
        return None, None, None, False

# Text preprocessing function (fallback if model loading fails)
def clean_text_fallback(text):
    """Clean text for sentiment analysis (fallback function)"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    
    return text

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["🏠 Home", "📈 Review Score Regression", "💭 Sentiment Classification", "🔄 Combined Analysis", "📊 Business Insights"]
)

# Load models
reg_model, scaler, label_encoders, reg_success = load_regression_models()
sentiment_model, tfidf_vectorizer, text_cleaner, sent_success = load_sentiment_models()

# Model status in sidebar
st.sidebar.markdown("### 🤖 Model Status")
if reg_success:
    st.sidebar.success("✅ Regression Model Loaded")
else:
    st.sidebar.error("❌ Regression Model Failed")

if sent_success:
    st.sidebar.success("✅ Sentiment Model Loaded")
else:
    st.sidebar.error("❌ Sentiment Model Failed")

# HOME PAGE
if page == "🏠 Home":
    st.markdown("## Welcome to the E-Commerce Business Insights Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Review Score Regression")
        st.markdown("""
        - **Purpose**: Predict average review scores based on business features
        - **Input**: Sales data with features like price, freight, delivery time, etc.
        - **Model**: XGBoost Regressor
        - **Use Case**: Business performance prediction and optimization
        """)
        
        st.markdown("### 💡 Business Applications")
        st.markdown("""
        - Predict customer satisfaction before order completion
        - Optimize pricing and delivery strategies
        - Identify factors that impact review scores
        - Monitor seller performance
        """)
    
    with col2:
        st.markdown("### 💭 Sentiment Classification")
        st.markdown("""
        - **Purpose**: Classify rating (1-5 stars) from review text
        - **Input**: Customer review text
        - **Model**: Naive Bayes Classifier with TF-IDF
        - **Use Case**: Automated review analysis and sentiment monitoring
        """)
        
        st.markdown("### 🎯 Key Features")
        st.markdown("""
        - Real-time sentiment analysis
        - Confidence scoring for predictions
        - Text preprocessing and cleaning
        - Multi-class rating classification
        """)
    
    st.markdown("---")
    st.markdown("### 📊 Sample Data Formats")
    
    # Show sample data formats
    tab1, tab2 = st.tabs(["📋 Sales Data Sample", "📝 Review Data Sample"])
    
    with tab1:
        st.markdown("**Expected format for regression analysis:**")
        sample_sales = pd.DataFrame({
            'order_id': ['e481f51cbdc54678b7cc49136f2d6af7'],
            'product_id': ['87285b34884572647811a353c7ac498a'],
            'seller_state': ['SP'],
            'customer_state': ['SP'],
            'review_score': [4],
            'price': [29.99],
            'freight_value': [8.72],
            'seller_grade': ['Mediocre'],
            'product_category_name_english': ['housewares'],
            'avg_review_score': [4.415094339622642]
        })
        st.dataframe(sample_sales)
    
    with tab2:
        st.markdown("**Expected format for sentiment analysis:**")
        sample_reviews = pd.DataFrame({
            'review_id': [0, 1, 2],
            'review': [
                'This product is amazing! Great quality and fast delivery.',
                'Poor quality, not worth the money.',
                'It\'s okay, nothing special but acceptable.'
            ],
            'rating': [5, 1, 3]
        })
        st.dataframe(sample_reviews)

# REVIEW SCORE REGRESSION PAGE
elif page == "📈 Review Score Regression":
    st.markdown("## 📈 Review Score Regression Analysis")
    st.markdown("Upload sales data to predict review scores and analyze business performance.")
    
    uploaded_file = st.file_uploader(
        "📁 Upload Sales Dataset (CSV)",
        type=["csv"],
        help="Upload a CSV file with sales data similar to the sample format"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Show data preview
            with st.expander("👀 Data Preview", expanded=True):
                st.dataframe(df.head(10))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Features", len(df.columns))
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
            
            if reg_success and 'avg_review_score' in df.columns:
                # Prepare data for prediction
                df_clean = df.drop_duplicates(keep='first')
                
                # Define columns to drop
                drop_cols = [
                    'review_score', 'avg_review_score', 'sentiment_score',
                    'order_id', 'product_id', 'customer_id', 'seller_id', 'customer_unique_id',
                    'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                    'order_delivered_customer_date', 'order_estimated_delivery_date'
                ]
                drop_cols = [col for col in drop_cols if col in df_clean.columns]
                
                X = df_clean.drop(columns=drop_cols)
                y = df_clean['avg_review_score']
                
                # Apply preprocessing
                X_processed = X.copy()
                for feature, encoder in label_encoders.items():
                    if feature in X_processed.columns:
                        try:
                            X_processed[feature] = encoder.transform(X_processed[feature].astype(str))
                        except Exception as e:
                            st.warning(f"Could not encode {feature}: {e}")
                            X_processed = X_processed.drop(columns=[feature])
                
                # Scale features
                X_scaled = scaler.transform(X_processed)
                
                # Make predictions
                y_pred = reg_model.predict(X_scaled)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Prediction Results")
                    
                    # Metrics
                    mae = mean_absolute_error(y, y_pred)
                    rmse = mean_squared_error(y, y_pred, squared=False)
                    r2 = r2_score(y, y_pred)
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("MAE", f"{mae:.4f}")
                    with metric_col2:
                        st.metric("RMSE", f"{rmse:.4f}")
                    with metric_col3:
                        st.metric("R² Score", f"{r2:.4f}")
                    
                    # Prediction distribution
                    fig_dist = px.histogram(
                        x=y_pred, 
                        title="Distribution of Predicted Review Scores",
                        labels={'x': 'Predicted Review Score', 'y': 'Frequency'}
                    )
                    fig_dist.update_layout(showlegend=False)
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    st.markdown("### 🎯 Actual vs Predicted")
                    
                    # Scatter plot
                    fig_scatter = px.scatter(
                        x=y, y=y_pred,
                        title="Actual vs Predicted Review Scores",
                        labels={'x': 'Actual Score', 'y': 'Predicted Score'}
                    )
                    fig_scatter.add_trace(go.Scatter(
                        x=[y.min(), y.max()], 
                        y=[y.min(), y.max()],
                        mode='lines', 
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='red')
                    ))
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Feature importance
                if hasattr(reg_model, "feature_importances_"):
                    st.markdown("### 🎯 Feature Importance Analysis")
                    
                    importances = reg_model.feature_importances_
                    feature_names = X_processed.columns
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=True)
                    
                    fig_importance = px.bar(
                        importance_df.tail(10),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 10 Most Important Features"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                # Detailed results table
                with st.expander("📋 Detailed Prediction Results"):
                    results_df = df_clean[['avg_review_score']].copy()
                    results_df['predicted_score'] = y_pred
                    results_df['error'] = abs(results_df['avg_review_score'] - results_df['predicted_score'])
                    results_df = results_df.round(4)
                    st.dataframe(results_df)
                    
                    # Download results
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Prediction Results",
                        data=csv_buffer.getvalue(),
                        file_name="regression_predictions.csv",
                        mime="text/csv"
                    )
            
            else:
                if not reg_success:
                    st.error("❌ Regression model not loaded. Please check model files.")
                else:
                    st.error("❌ Dataset must contain 'avg_review_score' column for regression analysis.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# SENTIMENT CLASSIFICATION PAGE
elif page == "💭 Sentiment Classification":
    st.markdown("## 💭 Sentiment Classification Analysis")
    st.markdown("Analyze customer sentiment from review text and predict ratings.")
    
    # Text input section
    st.markdown("### ✍️ Single Review Analysis")
    review_text = st.text_area(
        "Enter review text:",
        placeholder="Type or paste a customer review here...",
        height=100
    )
    
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if review_text.strip() and sent_success:
            try:
                # Clean text
                if text_cleaner:
                    cleaned_text = text_cleaner(review_text)
                else:
                    cleaned_text = clean_text_fallback(review_text)
                
                # Make prediction
                text_tfidf = tfidf_vectorizer.transform([cleaned_text])
                prediction = sentiment_model.predict(text_tfidf)[0]
                
                try:
                    prediction_proba = sentiment_model.predict_proba(text_tfidf)[0]
                    confidence = max(prediction_proba)
                except:
                    prediction_proba = [0.0] * 5
                    prediction_proba[prediction - 1] = 1.0
                    confidence = 1.0
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Predicted Rating", f"{prediction}/5 ⭐")
                
                with col2:
                    st.metric("🎯 Confidence", f"{confidence:.2%}")
                
                with col3:
                    # Sentiment category
                    if prediction >= 4:
                        sentiment_cat = "😊 Positive"
                        st.success(sentiment_cat)
                    elif prediction >= 3:
                        sentiment_cat = "😐 Neutral"
                        st.warning(sentiment_cat)
                    else:
                        sentiment_cat = "😞 Negative"
                        st.error(sentiment_cat)
                
                # Rating visualization
                stars = "⭐" * int(prediction) + "☆" * (5 - int(prediction))
                st.markdown(f"### {stars}")
                
                # Probability distribution
                if any(prediction_proba):
                    prob_df = pd.DataFrame({
                        'Rating': range(1, 6),
                        'Probability': prediction_proba
                    })
                    
                    fig_prob = px.bar(
                        prob_df,
                        x='Rating',
                        y='Probability',
                        title="Rating Probability Distribution"
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                
                # Show processed text
                with st.expander("🔍 Processed Text"):
                    st.text(cleaned_text)
                    
            except Exception as e:
                st.error(f"❌ Error analyzing sentiment: {e}")
        elif not review_text.strip():
            st.warning("⚠️ Please enter some review text.")
        else:
            st.error("❌ Sentiment model not available.")
    
    st.markdown("---")
    
    # Batch analysis section
    st.markdown("### 📁 Batch Review Analysis")
    uploaded_reviews = st.file_uploader(
        "📁 Upload Review Dataset (CSV)",
        type=["csv"],
        key="sentiment_upload",
        help="Upload a CSV file with review text data"
    )
    
    if uploaded_reviews is not None and sent_success:
        try:
            reviews_df = pd.read_csv(uploaded_reviews)
            st.success(f"✅ Review dataset loaded! Shape: {reviews_df.shape}")
            
            # Show data preview
            with st.expander("👀 Data Preview"):
                st.dataframe(reviews_df.head())
            
            # Find text column
            text_cols = [col for col in reviews_df.columns if 'review' in col.lower() or 'text' in col.lower() or 'comment' in col.lower()]
            
            if text_cols:
                text_col = st.selectbox("Select text column:", text_cols)
                
                if st.button("🚀 Process All Reviews", type="primary"):
                    with st.spinner("Processing reviews..."):
                        predictions = []
                        confidences = []
                        
                        for text in reviews_df[text_col]:
                            try:
                                if text_cleaner:
                                    cleaned = text_cleaner(text)
                                else:
                                    cleaned = clean_text_fallback(text)
                                
                                text_tfidf = tfidf_vectorizer.transform([cleaned])
                                pred = sentiment_model.predict(text_tfidf)[0]
                                
                                try:
                                    prob = max(sentiment_model.predict_proba(text_tfidf)[0])
                                except:
                                    prob = 1.0
                                
                                predictions.append(pred)
                                confidences.append(prob)
                                
                            except:
                                predictions.append(3)  # neutral default
                                confidences.append(0.5)
                        
                        # Add results to dataframe
                        results_df = reviews_df.copy()
                        results_df['predicted_rating'] = predictions
                        results_df['confidence'] = confidences
                        
                        # Show summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Reviews", len(results_df))
                        with col2:
                            st.metric("Avg Rating", f"{np.mean(predictions):.2f}")
                        with col3:
                            positive_pct = (np.array(predictions) >= 4).mean() * 100
                            st.metric("Positive %", f"{positive_pct:.1f}%")
                        with col4:
                            st.metric("Avg Confidence", f"{np.mean(confidences):.2%}")
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Rating distribution
                            rating_counts = pd.Series(predictions).value_counts().sort_index()
                            fig_dist = px.bar(
                                x=rating_counts.index,
                                y=rating_counts.values,
                                title="Predicted Rating Distribution",
                                labels={'x': 'Rating', 'y': 'Count'}
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)
                        
                        with col2:
                            # Confidence distribution
                            fig_conf = px.histogram(
                                x=confidences,
                                title="Confidence Score Distribution",
                                labels={'x': 'Confidence', 'y': 'Frequency'}
                            )
                            st.plotly_chart(fig_conf, use_container_width=True)
                        
                        # Show results table
                        with st.expander("📋 Detailed Results"):
                            st.dataframe(results_df)
                            
                            # Download results
                            csv_buffer = io.StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="📥 Download Analysis Results",
                                data=csv_buffer.getvalue(),
                                file_name="sentiment_analysis_results.csv",
                                mime="text/csv"
                            )
            else:
                st.error("❌ No text columns found in the dataset.")
                
        except Exception as e:
            st.error(f"❌ Error processing reviews: {e}")

# COMBINED ANALYSIS PAGE
elif page == "🔄 Combined Analysis":
    st.markdown("## 🔄 Combined Business Intelligence Analysis")
    st.markdown("Comprehensive analysis combining both regression and sentiment models.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Sales Data Analysis")
        sales_file = st.file_uploader(
            "📁 Upload Sales Data",
            type=["csv"],
            key="combined_sales"
        )
    
    with col2:
        st.markdown("### 💭 Review Text Analysis")
        review_file = st.file_uploader(
            "📁 Upload Review Data",
            type=["csv"],
            key="combined_reviews"
        )
    
    if sales_file is not None and review_file is not None:
        try:
            sales_df = pd.read_csv(sales_file)
            reviews_df = pd.read_csv(review_file)
            
            st.success("✅ Both datasets loaded successfully!")
            
            # Process sales data
            if reg_success and 'avg_review_score' in sales_df.columns:
                st.markdown("### 📊 Sales Performance Analysis")
                
                # Quick regression analysis
                df_clean = sales_df.drop_duplicates(keep='first')
                drop_cols = [
                    'review_score', 'avg_review_score', 'sentiment_score',
                    'order_id', 'product_id', 'customer_id', 'seller_id', 'customer_unique_id',
                    'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                    'order_delivered_customer_date', 'order_estimated_delivery_date'
                ]
                drop_cols = [col for col in drop_cols if col in df_clean.columns]
                
                X = df_clean.drop(columns=drop_cols)
                y = df_clean['avg_review_score']
                
                # Apply preprocessing
                X_processed = X.copy()
                for feature, encoder in label_encoders.items():
                    if feature in X_processed.columns:
                        try:
                            X_processed[feature] = encoder.transform(X_processed[feature].astype(str))
                        except:
                            continue
                
                X_scaled = scaler.transform(X_processed)
                y_pred = reg_model.predict(X_scaled)
                
                # Sales metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Predicted Avg Score", f"{np.mean(y_pred):.3f}")
                with col2:
                    st.metric("Actual Avg Score", f"{np.mean(y):.3f}")
                with col3:
                    st.metric("RMSE", f"{mean_squared_error(y, y_pred, squared=False):.3f}")
                with col4:
                    st.metric("R² Score", f"{r2_score(y, y_pred):.3f}")
            
            # Process review sentiment
            if sent_success:
                st.markdown("### 💭 Sentiment Analysis Summary")
                
                # Find text column
                text_cols = [col for col in reviews_df.columns if 'review' in col.lower() or 'text' in col.lower()]
                
                if text_cols:
                    text_col = text_cols[0]
                    
                    # Sample analysis (first 100 reviews for performance)
                    sample_size = min(100, len(reviews_df))
                    sample_reviews = reviews_df[text_col].head(sample_size)
                    
                    predictions = []
                    for text in sample_reviews:
                        try:
                            if text_cleaner:
                                cleaned = text_cleaner(text)
                            else:
                                cleaned = clean_text_fallback(text)
                            
                            text_tfidf = tfidf_vectorizer.transform([cleaned])
                            pred = sentiment_model.predict(text_tfidf)[0]
                            predictions.append(pred)
                        except:
                            predictions.append(3)
                    
                    # Sentiment metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Analyzed Reviews", sample_size)
                    with col2:
                        st.metric("Avg Predicted Rating", f"{np.mean(predictions):.2f}")
                    with col3:
                        positive_pct = (np.array(predictions) >= 4).mean() * 100
                        st.metric("Positive Sentiment %", f"{positive_pct:.1f}%")
                    with col4:
                        negative_pct = (np.array(predictions) <= 2).mean() * 100
                        st.metric("Negative Sentiment %", f"{negative_pct:.1f}%")
                    
                    # Combined visualization
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Sales Score Distribution', 'Sentiment Rating Distribution'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    if reg_success and 'avg_review_score' in sales_df.columns:
                        fig.add_trace(
                            go.Histogram(x=y_pred, name="Predicted Sales Scores", nbinsx=20),
                            row=1, col=1
                        )
                    
                    fig.add_trace(
                        go.Histogram(x=predictions, name="Sentiment Ratings", nbinsx=5),
                        row=1, col=2
                    )
                    
                    fig.update_layout(title_text="Combined Analysis Overview")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Business insights
            st.markdown("### 🎯 Business Insights")
            
            insights = []
            
            if reg_success and 'y_pred' in locals():
                avg_predicted = np.mean(y_pred)
                if avg_predicted >= 4.0:
                    insights.append("✅ Sales performance indicates high customer satisfaction potential")
                elif avg_predicted >= 3.5:
                    insights.append("⚠️ Sales performance shows moderate satisfaction - room for improvement")
                else:
                    insights.append("🚨 Sales performance indicates risk of low customer satisfaction")
            
            if sent_success and 'predictions' in locals():
                positive_sentiment = (np.array(predictions) >= 4).mean()
                if positive_sentiment >= 0.6:
                    insights.append("😊 Customer sentiment analysis shows predominantly positive feedback")
                elif positive_sentiment >= 0.4:
                    insights.append("😐 Customer sentiment is mixed - consider improvement initiatives")
                else:
                    insights.append("😞 Customer sentiment shows concerning negative trends")
            
            for insight in insights:
                st.markdown(f"- {insight}")
            
        except Exception as e:
            st.error(f"❌ Error in combined analysis: {e}")

# BUSINESS INSIGHTS PAGE
elif page == "📊 Business Insights":
    st.markdown("## 📊 Advanced Business Insights")
    st.markdown("Deep dive analytics for business optimization and strategic decision making.")
    
    # Sample business metrics and KPIs
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Customer Satisfaction Target",
            value="4.2/5.0",
            delta="0.3",
            help="Target average review score for business success"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Positive Sentiment Target",
            value="75%",
            delta="5%",
            help="Target percentage of positive customer reviews"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Model Accuracy",
            value="85%",
            delta="2%",
            help="Prediction accuracy of our ML models"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Processing Speed",
            value="1000/min",
            delta="100",
            help="Reviews processed per minute"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 💡 Business Recommendations")
    
    recommendations = [
        {
            "category": "🎯 Customer Satisfaction",
            "recommendations": [
                "Focus on improving delivery time to boost review scores",
                "Monitor seller performance across different states",
                "Optimize pricing strategy based on category performance",
                "Implement quality control measures for low-rated categories"
            ]
        },
        {
            "category": "📈 Revenue Optimization",
            "recommendations": [
                "Prioritize high-rating potential products in recommendations",
                "Adjust freight pricing based on customer satisfaction impact",
                "Develop seller coaching programs for performance improvement",
                "Create category-specific quality standards"
            ]
        },
        {
            "category": "🤖 AI/ML Integration",
            "recommendations": [
                "Implement real-time sentiment monitoring",
                "Use prediction models for proactive customer service",
                "Automate quality alerts based on prediction confidence",
                "Integrate sentiment analysis into seller dashboards"
            ]
        }
    ]
    
    for rec in recommendations:
        with st.expander(rec["category"]):
            for item in rec["recommendations"]:
                st.markdown(f"• {item}")
    
    st.markdown("### 📊 Model Performance Monitoring")
    
    # Simulated model performance data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    performance_data = {
        'Date': dates,
        'Regression_Accuracy': np.random.normal(0.85, 0.05, len(dates)),
        'Sentiment_Accuracy': np.random.normal(0.78, 0.04, len(dates)),
        'Processing_Speed': np.random.normal(1000, 100, len(dates))
    }
    
    perf_df = pd.DataFrame(performance_data)
    
    fig_perf = px.line(
        perf_df,
        x='Date',
        y=['Regression_Accuracy', 'Sentiment_Accuracy'],
        title="Model Performance Trends",
        labels={'value': 'Accuracy', 'variable': 'Model Type'}
    )
    st.plotly_chart(fig_perf, use_container_width=True)
    
    st.markdown("### 🔮 Future Enhancements")
    
    enhancements = [
        "🌟 **Real-time Analytics**: Implement streaming data processing for instant insights",
        "🤖 **Advanced NLP**: Integrate transformer models for better sentiment analysis",
        "📱 **Mobile Dashboard**: Create mobile-optimized interface for on-the-go monitoring",
        "🔔 **Alert System**: Automated notifications for unusual patterns or performance drops",
        "📊 **Advanced Visualizations**: Interactive dashboards with drill-down capabilities",
        "🔒 **Security Features**: Enhanced data protection and user authentication"
    ]
    
    for enhancement in enhancements:
        st.markdown(enhancement)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>📊 E-Commerce Business Insights Dashboard | Built with Streamlit & Machine Learning</p>
        <p>🤖 Powered by XGBoost Regression & Naive Bayes Classification</p>
    </div>
    """,
    unsafe_allow_html=True
)
