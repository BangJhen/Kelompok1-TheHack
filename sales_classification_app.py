import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Sales Parameter Classification",
    layout="wide"
)

# Enhanced text cleaning function
stop_words = set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with'])

stemmer = PorterStemmer()

def advanced_clean_text(text):
    """
    Enhanced text cleaning function
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep important punctuation for sentiment
    text = re.sub(r'[^a-zA-Z\s!?.]', ' ', text)
    
    # Handle negations (important for sentiment)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    
    # Handle repeated characters (e.g., "gooood" -> "good")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Remove very short words (less than 2 characters) and stopwords
    cleaned_tokens = []
    for word in tokens:
        if len(word) > 1 and word not in stop_words:
            # Apply stemming
            try:
                stemmed_word = stemmer.stem(word)
                cleaned_tokens.append(stemmed_word)
            except:
                cleaned_tokens.append(word)
    
    return ' '.join(cleaned_tokens)

# Load sample data for reference
@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv("sample list sales.csv")
        return df
    except:
        return None

# Create classification categories for avg_review_score
def classify_review_score(score):
    if score < 3.0:
        return "Poor (< 3.0)"
    elif score < 3.5:
        return "Below Average (3.0-3.5)"
    elif score < 4.0:
        return "Average (3.5-4.0)"
    elif score < 4.5:
        return "Good (4.0-4.5)"
    else:
        return "Excellent (≥ 4.5)"

# Load trained models and preprocessing components
@st.cache_resource
def load_models():
    """Load the trained Random Forest model and preprocessing components"""
    try:
        # Load the trained Random Forest model
        model = joblib.load('model/best_model_RandomForest.pkl')
        
        # Load preprocessing components
        scaler = joblib.load('model/minmax_scaler.pkl')
        
        # Load label encoders
        label_encoders = {}
        encoder_files = [
            'seller_state', 'customer_state', 'seller_grade', 
            'product_category_name_english', 'delivery_delay_range', 
            'price_range', 'freight_range', 'same_state'
        ]
        
        for encoder_name in encoder_files:
            try:
                encoder_path = f'model/label_encoder_{encoder_name}.pkl'
                if os.path.exists(encoder_path):
                    label_encoders[encoder_name] = joblib.load(encoder_path)
            except Exception as e:
                st.warning(f"Could not load encoder for {encoder_name}: {e}")
        
        return model, scaler, label_encoders
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Load sentiment analysis models
@st.cache_resource
def load_sentiment_models():
    """Load the sentiment analysis model and preprocessing components"""
    try:
        # Load sentiment analysis components
        sentiment_model = joblib.load('model/sentiment_model.pkl')
        tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
        # Use the locally implemented advanced_clean_text function instead of loading from pickle
        text_cleaner = advanced_clean_text
        
        return sentiment_model, tfidf_vectorizer, text_cleaner
    except Exception as e:
        st.error(f"Error loading sentiment models: {e}")
        return None, None, None

# Enhanced function to predict sentiment of new reviews
def predict_sentiment_enhanced(review_text, model, vectorizer, cleaner):
    """
    Predict sentiment rating for a new review using enhanced preprocessing
    """
    try:
        # Clean the text
        cleaned_text = cleaner(review_text)
        
        # Check if text is empty after cleaning
        if not cleaned_text or len(cleaned_text.strip()) == 0:
            return 3, [0.0, 0.0, 1.0, 0.0, 0.0]  # Return neutral if no text
        
        # Transform to TF-IDF
        text_tfidf = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_tfidf)[0]
        
        # Get probability if available
        try:
            probability = model.predict_proba(text_tfidf)[0]
        except:
            # If predict_proba is not available, create dummy probabilities
            probability = [0.0] * 5
            probability[prediction - 1] = 1.0
        
        return prediction, probability
    except Exception as e:
        st.error(f"Error in sentiment prediction: {e}")
        return 3, [0.0, 0.0, 1.0, 0.0, 0.0]

# Create range categorization functions (matching exact EDA notebook logic)
def categorize_delivery_delay(delay_hours):
    """Categorize delivery delay into ranges - exact match with EDA notebook"""
    if delay_hours < -24:
        return "Early"
    elif delay_hours < 0:
        return "On-Time"
    elif delay_hours < 24:
        return "Slight Delay"
    elif delay_hours < 168:
        return "Moderate Delay"
    else:
        return "Significant Delay"

def categorize_price(price):
    """Categorize price into ranges - exact match with EDA notebook"""
    if price < 50:
        return "0-50"
    elif price < 100:
        return "51-100"
    elif price < 200:
        return "101-200"
    elif price < 500:
        return "201-500"
    elif price < 1000:
        return "501-1000"
    else:
        return ">1000"

def categorize_freight(freight):
    """Categorize freight value into ranges - exact match with EDA notebook"""
    if freight < 10:
        return "0-10"
    elif freight < 20:
        return "11-20"
    elif freight < 30:
        return "21-30"
    elif freight < 50:
        return "31-50"
    else:
        return ">50"

# Create prediction function using trained model
def predict_review_score_with_model(features):
    """Use the trained Random Forest model to predict average review score"""
    model, scaler, label_encoders = load_models()
    
    if model is None or scaler is None or not label_encoders:
        st.error("Could not load trained models. Using fallback prediction.")
        return predict_fallback(features)
    
    try:
        # Prepare the feature vector in the same order as training
        feature_order = [
            'delivery_delay_hours', 'seller_state', 'customer_state', 'price',
            'freight_value', 'seller_grade', 'product_category_name_english',
            'product_description_lenght', 'product_photos_qty', 'time_to_ship_hours',
            'purchase_count', 'delivery_delay_range', 'price_range', 'freight_range', 'same_state'
        ]
        
        # Create feature vector
        feature_vector = []
        
        # Add continuous features
        feature_vector.append(features['delivery_delay_hours'])
        feature_vector.append(0)  # seller_state (will be encoded)
        feature_vector.append(0)  # customer_state (will be encoded)
        feature_vector.append(features['price'])
        feature_vector.append(features['freight_value'])
        feature_vector.append(0)  # seller_grade (will be encoded)
        feature_vector.append(0)  # product_category (will be encoded)
        feature_vector.append(features['product_description_length'])
        feature_vector.append(features['product_photos_qty'])
        feature_vector.append(features['time_to_ship_hours'])
        feature_vector.append(features['purchase_count'])
        feature_vector.append(0)  # delivery_delay_range (will be encoded)
        feature_vector.append(0)  # price_range (will be encoded)
        feature_vector.append(0)  # freight_range (will be encoded)
        feature_vector.append(1 if features['seller_state'] == features['customer_state'] else 0)  # same_state
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame([feature_vector], columns=feature_order)
        
        # Apply label encoding for categorical features
        categorical_mappings = {
            'seller_state': features['seller_state'],
            'customer_state': features['customer_state'],
            'seller_grade': features['seller_grade'],
            'product_category_name_english': features['product_category_name_english'],
            'delivery_delay_range': categorize_delivery_delay(features['delivery_delay_hours']),
            'price_range': categorize_price(features['price']),
            'freight_range': categorize_freight(features['freight_value'])
        }
        
        for feature, value in categorical_mappings.items():
            if feature in label_encoders:
                try:
                    # Check if the value exists in the encoder's classes
                    if value in label_encoders[feature].classes_:
                        df[feature] = label_encoders[feature].transform([value])[0]
                    else:
                        # Use the most common class (index 0) as fallback
                        df[feature] = 0
                        st.warning(f"Unknown value '{value}' for {feature}. Using fallback.")
                except Exception as e:
                    df[feature] = 0
                    st.warning(f"Error encoding {feature}: {e}")
        
        # Apply scaling
        df_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(df_scaled)[0]
        
        # Ensure prediction is within valid range
        prediction = max(1.0, min(5.0, prediction))
        
        return prediction
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return predict_fallback(features)

# Fallback prediction function
def predict_fallback(features):
    """Fallback rule-based prediction if model loading fails"""
    grade_mapping = {'Bad': 0, 'Mediocre': 1, 'Super': 2}
    seller_grade_num = grade_mapping.get(features['seller_grade'], 1)
    
    base_score = 3.5
    
    if 50 <= features['price'] <= 200:
        base_score += 0.3
    elif features['price'] > 500:
        base_score -= 0.2
    
    if features['delivery_delay_hours'] < -50:
        base_score += 0.4
    elif features['delivery_delay_hours'] > 100:
        base_score -= 0.6
    
    base_score += (seller_grade_num - 1) * 0.2
    
    if features['time_to_ship_hours'] < 24:
        base_score += 0.2
    elif features['time_to_ship_hours'] > 72:
        base_score -= 0.3
    
    if features['product_photos_qty'] >= 3:
        base_score += 0.1
    
    if features['seller_state'] == features['customer_state']:
        base_score += 0.1
    
    if features['freight_value'] < 10:
        base_score += 0.1
    elif features['freight_value'] > 30:
        base_score -= 0.2
    
    return max(1.0, min(5.0, base_score))

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-container {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-container {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .error-container {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .section-divider {
        border: 0;
        height: 2px;
        background: linear-gradient(to right, transparent, #1f77b4, transparent);
        margin: 2rem 0;
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-section {
        background-color: #fff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced main app title
st.markdown('<h1 class="main-header">Sales Parameter Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Prediksi Rating Review berdasarkan Parameter Penjualan dan Analisis Sentimen</p>', unsafe_allow_html=True)

# Add description section
with st.container():
    st.markdown("""
    <div class="input-section">
        <h4>Tentang Aplikasi Ini</h4>
        <p>Aplikasi ini menggunakan machine learning untuk memprediksi rating review produk berdasarkan</p>
        <ul>
            <li><strong>Parameter Penjualan:</strong> Harga, kategori produk, biaya kirim, informasi penjual, dll.</li>
            <li><strong>Analisis Sentimen:</strong> Analisis teks review untuk memprediksi rating berdasarkan sentimen</li>
        </ul>
        <p>Kedua metode prediksi akan dibandingkan untuk memberikan insight yang komprehensif.</p>
    </div>
    """, unsafe_allow_html=True)

# Load sample data
sample_df = load_sample_data()

# Create input layout with better organization
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Input sections with enhanced styling
st.markdown("""
<div class="input-section">
    <h2>Input Parameter untuk Prediksi</h2>
</div>
""", unsafe_allow_html=True)

# First row: Sales parameters
st.markdown("""
<div class="input-section">
    <h3>Parameter Penjualan</h3>
    <p>Masukkan detail produk dan informasi penjualan untuk prediksi rating:</p>
</div>
""", unsafe_allow_html=True)

input_col1, input_col2 = st.columns(2)

# Column 1: Basic Product Information
with input_col1:
    st.markdown("#### Informasi Produk")
    product_category = st.selectbox(
        "Kategori Produk:",
        options=[
            'bed_bath_table', 'sports_leisure', 'health_beauty', 'computers_accessories',
            'furniture_decor', 'watches_gifts', 'telephony', 'auto', 'toys', 'garden_tools',
            'baby', 'stationery', 'perfumery', 'construction_tools_safety', 'cool_stuff'
        ],
        index=0
    )
    
    price = st.number_input("Harga Produk (BRL):", min_value=0.0, value=100.0, step=10.0)
    freight_value = st.number_input("Biaya Pengiriman (BRL):", min_value=0.0, value=20.0, step=5.0)
    product_description_length = st.number_input("Panjang Deskripsi Produk:", min_value=0, value=100, step=50)
    product_photos_qty = st.number_input("Jumlah Foto Produk:", min_value=0, value=3, step=1)

# Column 2: Seller and Delivery Information
with input_col2:
    st.markdown("#### Informasi Penjual & Pengiriman")
    seller_state = st.selectbox(
        "Wilayah Penjual:",
        options=[
            'SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'GO', 'PE',
            'ES', 'PB', 'CE', 'RN', 'PI', 'AL', 'SE', 'MT', 'MS', 'RO',
            'AM', 'PA', 'MA', 'TO', 'AC', 'AP', 'RR'
        ],
        index=0
    )
    
    customer_state = st.selectbox(
        "Wilayah Pembeli:",
        options=[
            'SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'GO', 'PE',
            'ES', 'PB', 'CE', 'RN', 'PI', 'AL', 'SE', 'MT', 'MS', 'RO',
            'AM', 'PA', 'MA', 'TO', 'AC', 'AP', 'RR'
        ],
        index=0
    )
    
    seller_grade = st.selectbox(
        "Grade Penjual:",
        options=['Bad', 'Mediocre', 'Super'],
        index=2
    )
    
    delivery_delay_hours = st.number_input("Delay Pengiriman (jam):", value=0.0, step=1.0)
    time_to_ship_hours = st.number_input("Waktu Pengiriman (jam):", min_value=0.0, value=24.0, step=1.0)
    purchase_count = st.number_input("Jumlah Pembelian:", min_value=0, value=1, step=1)

# Second row: Review text analysis
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

st.markdown("""
<div class="input-section">
    <h3>Analisis Sentimen Review</h3>
    <p>Masukkan teks review pelanggan untuk analisis sentimen dan prediksi rating:</p>
    <p><em>Catatan: Untuk hasil analisis sentimen yang optimal, gunakan teks review dalam bahasa Inggris.</em></p>
</div>
""", unsafe_allow_html=True)

# Create a dedicated section for review input
review_text = st.text_area(
    "Teks Review Pelanggan:",
    height=150,
    placeholder="Contoh: 'This product is amazing! Great quality and fast delivery. Very satisfied with my purchase!'",
    help="Review ini akan dianalisis untuk memprediksi rating berdasarkan sentimen. Untuk hasil optimal, gunakan teks review dalam bahasa Inggris."
)

# Prediction button in a separate section with enhanced styling
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

st.markdown("""
<div class="input-section" style="text-align: center;">
    <h3>Mulai Prediksi</h3>
    <p>Klik tombol di bawah untuk memulai analisis dan mendapatkan prediksi rating review</p>
</div>
""", unsafe_allow_html=True)

prediction_col1, prediction_col2, prediction_col3 = st.columns([1, 2, 1])

with prediction_col2:
    predict_button = st.button(
        "Prediksi Review Score", 
        type="primary", 
        use_container_width=True,
        help="Klik untuk melakukan prediksi berdasarkan parameter penjualan dan analisis sentimen"
    )

if predict_button:
    # Prepare features dictionary
    features = {
        'product_category_name_english': product_category,
        'price': price,
        'freight_value': freight_value,
        'product_description_length': product_description_length,
        'product_photos_qty': product_photos_qty,
        'seller_state': seller_state,
        'customer_state': customer_state,
        'seller_grade': seller_grade,
        'delivery_delay_hours': delivery_delay_hours,
        'time_to_ship_hours': time_to_ship_hours,
        'purchase_count': purchase_count
    }
    
    # Make prediction using trained model
    predicted_score = predict_review_score_with_model(features)
    predicted_class = classify_review_score(predicted_score)
    
    # Create results section with enhanced styling
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-section">
        <h2>Hasil Prediksi</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Success message with custom styling
    st.markdown("""
    <div class="success-container">
        <h4>Prediksi berhasil dilakukan!</h4>
        <p>Berikut adalah hasil analisis berdasarkan parameter yang Anda masukkan:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Results layout
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.markdown("#### Prediksi dari Parameter Penjualan")
        # Gauge for sales parameter prediction
        fig_gauge1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Rating Berdasarkan Parameter Penjualan", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [1, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [1, 2], 'color': '#ffcccc'},
                    {'range': [2, 3], 'color': '#ffd9b3'},
                    {'range': [3, 4], 'color': '#ffffcc'},
                    {'range': [4, 5], 'color': '#ccffcc'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_score}}))
        
        fig_gauge1.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge1, use_container_width=True)
        
        # Sales parameter metrics
        st.metric(
            label="Rating Prediksi",
            value=f"{predicted_score:.2f}",
            delta=f"Kategori: {predicted_class}"
        )
    
    # Sentiment analysis prediction (only if review text is provided)
    sentiment_rating = None
    
    if review_text.strip():
        with result_col2:
            st.markdown("#### Prediksi dari Analisis Sentimen")
            
            # Load sentiment models
            sentiment_model, tfidf_vectorizer, text_cleaner = load_sentiment_models()
            
            if sentiment_model is not None and tfidf_vectorizer is not None and text_cleaner is not None:
                # Predict sentiment rating
                sentiment_rating, sentiment_probabilities = predict_sentiment_enhanced(
                    review_text, sentiment_model, tfidf_vectorizer, text_cleaner
                )
                
                # Gauge for sentiment analysis prediction
                fig_gauge2 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sentiment_rating,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Rating Berdasarkan Sentimen Review", 'font': {'size': 14}},
                    gauge={
                        'axis': {'range': [1, 5], 'tickwidth': 1, 'tickcolor': "darkgreen"},
                        'bar': {'color': "darkgreen"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [1, 2], 'color': '#ffcccc'},
                            {'range': [2, 3], 'color': '#ffd9b3'},
                            {'range': [3, 4], 'color': '#ffffcc'},
                            {'range': [4, 5], 'color': '#ccffcc'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': sentiment_rating}}))
                
                fig_gauge2.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge2, use_container_width=True)
                
                # Sentiment metrics
                st.metric(
                    label="Rating Sentimen",
                    value=f"{sentiment_rating:.2f}",
                    delta=f"Kategori: {classify_review_score(sentiment_rating)}"
                )
            else:
                st.error("Model sentimen tidak dapat dimuat")
    else:
        with result_col2:
            st.markdown("#### Prediksi dari Analisis Sentimen")
            st.info("Masukkan teks review untuk mendapatkan prediksi sentimen")
    
    # Comparison section (only if both predictions are available)
    if review_text.strip() and sentiment_rating is not None:
        st.markdown("---")
        st.markdown("### Perbandingan Hasil Prediksi")
        
        difference = abs(predicted_score - sentiment_rating)
        
        # Create comparison visualization
        comparison_data = {
            'Metode': ['Parameter Penjualan', 'Analisis Sentimen'],
            'Rating': [predicted_score, sentiment_rating],
            'Kategori': [predicted_class, classify_review_score(sentiment_rating)]
        }
        
        comparison_col1, comparison_col2 = st.columns([2, 1])
        
        with comparison_col1:
            # Bar chart comparison
            fig_comparison = go.Figure(data=[
                go.Bar(name='Rating Prediksi', x=comparison_data['Metode'], y=comparison_data['Rating'],
                      marker_color=['#1f77b4', '#ff7f0e'])
            ])
            fig_comparison.update_layout(
                title="Perbandingan Rating Prediksi",
                yaxis_title="Rating (1-5)",
                xaxis_title="Metode Prediksi",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with comparison_col2:
            st.markdown("#### Analisis Konsistensi")
            
            # Display comparison result with better styling
            if difference <= 0.5:
                st.success("Konsistensi Tinggi")
                st.markdown(f"""
                - **Selisih**: {difference:.2f}
                - **Status**: Sangat Konsisten
                - **Interpretasi**: Prediksi dari parameter penjualan dan sentimen review sangat selaras
                """)
            elif difference <= 1.0:
                st.warning("Konsistensi Sedang")
                st.markdown(f"""
                - **Selisih**: {difference:.2f}
                - **Status**: Cukup Konsisten
                - **Interpretasi**: Ada perbedaan kecil antara prediksi parameter dan sentimen
                """)
            else:
                st.error("Konsistensi Rendah")
                st.markdown(f"""
                - **Selisih**: {difference:.2f}
                - **Status**: Tidak Konsisten
                - **Interpretasi**: Terdapat perbedaan signifikan antara prediksi parameter dan sentimen
                """)
