import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Sales Parameter Classification",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Sales Parameter Classification")
st.markdown("### Prediksi Klasifikasi Average Review Score berdasarkan Parameter Penjualan")
st.markdown("---")

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

# Sidebar for input parameters
st.sidebar.header("📝 Input Parameter Penjualan")

# Load sample data for reference values
sample_df = load_sample_data()

# Input fields based on the EDA notebook features
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Product Information")
    
    # Product category
    if sample_df is not None:
        categories = sample_df['product_category_name_english'].unique()
        categories = [cat for cat in categories if pd.notna(cat)]
    else:
        categories = ['housewares', 'perfumery', 'auto', 'pet_shop', 'stationery', 'health_beauty', 'sports_leisure']
    
    product_category = st.selectbox(
        "Kategori Produk",
        options=categories,
        help="Pilih kategori produk"
    )
    
    price = st.number_input(
        "Harga Produk (R$)",
        min_value=0.0,
        max_value=10000.0,
        value=100.0,
        step=1.0,
        help="Masukkan harga produk dalam Real Brasil"
    )
    
    freight_value = st.number_input(
        "Biaya Pengiriman (R$)",
        min_value=0.0,
        max_value=500.0,
        value=15.0,
        step=0.1,
        help="Masukkan biaya pengiriman"
    )
    
    product_description_length = st.number_input(
        "Panjang Deskripsi Produk",
        min_value=0,
        max_value=5000,
        value=500,
        step=10,
        help="Jumlah karakter dalam deskripsi produk"
    )
    
    product_photos_qty = st.number_input(
        "Jumlah Foto Produk",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help="Jumlah foto produk yang ditampilkan"
    )

with col2:
    st.subheader("🚚 Delivery & Seller Information")
    
    delivery_delay_hours = st.number_input(
        "Keterlambatan Pengiriman (jam)",
        min_value=-1000.0,
        max_value=1000.0,
        value=0.0,
        step=1.0,
        help="Keterlambatan pengiriman dalam jam (negatif = lebih cepat)"
    )
    
    time_to_ship_hours = st.number_input(
        "Waktu Pengiriman (jam)",
        min_value=0.0,
        max_value=500.0,
        value=48.0,
        step=1.0,
        help="Waktu dari pemesanan hingga pengiriman"
    )
    
    seller_grade_options = ['Poor', 'Mediocre', 'Good', 'Excellent']
    seller_grade = st.selectbox(
        "Grade Penjual",
        options=seller_grade_options,
        index=1,
        help="Rating kualitas penjual"
    )
    
    purchase_count = st.number_input(
        "Jumlah Pembelian Sebelumnya",
        min_value=1,
        max_value=2000,
        value=100,
        step=1,
        help="Jumlah pembelian produk ini sebelumnya"
    )
    
    # State selection
    states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'GO', 'PE', 'CE']
    seller_state = st.selectbox(
        "State Penjual",
        options=states,
        help="State tempat penjual berada"
    )
    
    customer_state = st.selectbox(
        "State Pembeli",
        options=states,
        help="State tempat pembeli berada"
    )

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

# Create range categorization functions (matching exact EDA notebook logic)
def categorize_delivery_delay(delay_hours):
    """Categorize delivery delay into ranges - exact match with EDA notebook"""
    # bins = [-float('inf'), -24, 0, 24, 168, float('inf')]
    # labels = ['Early', 'On-Time', 'Slight Delay', 'Moderate Delay', 'Significant Delay']
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
    # price_bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    # price_labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '>1000']
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
    # freight_bins = [0, 10, 20, 30, 50, float('inf')]
    # freight_labels = ['0-10', '11-20', '21-30', '31-50', '>50']
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
    grade_mapping = {'Poor': 0, 'Mediocre': 1, 'Good': 2, 'Excellent': 3}
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

# Prediction button and results
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔮 Prediksi Klasifikasi Review Score", type="primary", use_container_width=True):
        # Prepare features dictionary
        features = {
            'product_category_name_english': product_category,
            'price': price,
            'freight_value': freight_value,
            'product_description_length': product_description_length,
            'product_photos_qty': product_photos_qty,
            'delivery_delay_hours': delivery_delay_hours,
            'time_to_ship_hours': time_to_ship_hours,
            'seller_grade': seller_grade,
            'purchase_count': purchase_count,
            'seller_state': seller_state,
            'customer_state': customer_state
        }
        
        # Make prediction using trained model
        predicted_score = predict_review_score_with_model(features)
        predicted_class = classify_review_score(predicted_score)
        
        # Display results
        st.success("✅ Prediksi Berhasil!")
        
        # Create metrics display
        col_metric1, col_metric2 = st.columns(2)
        
        with col_metric1:
            st.metric(
                label="Predicted Average Review Score",
                value=f"{predicted_score:.2f}",
                delta=f"{predicted_score - 4.0:.2f} dari rata-rata"
            )
        
        with col_metric2:
            st.metric(
                label="Klasifikasi",
                value=predicted_class
            )
        
        # Create visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = predicted_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Review Score Prediction"},
            delta = {'reference': 4.0},
            gauge = {
                'axis': {'range': [None, 5]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 2], 'color': "lightgray"},
                    {'range': [2, 3], 'color': "red"},
                    {'range': [3, 3.5], 'color': "orange"},
                    {'range': [3.5, 4], 'color': "yellow"},
                    {'range': [4, 4.5], 'color': "lightgreen"},
                    {'range': [4.5, 5], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 4.0
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance explanation
        st.subheader("📈 Analisis Faktor Prediksi")
        
        importance_data = {
            'Faktor': [
                'Keterlambatan Pengiriman',
                'Grade Penjual', 
                'Harga Produk',
                'Waktu Pengiriman',
                'Biaya Freight',
                'Jumlah Foto Produk',
                'Same State'
            ],
            'Pengaruh': [
                'Tinggi' if abs(delivery_delay_hours) > 50 else 'Sedang',
                'Tinggi' if seller_grade in ['Good', 'Excellent'] else 'Rendah',
                'Sedang' if 50 <= price <= 200 else 'Rendah',
                'Sedang' if time_to_ship_hours < 48 else 'Rendah',
                'Sedang' if freight_value < 20 else 'Rendah',
                'Rendah' if product_photos_qty >= 3 else 'Sangat Rendah',
                'Rendah' if seller_state == customer_state else 'Tidak Ada'
            ]
        }
        
        importance_df = pd.DataFrame(importance_data)
        st.dataframe(importance_df, use_container_width=True)

# Information section
st.markdown("---")
st.subheader("ℹ️ Informasi Aplikasi")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Cara Menggunakan:**
    1. Masukkan parameter penjualan di sidebar
    2. Klik tombol 'Prediksi Klasifikasi Review Score'
    3. Lihat hasil prediksi dan analisis faktor
    """)

with col2:
    st.warning("""
    **Catatan:**
    - Model ini menggunakan logika berbasis aturan sederhana
    - Untuk akurasi yang lebih baik, gunakan model ML yang telah dilatih
    - Hasil prediksi bersifat estimasi berdasarkan pola umum e-commerce
    """)

# Sample data display
if sample_df is not None:
    st.markdown("---")
    st.subheader("📋 Sample Data Reference")
    
    if st.checkbox("Tampilkan Sample Data"):
        st.dataframe(sample_df.head(10), use_container_width=True)
        
        # Basic statistics
        st.subheader("📊 Statistik Dasar Sample Data")
        
        numeric_cols = ['price', 'freight_value', 'delivery_delay_hours', 'time_to_ship_hours', 'avg_review_score']
        stats_df = sample_df[numeric_cols].describe()
        st.dataframe(stats_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Sales Parameter Classification App | Built with Streamlit</p>
    <p>Based on E-commerce Data Analysis Workflow</p>
</div>
""", unsafe_allow_html=True)
