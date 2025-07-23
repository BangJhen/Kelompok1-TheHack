import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import joblib

st.set_page_config(page_title="E-Commerce Review Score Regression", layout="wide")
st.title("E-Commerce Review Score Regression Dashboard")

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

        # Load preprocessing objects and model
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

        # Replace encoding and scaling with preloaded objects
        for feature, encoder in label_encoders.items():
            if feature in X.columns:
                X[feature] = encoder.transform(X[feature].astype(str))

        X_scaled = scaler.transform(X)

        # Use the loaded model for predictions
        y_pred = model.predict(X_scaled)

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
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_names = X.columns
            imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
            st.bar_chart(imp_df.set_index("Feature"))
        elif hasattr(model, "coef_"):
            coef = model.coef_
            feat_names = X.columns
            coef_df = pd.DataFrame({"Feature": feat_names, "Coefficient": coef})
            st.bar_chart(coef_df.set_index("Feature"))
        else:
            st.write("Feature importances not available for this model.")
else:
    st.info("Please upload a CSV file to start.")
