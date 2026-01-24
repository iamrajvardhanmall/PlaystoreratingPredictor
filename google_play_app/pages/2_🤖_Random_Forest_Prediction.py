import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_and_preprocess_data, engineer_features, prepare_modeling_features,
                   train_random_forest_model, predict_rating, save_models, load_models)

st.set_page_config(page_title="Random Forest Prediction", page_icon="🤖", layout="wide")

st.title("🤖 Random Forest Rating Prediction")
st.markdown("### Predict app ratings using Random Forest Regressor with PCA")

@st.cache_resource
def load_or_train_model():
    """Load existing model or train a new one"""
    models = load_models()
    
    if models is None:
        with st.spinner("Training Random Forest model... This may take a minute."):
            df = load_and_preprocess_data()
            df, _, _ = engineer_features(df)
            
            X, y, feature_names = prepare_modeling_features(df)
            
            rf_model, pca, scaler, train_score, test_score = train_random_forest_model(X, y)
            
            from utils import train_kmeans_model
            kmeans, pca_km, scaler_km, _ = train_kmeans_model(X)
            
            save_models(rf_model, kmeans, pca, pca_km, scaler, scaler_km, feature_names)
            
            return rf_model, pca, scaler, feature_names, train_score, test_score, df
    else:
        rf_model = models['rf_model']
        pca = models['pca_rf']
        scaler = models['scaler_rf']
        feature_names = models['feature_names']
        
        df = load_and_preprocess_data()
        df, _, _ = engineer_features(df)
        X, y, _ = prepare_modeling_features(df)
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        scaler_temp = StandardScaler()
        X_scaled = scaler_temp.fit_transform(X)
        from sklearn.decomposition import PCA
        pca_temp = PCA(n_components=2)
        X_pca = pca_temp.fit_transform(X_scaled)
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
        
        train_score = rf_model.score(X_train, y_train)
        test_score = rf_model.score(X_test, y_test)
        
        return rf_model, pca, scaler, feature_names, train_score, test_score, df

try:
    rf_model, pca, scaler, feature_names, train_score, test_score, df = load_or_train_model()
    
    st.header("📈 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training R² Score", f"{train_score:.4f}")
    with col2:
        st.metric("Testing R² Score", f"{test_score:.4f}")
    with col3:
        st.metric("PCA Components", "2")
    with col4:
        st.metric("Model Type", "Random Forest")
    
    st.info("""
    **About the Model:**
    - The model uses **Principal Component Analysis (PCA)** to reduce 4 features to 2 components
    - **Random Forest Regressor** with 200 estimators is trained on PCA-transformed data
    - Features: Log(Reviews), Log(Installs), Size, and Price
    """)
    
    st.markdown("---")
    
    st.header("🎯 Make a Prediction")
    st.markdown("Enter the app characteristics below to predict its rating:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("App Metrics")
        
        reviews = st.number_input(
            "Number of Reviews",
            min_value=0,
            max_value=100000000,
            value=10000,
            step=1000,
            help="Total number of user reviews"
        )
        
        installs = st.selectbox(
            "Number of Installs",
            options=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 
                    1000000, 5000000, 10000000, 50000000, 100000000],
            index=6,
            help="Total number of app installs"
        )
        
    with col2:
        st.subheader("App Properties")
        
        size = st.slider(
            "App Size (MB)",
            min_value=0.1,
            max_value=100.0,
            value=20.0,
            step=0.1,
            help="Size of the app in megabytes"
        )
        
        price = st.number_input(
            "Price ($)",
            min_value=0.0,
            max_value=400.0,
            value=0.0,
            step=0.99,
            help="App price in dollars (0 for free apps)"
        )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔮 Predict Rating", type="primary", use_container_width=True)
    
    if predict_button:
        input_data = [reviews, installs, size, price]
        
        predicted_rating = predict_rating(input_data, rf_model, pca, scaler, feature_names)
        
        predicted_rating = max(1.0, min(5.0, predicted_rating))
        
        st.markdown("---")
        st.header("🎊 Prediction Result")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>⭐ {predicted_rating:.2f} / 5.0</h1>", 
                       unsafe_allow_html=True)
            
            if predicted_rating >= 4.5:
                category = "Excellent"
                color = "green"
                emoji = "🏆"
            elif predicted_rating >= 4.0:
                category = "Very Good"
                color = "blue"
                emoji = "👍"
            elif predicted_rating >= 3.5:
                category = "Good"
                color = "orange"
                emoji = "👌"
            elif predicted_rating >= 3.0:
                category = "Fair"
                color = "orange"
                emoji = "😐"
            else:
                category = "Poor"
                color = "red"
                emoji = "👎"
            
            st.markdown(f"<h3 style='text-align: center;'>{emoji} {category}</h3>", 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("📋 Input Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Reviews", f"{reviews:,}")
        with col2:
            st.metric("Installs", f"{installs:,}")
        with col3:
            st.metric("Size", f"{size} MB")
        with col4:
            st.metric("Price", f"${price:.2f}")
        
        st.markdown("---")
        st.subheader("📊 Comparison with Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Your App vs Average:**")
            avg_rating = df['Rating'].mean()
            avg_reviews = df['Reviews'].mean()
            avg_installs = df['Installs'].mean()
            avg_size = df['Size'].mean()
            
            comparison_data = {
                'Metric': ['Rating', 'Reviews', 'Installs', 'Size (MB)'],
                'Your App': [f"{predicted_rating:.2f}", f"{reviews:,}", f"{installs:,}", f"{size:.1f}"],
                'Dataset Avg': [f"{avg_rating:.2f}", f"{avg_reviews:,.0f}", f"{avg_installs:,.0f}", f"{avg_size:.1f}"]
            }
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.write("**Percentile Ranking:**")
            
            reviews_percentile = (df['Reviews'] <= reviews).sum() / len(df) * 100
            installs_percentile = (df['Installs'] <= installs).sum() / len(df) * 100
            size_percentile = (df['Size'] <= size).sum() / len(df) * 100
            
            st.progress(reviews_percentile / 100, text=f"Reviews: {reviews_percentile:.1f}th percentile")
            st.progress(installs_percentile / 100, text=f"Installs: {installs_percentile:.1f}th percentile")
            st.progress(size_percentile / 100, text=f"Size: {size_percentile:.1f}th percentile")
    
    st.markdown("---")
    st.header("📊 Feature Importance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        explained_variance = pca.explained_variance_ratio_
        
        components = ['PC1', 'PC2']
        ax.bar(components, explained_variance, color=['#1f77b4', '#ff7f0e'], edgecolor='black')
        ax.set_title('PCA Components - Explained Variance', fontsize=14, fontweight='bold')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_xlabel('Principal Components')
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(explained_variance):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("### PCA Insights")
        st.write(f"""
        **Explained Variance:**
        - PC1: {explained_variance[0]:.2%}
        - PC2: {explained_variance[1]:.2%}
        - Total: {sum(explained_variance):.2%}
        
        The first two principal components capture {sum(explained_variance):.1%} of the total variance in the data.
        """)
    
    st.markdown("---")
    st.header("ℹ️ Model Information")
    
    st.write("""
    ### How it works:
    
    1. **Feature Engineering**: The input features (Reviews, Installs, Size, Price) are log-transformed where appropriate
    2. **Standardization**: Features are scaled to have zero mean and unit variance
    3. **PCA Transformation**: The 4-dimensional feature space is reduced to 2 principal components
    4. **Prediction**: Random Forest Regressor predicts the rating based on the 2 PCA components
    
    ### Model Details:
    - **Algorithm**: Random Forest Regressor
    - **Number of Trees**: 200
    - **Dimensionality Reduction**: PCA (4D → 2D)
    - **Training Data**: Google Play Store dataset
    
    ### Limitations:
    - Predictions are based on historical data patterns
    - Actual ratings may vary due to factors not captured in the model
    - Model performs best for apps with characteristics similar to the training data
    """)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please ensure the 'googleplaystore.csv' file is in the correct directory and try again.")
