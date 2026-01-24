import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pickle
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_and_preprocess_data(filepath=None):
    """Load and preprocess the Google Play Store dataset"""
    if filepath is None:
        filepath = os.path.join(SCRIPT_DIR, 'googleplaystore.csv')
    df = pd.read_csv(filepath)
    
    df.columns = df.columns.str.strip()
    
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    mean_rating = df['Rating'].mean()
    df['Rating'].fillna(mean_rating, inplace=True)
    
    def clean_size(size):
        if isinstance(size, str):
            if 'M' in size:
                return float(size.replace('M', '').strip())
            elif 'k' in size.lower():
                return float(size.lower().replace('k', '').strip()) / 1024
            elif size == 'Varies with device':
                return np.nan
        return np.nan
    
    df['Size'] = df['Size'].apply(clean_size)
    df['Size'].fillna(df['Size'].median(), inplace=True)
    
    def clean_installs(installs):
        if isinstance(installs, str):
            installs = installs.replace('+', '').replace(',', '')
            if installs == 'Free':
                return 0
            try:
                return int(installs)
            except:
                return 0
        return 0
    
    df['Installs'] = df['Installs'].apply(clean_installs)
    
    def clean_price(price):
        if isinstance(price, str):
            price = price.replace('$', '').strip()
            if price == 'Everyone':
                return 0
            try:
                return float(price)
            except:
                return 0
        return 0
    
    df['Price'] = df['Price'].apply(clean_price)
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    df['Reviews'].fillna(0, inplace=True)
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
    df['Last Updated_Days'] = (pd.Timestamp.now() - df['Last Updated']).dt.days
    df['Last Updated_Days'].fillna(df['Last Updated_Days'].median(), inplace=True)
    df['Content Rating'] = df['Content Rating'].fillna('Everyone')
    
    df = df.dropna()
    
    return df

def engineer_features(df):
    """Create engineered features for modeling"""
    df = df.copy()
    
    df['Log_Reviews'] = np.log1p(df['Reviews'])
    df['Log_Installs'] = np.log1p(df['Installs'])
    
    df['Price_Category'] = np.where(df['Price'] > 0, 'Paid', 'Free')
    
    max_size = df['Size'].max()
    if max_size <= 100:
        max_size = 101
    
    df['Size_Category'] = pd.cut(
        df['Size'],
        bins=[0, 10, 50, 100, max_size],
        labels=['Small', 'Medium', 'Large', 'Very Large'],
        include_lowest=True
    )
    
    le_type = LabelEncoder()
    df['Type_Encoded'] = le_type.fit_transform(df['Type'].astype(str))
    
    le_size = LabelEncoder()
    df['Size_Category_Encoded'] = le_size.fit_transform(df['Size_Category'].astype(str))
    
    return df, le_type, le_size

def prepare_modeling_features(df):
    """Prepare features for PCA and modeling"""
    features = [
        'Log_Reviews',
        'Log_Installs',
        'Size',
        'Price'
    ]
    
    X = df[features]
    y = df['Rating']
    
    return X, y, features

def train_random_forest_model(X, y):
    """Train Random Forest model with PCA"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )
    
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    train_score = rf_model.score(X_train, y_train)
    test_score = rf_model.score(X_test, y_test)
    
    return rf_model, pca, scaler, train_score, test_score

def train_kmeans_model(X):
    """Train K-Means clustering model with PCA"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    
    return kmeans, pca, scaler, cluster_labels

def predict_rating(input_data, rf_model, pca, scaler, feature_names):
    """Predict rating using trained Random Forest model"""
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    input_df['Log_Reviews'] = np.log1p(input_df['Log_Reviews'])
    input_df['Log_Installs'] = np.log1p(input_df['Log_Installs'])
    
    input_scaled = scaler.transform(input_df)
    input_pca = pca.transform(input_scaled)
    prediction = rf_model.predict(input_pca)
    
    return prediction[0]

def predict_cluster(input_data, kmeans, pca, scaler, feature_names):
    """Predict cluster using trained K-Means model"""
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    input_df['Log_Reviews'] = np.log1p(input_df['Log_Reviews'])
    input_df['Log_Installs'] = np.log1p(input_df['Log_Installs'])
    
    input_scaled = scaler.transform(input_df)
    input_pca = pca.transform(input_scaled)
    cluster = kmeans.predict(input_pca)
    
    return cluster[0]

def get_cluster_characteristics(df, cluster_labels, features):
    """Get characteristics of each cluster"""
    df_copy = df.copy()
    df_copy['Cluster'] = cluster_labels
    
    cluster_stats = df_copy.groupby('Cluster')[features].mean()
    cluster_counts = df_copy['Cluster'].value_counts().sort_index()
    
    return cluster_stats, cluster_counts

def save_models(rf_model, kmeans, pca_rf, pca_km, scaler_rf, scaler_km, feature_names):
    """Save trained models to disk"""
    models = {
        'rf_model': rf_model,
        'kmeans': kmeans,
        'pca_rf': pca_rf,
        'pca_km': pca_km,
        'scaler_rf': scaler_rf,
        'scaler_km': scaler_km,
        'feature_names': feature_names
    }
    
    models_path = os.path.join(SCRIPT_DIR, 'models.pkl')
    with open(models_path, 'wb') as f:
        pickle.dump(models, f)
    
    return True

def load_models():
    """Load trained models from disk"""
    models_path = os.path.join(SCRIPT_DIR, 'models.pkl')
    if os.path.exists(models_path):
        with open(models_path, 'rb') as f:
            models = pickle.load(f)
        return models
    return None
