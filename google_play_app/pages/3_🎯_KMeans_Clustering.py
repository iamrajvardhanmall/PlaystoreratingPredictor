import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_and_preprocess_data, engineer_features, prepare_modeling_features,
                   train_kmeans_model, predict_cluster, get_cluster_characteristics, load_models)

st.set_page_config(page_title="K-Means Clustering", page_icon="🎯", layout="wide")

st.title("🎯 K-Means Clustering Prediction")
st.markdown("### Discover which cluster your app belongs to using K-Means with PCA")

@st.cache_resource
def load_clustering_model():
    """Load clustering model and data"""
    models = load_models()
    
    if models is None:
        with st.spinner("Training K-Means clustering model..."):
            df = load_and_preprocess_data()
            df, _, _ = engineer_features(df)
            
            X, y, feature_names = prepare_modeling_features(df)
            
            kmeans, pca, scaler, cluster_labels = train_kmeans_model(X)
            
            cluster_stats, cluster_counts = get_cluster_characteristics(df, cluster_labels, feature_names)
            
            return kmeans, pca, scaler, feature_names, df, cluster_labels, cluster_stats, cluster_counts
    else:
        kmeans = models['kmeans']
        pca = models['pca_km']
        scaler = models['scaler_km']
        feature_names = models['feature_names']
        
        df = load_and_preprocess_data()
        df, _, _ = engineer_features(df)
        X, _, _ = prepare_modeling_features(df)
        
        from sklearn.preprocessing import StandardScaler
        scaler_temp = StandardScaler()
        X_scaled = scaler_temp.fit_transform(X)
        
        from sklearn.decomposition import PCA
        pca_temp = PCA(n_components=2)
        X_pca = pca_temp.fit_transform(X_scaled)
        
        cluster_labels = kmeans.predict(X_pca)
        
        cluster_stats, cluster_counts = get_cluster_characteristics(df, cluster_labels, feature_names)
        
        return kmeans, pca, scaler, feature_names, df, cluster_labels, cluster_stats, cluster_counts

try:
    kmeans, pca, scaler, feature_names, df, cluster_labels, cluster_stats, cluster_counts = load_clustering_model()
    
    st.header("📊 Clustering Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Number of Clusters", "3")
    with col2:
        st.metric("Total Apps", f"{len(df):,}")
    with col3:
        st.metric("PCA Components", "2")
    with col4:
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA as PCA_sklearn
        scaler_temp = StandardScaler()
        X, _, _ = prepare_modeling_features(df)
        X_scaled = scaler_temp.fit_transform(X)
        pca_temp = PCA_sklearn(n_components=2)
        X_pca = pca_temp.fit_transform(X_scaled)
        sil_score = silhouette_score(X_pca, cluster_labels)
        st.metric("Silhouette Score", f"{sil_score:.4f}")
    
    st.info("""
    **About K-Means Clustering:**
    - Apps are grouped into **3 distinct clusters** based on their characteristics
    - **PCA** reduces 4 features to 2 components for clustering
    - Features: Log(Reviews), Log(Installs), Size, and Price
    - Silhouette Score measures cluster quality (higher is better, range: -1 to 1)
    """)
    
    st.markdown("---")
    
    st.header("📈 Cluster Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax.pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], 
               autopct='%1.1f%%', colors=colors, startangle=90, 
               textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax.set_title('Distribution of Apps Across Clusters', fontsize=14, fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.bar([f'Cluster {i}' for i in cluster_counts.index], cluster_counts.values, 
               color=colors, edgecolor='black', linewidth=2)
        ax.set_ylabel('Number of Apps', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_title('Apps per Cluster', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(cluster_counts.values):
            ax.text(i, v + len(df)*0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.header("🔍 Cluster Characteristics")
    
    st.markdown("### Average Values per Cluster")
    
    cluster_display = cluster_stats.copy()
    cluster_display.index = [f'Cluster {i}' for i in cluster_display.index]
    cluster_display.columns = ['Avg Log(Reviews)', 'Avg Log(Installs)', 'Avg Size (MB)', 'Avg Price ($)']
    
    st.dataframe(cluster_display.style.background_gradient(cmap='YlOrRd', axis=0), use_container_width=True)
    
    st.markdown("---")
    st.subheader("📝 Cluster Insights")
    
    col1, col2, col3 = st.columns(3)
    
    for idx, cluster_id in enumerate(cluster_stats.index):
        col = [col1, col2, col3][idx]
        with col:
            stats = cluster_stats.loc[cluster_id]
            count = cluster_counts.loc[cluster_id]
            
            if stats['Log_Installs'] > cluster_stats['Log_Installs'].median():
                cluster_type = "🏆 Popular Apps"
                description = "High installs and reviews"
            elif stats['Price'] > cluster_stats['Price'].median():
                cluster_type = "💰 Premium Apps"
                description = "Higher priced applications"
            else:
                cluster_type = "🌱 Niche Apps"
                description = "Lower popularity apps"
            
            st.markdown(f"#### Cluster {cluster_id}")
            st.markdown(f"**{cluster_type}**")
            st.write(f"*{description}*")
            st.write(f"**Apps:** {count:,} ({count/len(df)*100:.1f}%)")
            st.write(f"**Avg Reviews:** {np.expm1(stats['Log_Reviews']):,.0f}")
            st.write(f"**Avg Installs:** {np.expm1(stats['Log_Installs']):,.0f}")
            st.write(f"**Avg Size:** {stats['Size']:.1f} MB")
            st.write(f"**Avg Price:** ${stats['Price']:.2f}")
    
    st.markdown("---")
    
    st.header("🗺️ Cluster Visualization in PCA Space")
    
    X, _, _ = prepare_modeling_features(df)
    scaler_temp = StandardScaler()
    X_scaled = scaler_temp.fit_transform(X)
    pca_temp = PCA_sklearn(n_components=2)
    X_pca = pca_temp.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', 
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Principal Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Principal Component 2', fontsize=12, fontweight='bold')
    ax.set_title('Apps Clustered in PCA Space', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=300, alpha=0.8, 
              edgecolors='black', linewidth=2, marker='*', label='Centroids')
    
    plt.colorbar(scatter, label='Cluster', ax=ax)
    ax.legend(fontsize=10)
    st.pyplot(fig)
    
    st.markdown("---")
    
    st.header("🎯 Predict Your App's Cluster")
    st.markdown("Enter your app characteristics to see which cluster it belongs to:")
    
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
        predict_button = st.button("🔮 Predict Cluster", type="primary", use_container_width=True)
    
    if predict_button:
        input_data = [reviews, installs, size, price]
        
        predicted_cluster = predict_cluster(input_data, kmeans, pca, scaler, feature_names)
        
        st.markdown("---")
        st.header("🎊 Prediction Result")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>🎯 Cluster {predicted_cluster}</h1>", 
                       unsafe_allow_html=True)
            
            cluster_info = cluster_stats.loc[predicted_cluster]
            
            if cluster_info['Log_Installs'] > cluster_stats['Log_Installs'].median():
                cluster_type = "🏆 Popular Apps Cluster"
                description = "Your app belongs to the high-popularity cluster with significant installs and reviews!"
            elif cluster_info['Price'] > cluster_stats['Price'].median():
                cluster_type = "💰 Premium Apps Cluster"
                description = "Your app is in the premium cluster, characterized by higher pricing!"
            else:
                cluster_type = "🌱 Niche Apps Cluster"
                description = "Your app is in the niche cluster with more focused audience appeal!"
            
            st.markdown(f"<h3 style='text-align: center;'>{cluster_type}</h3>", 
                       unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>{description}</p>", 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("📊 Your App vs Cluster Average")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Input Summary:**")
            input_summary = {
                'Metric': ['Reviews', 'Installs', 'Size (MB)', 'Price ($)'],
                'Your App': [f"{reviews:,}", f"{installs:,}", f"{size:.1f}", f"${price:.2f}"]
            }
            st.dataframe(pd.DataFrame(input_summary), use_container_width=True, hide_index=True)
        
        with col2:
            st.write("**Cluster Average:**")
            cluster_avg = {
                'Metric': ['Reviews', 'Installs', 'Size (MB)', 'Price ($)'],
                'Cluster Avg': [
                    f"{np.expm1(cluster_info['Log_Reviews']):,.0f}",
                    f"{np.expm1(cluster_info['Log_Installs']):,.0f}",
                    f"{cluster_info['Size']:.1f}",
                    f"${cluster_info['Price']:.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(cluster_avg), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("🔍 All Clusters Comparison")
        
        comparison_data = []
        for cluster_id in cluster_stats.index:
            stats = cluster_stats.loc[cluster_id]
            count = cluster_counts.loc[cluster_id]
            comparison_data.append({
                'Cluster': f'Cluster {cluster_id}',
                'Apps Count': f"{count:,}",
                'Avg Reviews': f"{np.expm1(stats['Log_Reviews']):,.0f}",
                'Avg Installs': f"{np.expm1(stats['Log_Installs']):,.0f}",
                'Avg Size (MB)': f"{stats['Size']:.1f}",
                'Avg Price ($)': f"${stats['Price']:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        def highlight_cluster(row):
            if row['Cluster'] == f'Cluster {predicted_cluster}':
                return ['background-color: #90EE90'] * len(row)
            return [''] * len(row)
        
        st.dataframe(comparison_df.style.apply(highlight_cluster, axis=1), 
                    use_container_width=True, hide_index=True)
        
        st.success(f"✨ Your app is highlighted in the table above (Cluster {predicted_cluster})")
    
    st.markdown("---")
    st.header("ℹ️ Clustering Information")
    
    st.write("""
    ### How K-Means Clustering Works:
    
    1. **Feature Engineering**: Input features are log-transformed where appropriate
    2. **Standardization**: Features are scaled to have zero mean and unit variance
    3. **PCA Transformation**: The 4-dimensional feature space is reduced to 2 principal components
    4. **K-Means Algorithm**: Groups apps into 3 clusters based on similarity in PCA space
    5. **Cluster Assignment**: Each app is assigned to the nearest cluster centroid
    
    ### Cluster Interpretation:
    
    - **Cluster 0**: Typically represents apps with specific characteristic patterns
    - **Cluster 1**: Represents a different group with distinct features
    - **Cluster 2**: The third group with unique characteristics
    
    The actual meaning of each cluster depends on the data and can represent:
    - **Popularity levels** (high vs. low installs/reviews)
    - **Price segments** (free vs. paid apps)
    - **App size categories** (lightweight vs. heavy apps)
    
    ### Use Cases:
    - **Market Segmentation**: Understand different app categories in the market
    - **Competitor Analysis**: See which cluster similar apps belong to
    - **Strategic Planning**: Position your app based on cluster characteristics
    """)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please ensure the 'googleplaystore.csv' file is in the correct directory and try again.")
