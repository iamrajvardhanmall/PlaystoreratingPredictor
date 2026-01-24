import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_and_preprocess_data, engineer_features

st.set_page_config(page_title="EDA - Google Play Store", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis")
st.markdown("### Comprehensive visualization and insights from Google Play Store data")

@st.cache_data
def load_data():
    try:
        # Debug: Show current directory and file paths
        import os
        current_dir = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        csv_path = os.path.join(parent_dir, 'googleplaystore.csv')
        
        # st.info(f"Current working directory: {current_dir}")
        # st.info(f"Script directory: {script_dir}")
        # st.info(f"Looking for CSV at: {csv_path}")
        # st.info(f"CSV file exists: {os.path.exists(csv_path)}")
        
        df = load_and_preprocess_data()
        df, _, _ = engineer_features(df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

df = load_data()

if df is not None:
    st.header("📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Apps", f"{len(df):,}")
    with col2:
        st.metric("Average Rating", f"{df['Rating'].mean():.2f}")
    with col3:
        st.metric("Total Categories", df['Category'].nunique())
    with col4:
        st.metric("Free Apps", f"{(df['Price'] == 0).sum():,}")
    
    st.markdown("---")
    
    st.header("⭐ Rating Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(df['Rating'], fill=True, alpha=0.6, color='blue', linewidth=2, ax=ax)
        ax.hist(df['Rating'], bins=30, alpha=0.3, color='gray', density=True)
        ax.set_title('Rating Distribution (KDE + Histogram)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Density')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        st.info(f"""
        **Insights:**
        - Mean Rating: **{df['Rating'].mean():.2f}**
        - Median Rating: **{df['Rating'].median():.2f}**
        - Most ratings are concentrated between 4.0 and 4.5
        - Very few apps have ratings below 2.0
        """)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        rating_counts = df['Rating'].value_counts().sort_index()
        ax.bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='black')
        ax.set_title('Rating Frequency Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.header("💰 Price and Installs Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        price_data = df[df['Price'] > 0]['Price']
        ax.hist(price_data, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.set_title('Price Distribution (Paid Apps)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Price ($)')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        st.info(f"""
        **Insights:**
        - Total Paid Apps: **{len(price_data):,}**
        - Average Price: **${price_data.mean():.2f}**
        - Max Price: **${price_data.max():.2f}**
        - Most apps are priced below $10
        """)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(np.log1p(df['Installs']), bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax.set_title('Log(Installs) Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Log(Installs + 1)')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        st.info(f"""
        **Insights:**
        - Total Installs: **{df['Installs'].sum():,}**
        - Median Installs: **{df['Installs'].median():,.0f}**
        - Distribution is heavily right-skewed
        """)
    
    st.markdown("---")
    
    st.header("💬 Reviews Distribution")
    fig, ax = plt.subplots(figsize=(12, 6))
    free_reviews = np.log1p(df[df['Type'] == 'Free']['Reviews'])
    paid_reviews = np.log1p(df[df['Type'] == 'Paid']['Reviews'])
    ax.hist(free_reviews, bins=25, alpha=0.7, label='Free', color='blue', edgecolor='black')
    ax.hist(paid_reviews, bins=25, alpha=0.7, label='Paid', color='red', edgecolor='black')
    ax.set_title('Log(Reviews) Distribution by App Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Log(Reviews + 1)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"""
        **Free Apps:**
        - Count: **{len(df[df['Type'] == 'Free']):,}**
        - Avg Reviews: **{df[df['Type'] == 'Free']['Reviews'].mean():,.0f}**
        """)
    with col2:
        st.success(f"""
        **Paid Apps:**
        - Count: **{len(df[df['Type'] == 'Paid']):,}**
        - Avg Reviews: **{df[df['Type'] == 'Paid']['Reviews'].mean():,.0f}**
        """)
    
    st.markdown("---")
    
    st.header("📦 App Size Distribution")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(df['Size'], fill=True, color='purple', alpha=0.5, linewidth=2, ax=ax)
    sns.rugplot(df['Size'], height=0.05, color='purple', alpha=0.3, ax=ax)
    ax.set_title('App Size Distribution (KDE + Rug Plot)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Size (MB)')
    ax.set_ylabel('Density')
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    st.info(f"""
    **Size Statistics:**
    - Average Size: **{df['Size'].mean():.2f} MB**
    - Median Size: **{df['Size'].median():.2f} MB**
    - Largest App: **{df['Size'].max():.2f} MB**
    - Most apps are under 50 MB
    """)
    
    st.markdown("---")
    
    st.header("📱 Category Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 8))
        top_categories = df['Category'].value_counts().head(10)
        y_pos = np.arange(len(top_categories))
        ax.barh(y_pos, top_categories.values, color='teal', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_categories.index)
        ax.set_xlabel('Number of Apps')
        ax.set_title('Top 10 App Categories', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 8))
        content_rating_avg = df.groupby('Content Rating')['Rating'].mean().sort_values(ascending=False)
        ax.bar(range(len(content_rating_avg)), content_rating_avg.values, color='coral', edgecolor='black')
        ax.set_xticks(range(len(content_rating_avg)))
        ax.set_xticklabels(content_rating_avg.index, rotation=45, ha='right')
        ax.set_title('Average Rating by Content Rating', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Rating')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.header("🆓 Free vs Paid Apps")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 8))
        type_counts = df['Type'].value_counts()
        colors = ['#66b3ff', '#ff9999']
        ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax.set_title('Free vs Paid Apps Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Key Insights:")
        st.write(f"""
        - **Free Apps:** {type_counts.get('Free', 0):,} ({type_counts.get('Free', 0)/len(df)*100:.1f}%)
        - **Paid Apps:** {type_counts.get('Paid', 0):,} ({type_counts.get('Paid', 0)/len(df)*100:.1f}%)
        
        **Average Ratings:**
        - Free: {df[df['Type'] == 'Free']['Rating'].mean():.2f}
        - Paid: {df[df['Type'] == 'Paid']['Rating'].mean():.2f}
        
        **Average Installs:**
        - Free: {df[df['Type'] == 'Free']['Installs'].mean():,.0f}
        - Paid: {df[df['Type'] == 'Paid']['Installs'].mean():,.0f}
        """)
    
    st.markdown("---")
    
    st.header("🔗 Correlation Analysis")
    numeric_cols = ['Rating', 'Reviews', 'Size', 'Installs', 'Price', 'Last Updated_Days']
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, 
                linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax, fmt='.2f')
    ax.set_title('Correlation Matrix of Numeric Features', fontsize=16, fontweight='bold')
    st.pyplot(fig)
    
    st.info("""
    **Key Correlations:**
    - Reviews and Installs show strong positive correlation
    - Price has weak correlation with most features
    - Rating shows moderate correlation with Reviews
    """)
    
    st.markdown("---")
    
    st.header("🏆 Top & Bottom Categories by Rating")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Categories")
        top_cats = df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(10)
        st.dataframe(top_cats.reset_index().rename(columns={'Category': 'Category', 'Rating': 'Avg Rating'}), 
                     use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Bottom 10 Categories")
        bottom_cats = df.groupby('Category')['Rating'].mean().sort_values().head(10)
        st.dataframe(bottom_cats.reset_index().rename(columns={'Category': 'Category', 'Rating': 'Avg Rating'}), 
                     use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.header("📊 Summary Statistics")
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
else:
    st.error("Failed to load data. Please ensure 'googleplaystore.csv' is in the correct directory.")
