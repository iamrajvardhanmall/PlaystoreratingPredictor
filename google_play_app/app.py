import streamlit as st

st.set_page_config(
    page_title="Google Play Store Analytics",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📱 Google Play Store Analytics Dashboard")

st.markdown("""
## Welcome to the Google Play Store Analysis Platform

This interactive dashboard provides comprehensive insights into Google Play Store applications through:

### 📊 Pages Overview

1. **EDA (Exploratory Data Analysis)**
   - Comprehensive visualizations of app ratings, prices, installs, and reviews
   - Category analysis and content rating distributions
   - Correlation analysis between features
   - Key insights from the dataset

2. **Random Forest Prediction**
   - Predict app ratings using Random Forest Regressor
   - Model trained on PCA-transformed features
   - Interactive input form for app characteristics
   - Real-time predictions based on user inputs

3. **K-Means Clustering**
   - Discover which cluster your app belongs to
   - Clustering performed on PCA-reduced dimensions
   - Understand app groupings and patterns
   - Cluster characteristics and insights

---

### 🚀 Getting Started

👈 **Select a page from the sidebar** to begin exploring the data and making predictions!

### 📈 Dataset Information

The analysis is based on the Google Play Store dataset containing:
- App ratings, reviews, and install counts
- Pricing information and app sizes
- Category classifications
- Content ratings and last update information

### 🔬 Methodology

Our analysis pipeline includes:
- **Data Cleaning**: Handling missing values and outliers
- **Feature Engineering**: Creating derived features and encodings
- **Dimensionality Reduction**: PCA for feature compression
- **Machine Learning**: Random Forest for regression and K-Means for clustering

---

*Developed with ❤️ using Streamlit, scikit-learn, and XGBoost*
""")

st.sidebar.title("About")
st.sidebar.info("""
This application analyzes Google Play Store data to provide:
- Visual insights through EDA
- Rating predictions using ML
- App clustering patterns

**Technologies Used:**
- Python
- Streamlit
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
""")

st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Google Play Store Analytics")
