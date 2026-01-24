# 📱 Play Store App Rating Predictor

An end-to-end **Machine Learning and Data Science project** that predicts **Google Play Store app ratings** using app metadata.  
This project demonstrates the complete ML lifecycle: **data preprocessing, exploratory data analysis (EDA), dimensionality reduction using PCA, model building with Random Forest and K-Means, and deployment using Streamlit**.

🔗 **Live Application:**  
https://storeapprating.streamlit.app/

🔗 **GitHub Repository:**  
https://github.com/iamrajvardhanmall/PlaystoreratingPredictor

---

## 📌 Project Overview

App ratings strongly influence visibility, downloads, and user trust on the Google Play Store.  
This project analyzes Play Store app data to:
- Predict app ratings using supervised learning
- Segment apps using unsupervised learning
- Provide insights through interactive visualizations

---

## 🎯 Objectives

- Perform detailed **Exploratory Data Analysis (EDA)**
- Clean and preprocess real-world Play Store data
- Apply **Principal Component Analysis (PCA)** for dimensionality reduction
- Build and evaluate:
  - **Random Forest** for rating prediction
  - **K-Means Clustering** for app segmentation
- Deploy a **multi-page Streamlit web application**

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Collection
The dataset contains Google Play Store app attributes such as:
- Category  
- Reviews  
- Installs  
- App Size  
- Type (Free / Paid)  
- Content Rating  
- Genres  
- Price  

---

### 2️⃣ Data Preprocessing
- Handling missing values using **SimpleImputer**
- Encoding categorical features using **Label Encoding**
- Scaling numerical features using **StandardScaler**
- Log transformation applied to skewed variables

---

### 3️⃣ Exploratory Data Analysis (EDA)
Key analyses include:
- Rating distribution
- Reviews vs Ratings
- Installs vs Ratings
- Category-wise rating comparison
- Outlier detection

Visualizations are created using **Matplotlib** and **Seaborn**.

---

### 4️⃣ Dimensionality Reduction (PCA)
- **PCA** is applied to:
  - Reduce dimensionality
  - Improve model efficiency
  - Remove multicollinearity
- PCA is applied **before Random Forest and K-Means**

---

### 5️⃣ Model Development

#### 🔹 Random Forest (Supervised Learning)
- Predicts app ratings
- Trained on PCA-transformed features
- Evaluated using accuracy and error metrics

#### 🔹 K-Means Clustering (Unsupervised Learning)
- Groups apps based on similarity
- Helps identify hidden patterns and segments
- Applied on PCA-reduced data

---

### 6️⃣ Model Evaluation
- **Random Forest:** Prediction accuracy and reliability
- **K-Means:** Cluster interpretation and PCA-based visualization

---

## 🌐 Streamlit Application

The project is deployed as a **three-page Streamlit application**.

### Page 1: Exploratory Data Analysis
- Interactive charts
- Key insights from Play Store data

### Page 2: Rating Prediction (Random Forest + PCA)
- User inputs app details
- Model predicts expected app rating

### Page 3: App Clustering (K-Means + PCA)
- Displays cluster grouping
- Shows which segment an app belongs to

---

## 🛠️ Technologies Used

| Category | Tools |
|--------|------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Dimensionality Reduction | PCA |
| Deployment | Streamlit |
| Environment | Anaconda, Jupyter Notebook |

---

