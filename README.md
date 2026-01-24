# 📱 Play Store App Rating Predictor

A **Machine Learning–based web application** that predicts the **Google Play Store app rating** using app metadata such as installs, reviews, category, size, type, and price.  
The project implements an end-to-end ML pipeline and is deployed using **Streamlit** for interactive user input and real-time predictions.

🔗 **Live App:** https://storeapprating.streamlit.app/  
🔗 **GitHub Repository:** https://github.com/iamrajvardhanmall/PlaystoreratingPredictor  

---

## 📌 Project Overview

Mobile app ratings significantly influence user trust and app success.  
This project predicts an app’s rating based on historical Google Play Store data using machine learning techniques.

The application enables users to:
- Analyze app-related features
- Input app parameters
- Obtain a predicted rating instantly

---

## 🚀 Features

- Complete **Machine Learning pipeline**
- Data cleaning and preprocessing
- Feature engineering and scaling
- **Dimensionality Reduction using PCA**
- **Random Forest–based prediction model**
- Interactive **Streamlit web interface**
- Real-time rating prediction

---

## 🧠 Machine Learning Workflow

1. **Data Collection**
   - Google Play Store apps dataset

2. **Data Preprocessing**
   - Handling missing values
   - Label encoding categorical variables
   - Log transformation of skewed features
   - Feature scaling using StandardScaler

3. **Dimensionality Reduction**
   - Principal Component Analysis (PCA) to reduce dimensionality and improve performance

4. **Model Training**
   - Random Forest model trained on PCA-transformed features

5. **Evaluation**
   - Model performance evaluated using appropriate metrics

6. **Deployment**
   - Trained model integrated into a Streamlit application

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn  
- **Dimensionality Reduction:** PCA  
- **Model:** Random Forest  
- **Web Framework:** Streamlit  

---

## 📂 Project Structure

