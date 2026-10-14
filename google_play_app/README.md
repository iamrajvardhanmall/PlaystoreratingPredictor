# Google Play Store Analytics Dashboard

A Streamlit-based data science project for exploring the Google Play Store dataset, visualizing app market trends, predicting app ratings, and grouping apps into clusters using machine learning.

This project is designed for analysts, data enthusiasts, and developers who want a fast, interactive way to understand what drives app quality and performance in the Google Play ecosystem.

## Overview

The app provides three main capabilities:

- Exploratory Data Analysis (EDA) to understand app distribution, pricing, installs, and category trends
- Rating prediction using a Random Forest regressor
- App segmentation using K-Means clustering to uncover distinct market groups

## Why this project

Google Play Store data contains many signals that can help answer practical questions such as:

- Which categories have the highest rating trends?
- How do install counts and app size relate to ratings?
- Can we estimate a likely rating for a new app based on measurable attributes?
- Which app profiles cluster together in the market?

This dashboard turns those questions into an interactive, beginner-friendly analytical workflow.

## Features

### 1. Exploratory Data Analysis

The EDA page includes:

- Rating distribution insights
- Pricing and app size analysis
- Install count trends and market patterns
- Category-wise comparisons
- Content rating summaries
- Correlation analysis between app attributes

### 2. Rating Prediction with Random Forest

Users can input characteristics such as:

- Reviews
- Installs
- Size
- Price

The app then preprocesses the data, applies PCA-based dimensionality reduction, and predicts a rating using a trained Random Forest Regressor.

### 3. K-Means Clustering

This section clusters apps into groups based on the same feature set, helping identify common patterns such as:

- Popular lower-cost apps
- Large-scale installed apps
- Premium or niche app segments

The dashboard visualizes cluster boundaries and summarizes cluster-level characteristics.

## Tech stack

- Python 3.9+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- XGBoost (included in dependencies for extension or experimentation)

## Project structure

```text
google_play_app/
├── app.py                     # Main Streamlit entry point
├── utils.py                   # Data cleaning, feature engineering, ML training utilities
├── requirements.txt           # Python dependencies
├── googleplaystore.csv        # Google Play Store dataset
├── models.pkl                 # Serialized trained models (generated at runtime)
├── pages/
│   ├── 1_📊_EDA.py
│   ├── 2_🤖_Random_Forest_Prediction.py
│   └── 3_🎯_KMeans_Clustering.py
└── README.md
```

## Dataset

The app uses a curated Google Play Store dataset (`googleplaystore.csv`) containing metadata such as:

- App name
- Category
- Rating
- Reviews
- Size
- Installs
- Type (Free/Paid)
- Price
- Content Rating
- Genres
- Last Updated date

The preprocessing logic in `utils.py` handles missing values, numeric conversion, price parsing, app-size normalization, and feature engineering before model training.

## Data science workflow

The project follows a practical ML workflow:

1. Load raw Google Play Store data
2. Clean and normalize fields
3. Engineer derived features such as log-scaled reviews and installs
4. Standardize features
5. Reduce dimensionality with PCA
6. Train:
   - Random Forest Regressor for rating prediction
   - K-Means for clustering
7. Serve interactive results through Streamlit

## Prerequisites

Before running the project, make sure you have:

- Python 3.9 or newer
- pip
- A local environment manager such as `venv` or Conda

## Quick start

### 1. Navigate to the app directory

```bash
cd google_play_app
```

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure the dataset is available

Place the dataset file in the same folder as the app:

```text
google_play_app/googleplaystore.csv
```

If the dataset is missing, the app will not be able to run its analysis and prediction logic.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Then open the browser URL displayed in the terminal, usually:

```text
http://localhost:8501
```

If port 8501 is busy, use:

```bash
streamlit run app.py --server.port 8502
```

## App pages

### EDA Page

Use this page to explore:

- App rating density
- Price and install distributions
- Category performance variations
- Content rating segmentation
- Relationship among quantitative metrics

### Prediction Page

This page trains and uses a Random Forest model to estimate a likely rating from user-provided values. It is ideal for experimentation such as evaluating how different app characteristics may impact app quality.

### Clustering Page

This page groups apps by similarity and highlights cluster profiles. It helps identify patterns in the marketplace and understand where an app may fit among similar products.

## Model details

### Random Forest Regressor

- Algorithm: `RandomForestRegressor`
- Input features: log-scaled reviews, log-scaled installs, size, and price
- Dimensionality reduction: PCA applied after StandardScaler
- Train/test split: 80/20
- Random state: fixed for reproducibility

### K-Means clustering

- Algorithm: `KMeans`
- Number of clusters: 3
- Input features: same processed characteristics used for the regression workflow
- Visualization: PCA-reduced 2D chart to show cluster separation

## Customization

You can adjust the modeling and app behavior in `utils.py`:

- Number of Random Forest estimators
- Number of K-Means clusters
- PCA component count
- Train/test split ratio
- Feature engineering logic

You can also update styling and chart visuals in the page files under `pages/`.

## Troubleshooting

### Dataset not found

If you see a file-loading error, confirm that `googleplaystore.csv` exists in the `google_play_app` folder.

### Missing Python packages

Install the project requirements again:

```bash
pip install -r requirements.txt
```

### Streamlit launch issues

If the app fails to start:

- Verify the virtual environment is active
- Confirm Python version is 3.9+
- Reinstall project dependencies

### Port already in use

Use a different port:

```bash
streamlit run app.py --server.port 8502
```

## Deployment ideas

This app can be deployed in several ways:

- Streamlit Community Cloud
- Hugging Face Spaces
- Docker container
- Self-hosted server environment

Example Docker pattern:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## Contributing

Contributions are welcome. Possible improvements include:

- Adding more advanced ML models
- Improving preprocessing for noisy dataset entries
- Adding download/export options for insights
- Building a richer dashboard layout
- Enhancing model interpretability and metrics

## License

This repository does not currently include a formal license file. If you plan to share or distribute the project publicly, add a license such as MIT before release.

## Acknowledgments

- Google Play Store dataset source
- Streamlit community
- scikit-learn ecosystem
- Data science and app analytics communities

## Notes

This project is intended for educational and exploratory analysis. Model predictions are based on historical dataset patterns and should be treated as informative estimates rather than guarantees of real-world performance.

---

Built for interactive app-market analysis and machine learning exploration.
