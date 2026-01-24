# Google Play Store Analytics - Streamlit App

A comprehensive data analytics and machine learning application for Google Play Store data analysis, featuring interactive visualizations, rating predictions, and clustering insights.

## 🚀 Features

### 📊 Page 1: Exploratory Data Analysis (EDA)
- **Comprehensive Visualizations**: Rating distribution, price analysis, installs patterns
- **Category Analysis**: Top categories, content ratings, and trends
- **Correlation Analysis**: Feature relationships and patterns
- **Interactive Insights**: Real-time statistics and key findings

### 🤖 Page 2: Random Forest Rating Prediction
- **ML-Powered Predictions**: Predict app ratings using Random Forest Regressor
- **PCA Transformation**: Dimensionality reduction for improved performance
- **Interactive Input Form**: Enter app characteristics for instant predictions
- **Performance Metrics**: Model accuracy and feature importance visualization
- **Comparison Analysis**: Compare predictions with dataset averages

### 🎯 Page 3: K-Means Clustering
- **Cluster Discovery**: Identify which cluster your app belongs to
- **Visual Analytics**: 2D PCA visualization of clusters
- **Cluster Characteristics**: Detailed statistics for each cluster
- **Interactive Prediction**: Real-time cluster assignment based on user input
- **Silhouette Analysis**: Cluster quality metrics

## 📁 Project Structure

```
google_play_app/
│
├── app.py                          # Main Streamlit application
├── utils.py                        # Data processing and ML utilities
├── requirements.txt                # Python dependencies
├── googleplaystore.csv            # Dataset (required)
│
└── pages/
    ├── 1_📊_EDA.py                # EDA page
    ├── 2_🤖_Random_Forest_Prediction.py  # Prediction page
    └── 3_🎯_KMeans_Clustering.py  # Clustering page
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or navigate to the project directory:**
   ```bash
   cd google_play_app
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the dataset is present:**
   - Place `googleplaystore.csv` in the project root directory

## 🚦 Running the Application

### Local Development

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Access the app:**
   - Open your browser and navigate to: `http://localhost:8501`
   - The app will automatically open in your default browser

### Alternative Port

If port 8501 is in use:
```bash
streamlit run app.py --server.port 8502
```

## 📊 Dataset Requirements

The application expects a CSV file named `googleplaystore.csv` with the following columns:
- `App`: Application name
- `Category`: App category
- `Rating`: User rating (1-5)
- `Reviews`: Number of reviews
- `Size`: App size
- `Installs`: Number of installs
- `Type`: Free or Paid
- `Price`: App price
- `Content Rating`: Age rating
- `Genres`: App genres
- `Last Updated`: Last update date

## 🔧 Configuration

### Modify Model Parameters

Edit `utils.py` to adjust:
- Random Forest estimators: `n_estimators=200`
- K-Means clusters: `n_clusters=3`
- PCA components: `n_components=2`
- Train-test split ratio: `test_size=0.2`

### Customize Visualizations

Edit page files in the `pages/` directory to:
- Change color schemes
- Adjust plot sizes
- Modify chart types
- Update styling

## 📈 Model Information

### Random Forest Model
- **Algorithm**: Random Forest Regressor
- **Features**: Log(Reviews), Log(Installs), Size, Price
- **Preprocessing**: StandardScaler + PCA
- **Components**: 2 principal components
- **Trees**: 200 estimators

### K-Means Clustering
- **Algorithm**: K-Means
- **Clusters**: 3
- **Features**: Same as Random Forest
- **Preprocessing**: StandardScaler + PCA
- **Metric**: Silhouette Score for quality assessment

## 🎯 Use Cases

1. **Market Analysis**: Understand app market trends and patterns
2. **Rating Prediction**: Estimate potential ratings for new apps
3. **Competitor Analysis**: Compare app characteristics with market clusters
4. **Strategic Planning**: Make data-driven decisions for app development
5. **Portfolio Optimization**: Identify gaps in app offerings

## 📝 Features Explained

### Data Preprocessing
- **Missing Value Handling**: Imputation with mean/median
- **Size Conversion**: Standardizes MB and KB formats
- **Price Cleaning**: Removes currency symbols
- **Date Processing**: Calculates days since last update

### Feature Engineering
- **Log Transformations**: Normalizes skewed distributions
- **Categorical Encoding**: Label and One-Hot encoding
- **Derived Features**: Price categories, size bins, rating categories

### Machine Learning Pipeline
1. Load and clean data
2. Engineer features
3. Apply standardization
4. Perform PCA transformation
5. Train models (Random Forest & K-Means)
6. Cache models for quick predictions

## 🐛 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: googleplaystore.csv not found`
- **Solution**: Ensure the CSV file is in the project root directory

**Issue**: `ModuleNotFoundError`
- **Solution**: Install all requirements: `pip install -r requirements.txt`

**Issue**: Models taking too long to train
- **Solution**: Models are cached after first run; subsequent runs are instant

**Issue**: Port already in use
- **Solution**: Use alternative port: `streamlit run app.py --server.port 8502`

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Deploy with one click

### Docker Deployment

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t play-store-app .
docker run -p 8501:8501 play-store-app
```

## 📚 Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib & Seaborn**: Data visualization
- **XGBoost**: Gradient boosting (available for extension)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add more ML models (XGBoost, Neural Networks)
- Implement A/B testing features
- Add download functionality for predictions
- Create API endpoints
- Enhance mobile responsiveness

## 📄 License

This project is open source and available under the MIT License.

## 👥 Authors

Developed with ❤️ for data science and app analytics enthusiasts.

## 🙏 Acknowledgments

- Google Play Store for the dataset
- Streamlit community for excellent documentation
- Scikit-learn for robust ML tools

---

**Note**: This application is for educational and analytical purposes. Predictions are based on historical data patterns and may not reflect actual future ratings.

## 📞 Support

For issues, questions, or suggestions:
- Create an issue in the repository
- Check the troubleshooting section
- Review Streamlit documentation

**Happy Analyzing! 📱📊🚀**
