# 🚗 Car Price Prediction Model

This project uses machine learning to predict car prices based on a variety of features such as car specifications, engine details, body type, and manufacturer. It is implemented using Python, with extensive data preprocessing, exploratory data analysis (EDA), and a linear regression model.

## 📁 Files

- `car_price_prediction_model.py` – Main script for data loading, cleaning, preprocessing, and modeling.
- `Car_Price_Prediction_Model.ipynb` – Jupyter Notebook version of the project with visualizations and step-by-step explanation.
- `CarsData.csv` – Main Dataset of the Model

## 🧠 Features

- Cleans and preprocesses raw car data (including handling outliers, encoding categories, and scaling).
- Extracts car manufacturer names from car names.
- Uses dummy variables for categorical features with multiple levels.
- Builds and evaluates a linear regression model.
- Calculates performance metrics: Mean Squared Error and R-squared.

## 📊 Technologies Used

- **Python**
- **Pandas, NumPy** – Data manipulation
- **Matplotlib, Seaborn** – Visualization
- **Scikit-learn** – Preprocessing, Model Training, Evaluation
- **Statsmodels** – Statistical modeling and VIF analysis

## 🧹 Data Cleaning Highlights

- Extracted manufacturer names and fixed typos (e.g., 'maxda' → 'mazda').
- Removed outliers using the IQR method.
- Encoded binary categorical variables and one-hot encoded multi-level categorical features.

## 📈 Model Performance

After training a linear regression model, the following metrics were observed:

- **R-squared**: `0.445` – Indicates that approximately 44.5% of the variability in car prices is explained by the model.
- **Mean Squared Error (MSE)**: `0.0257` – Represents the average squared difference between predicted and actual prices.

These values suggest the model captures some key trends but may benefit from further optimization (e.g., feature engineering or using more advanced algorithms).

## 🚀 Getting Started

1. Clone the repository.
2. Run either the `.py` script or Jupyter Notebook.

```bash
git clone https://github.com/mohanadsabha/car-price-prediction.git
cd car-price-prediction
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
python car_price_prediction_model.py
