🏠 House Price Prediction — Machine Learning Project

A beginner-friendly machine learning project that predicts house prices using **Linear Regression** and **Random Forest**. Built with Python and scikit-learn, based on the Ames Housing dataset structure.

---

📌 Project Overview

This project walks through a complete ML pipeline from scratch:

- Generating / loading a housing dataset
- Cleaning and exploring the data
- Engineering new features
- Training two ML models
- Evaluating and comparing model performance
- Predicting the price of a new house
- Visualizing results with plots

---

📊 Results

| Metric | Linear Regression | Random Forest |
|--------|:-----------------:|:-------------:|
| MAE (avg error) | $38,343 | $33,920 |
| RMSE | $49,263 | $43,351 |
| R² Score | 0.72 | **0.78** ✅ |

**Winner: Random Forest** with an R² of 0.78 (78% of price variation explained by the model).

---

 🗂️ Project Structure

```
house_price_prediction/
│
├── house_price_prediction.py   # main script — run this
├── house_price_results.png     # output plots (auto-generated)
├── requirements.txt            # required libraries
└── README.md                   # this file
```

---

⚙️ Requirements

- Python 3.8+
- pip

Install all dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the project
python house_price_prediction.py
```

That's it! The script will train both models, print all results, and save a plot as `house_price_results.png`.

---

 🔧 Features Used

| Feature | Description |
|---------|-------------|
| `GrLivArea` | Above-ground living area (sqft) |
| `OverallQual` | Overall material and finish quality (1–10) |
| `HouseAge` | Age of the house in years |
| `TotalBsmtSF` | Total basement area (sqft) |
| `FullBath` | Number of full bathrooms |
| `BedroomAbvGr` | Number of bedrooms |
| `GarageCars` | Garage capacity (cars) |
| `LotArea` | Lot size (sqft) |
| `Neighborhood` | Location / neighborhood |

---

 🧠 Models

### Linear Regression
Learns a straight-line relationship between features and price. Fast, simple, and interpretable — you can read exactly how much each feature contributes to the final price.

### Random Forest
An ensemble of 100 decision trees. More powerful than linear regression because it captures non-linear patterns in the data.



 📈 Output Plots

The script automatically generates a 4-panel figure:

1. **Price Distribution** — histogram of all sale prices
2. **Area vs Price** — scatter plot showing the relationship
3. **Actual vs Predicted** — how close the model's guesses are
4. **Feature Importance** — which features drive price the most

---

🔄 Using the Real Kaggle Dataset

This project uses a synthetic dataset by default. To use the real Ames Housing data:

1. Download from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
2. Place `train.csv` in the project folder
3. Replace the `generate_dataset()` call in the script with:

```python
df = pd.read_csv('train.csv')
```


 📚 What I Learned

- How a full ML pipeline works end-to-end
- Data cleaning and feature engineering
- Difference between Linear Regression and Random Forest
- How to evaluate a model using MAE, RMSE, and R²
- How feature importance helps understand model decisions



 🛠️ Built With

- [Python](https://www.python.org/)
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

---

 👤 Author

**Your Name**
- GitHub: Sobaan567


END
 
