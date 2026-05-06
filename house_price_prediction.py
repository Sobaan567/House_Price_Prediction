"""
╔══════════════════════════════════════════════════════════╗
║       HOUSE PRICE PREDICTION — ML Beginner Project       ║
║       Dataset: Ames Housing (synthetic replica)          ║
║       Models: Linear Regression + Random Forest          ║
╚══════════════════════════════════════════════════════════╝

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn

Run:
    python house_price_prediction.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════
# STEP 1 — GENERATE DATASET
# (Replace this block with pd.read_csv('train.csv') once
#  you download the real Kaggle dataset)
# ══════════════════════════════════════════════════════════
def generate_dataset(n=1460, seed=42):
    """Creates a realistic synthetic Ames Housing-style dataset."""
    np.random.seed(seed)

    neighborhoods = ['NAmes','CollgCr','OldTown','Edwards','Somerst',
                     'NridgHt','Gilbert','Sawyer','NWAmes','SawyerW']
    premiums = {'NAmes':1.0,'CollgCr':1.15,'OldTown':0.85,'Edwards':0.88,
                'Somerst':1.18,'NridgHt':1.35,'Gilbert':1.1,'Sawyer':0.92,
                'NWAmes':1.05,'SawyerW':1.08}

    neigh        = np.random.choice(neighborhoods, n)
    gr_liv_area  = np.random.randint(500,  4000, n)
    overall_qual = np.random.randint(1,    11,   n)
    year_built   = np.random.randint(1900, 2010, n)
    total_bsmt   = np.random.randint(0,    2500, n)
    full_bath    = np.random.randint(0,    4,    n)
    bedrooms     = np.random.randint(0,    6,    n)
    garage_cars  = np.random.randint(0,    4,    n)
    lot_area     = np.random.randint(1300, 20000, n)

    prem = np.array([premiums[x] for x in neigh])

    price = (
        50000
        + gr_liv_area  * 60
        + overall_qual * 12000
        + (2010 - year_built) * -300
        + total_bsmt   * 25
        + full_bath    * 8000
        + bedrooms     * 5000
        + garage_cars  * 10000
        + lot_area     * 2
    ) * prem + np.random.normal(0, 15000, n)

    price = np.clip(price, 50000, 750000).astype(int)

    return pd.DataFrame({
        'Neighborhood': neigh,
        'GrLivArea':    gr_liv_area,
        'OverallQual':  overall_qual,
        'YearBuilt':    year_built,
        'TotalBsmtSF':  total_bsmt,
        'FullBath':     full_bath,
        'BedroomAbvGr': bedrooms,
        'GarageCars':   garage_cars,
        'LotArea':      lot_area,
        'SalePrice':    price
    })


# ══════════════════════════════════════════════════════════
# STEP 2 — LOAD & EXPLORE
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  HOUSE PRICE PREDICTION — ML PROJECT")
print("="*55)

# If you have the real Kaggle CSV, use this instead:
# df = pd.read_csv('train.csv')
df = generate_dataset()

print(f"\n[1] Dataset loaded — Shape: {df.shape}")
print(df.head(3).to_string())

print(f"\n[2] Missing values:\n{df.isnull().sum()}")

print(f"\n[3] Price summary:")
print(df['SalePrice'].describe().apply(lambda x: f"${x:,.0f}"))


# ══════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
# Create new useful features
df['HouseAge']     = 2010 - df['YearBuilt']       # age is clearer than year
df['PricePerSqft'] = df['SalePrice'] / df['GrLivArea']  # insight metric

# Encode the categorical 'Neighborhood' column into numbers
le = LabelEncoder()
df['Neighborhood_enc'] = le.fit_transform(df['Neighborhood'])

print(f"\n[4] Feature engineering done — new columns: HouseAge, Neighborhood_enc")


# ══════════════════════════════════════════════════════════
# STEP 4 — PREPARE X (inputs) and y (target)
# ══════════════════════════════════════════════════════════
features = [
    'GrLivArea',      # living area in sqft
    'OverallQual',    # quality rating 1–10
    'HouseAge',       # age of house
    'TotalBsmtSF',    # basement area
    'FullBath',       # number of full bathrooms
    'BedroomAbvGr',   # number of bedrooms
    'GarageCars',     # garage capacity
    'LotArea',        # lot size
    'Neighborhood_enc'# encoded location
]

X = df[features]
y = df['SalePrice']

# 80% training data, 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[5] Train size: {len(X_train)} | Test size: {len(X_test)}")


# ══════════════════════════════════════════════════════════
# STEP 5 — TRAIN MODEL 1: Linear Regression
# ══════════════════════════════════════════════════════════
print("\n" + "-"*40)
print("  MODEL 1 — Linear Regression")
print("-"*40)

lr = LinearRegression()
lr.fit(X_train, y_train)     # ← learning happens here
lr_pred = lr.predict(X_test)

lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2   = r2_score(y_test, lr_pred)

print(f"MAE  (avg dollar error) : ${lr_mae:>10,.0f}")
print(f"RMSE (root mean sq err) : ${lr_rmse:>10,.0f}")
print(f"R²   (1.0 = perfect)    : {lr_r2:>10.4f}")

# What did the model learn?
print("\nLearned coefficients ($ per unit change):")
for feat, coef in sorted(zip(features, lr.coef_), key=lambda x: -abs(x[1])):
    print(f"  {feat:20s}: ${coef:>10,.0f}")


# ══════════════════════════════════════════════════════════
# STEP 6 — TRAIN MODEL 2: Random Forest
# ══════════════════════════════════════════════════════════
print("\n" + "-"*40)
print("  MODEL 2 — Random Forest")
print("-"*40)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2   = r2_score(y_test, rf_pred)

print(f"MAE  (avg dollar error) : ${rf_mae:>10,.0f}")
print(f"RMSE (root mean sq err) : ${rf_rmse:>10,.0f}")
print(f"R²   (1.0 = perfect)    : {rf_r2:>10.4f}")

print("\nFeature importances:")
importances = sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1])
for feat, imp in importances:
    bar = '█' * int(imp * 40)
    print(f"  {feat:20s} {bar} {imp:.3f}")


# ══════════════════════════════════════════════════════════
# STEP 7 — COMPARE MODELS
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  MODEL COMPARISON")
print("="*55)
print(f"{'Metric':<12} {'Linear Reg':>15} {'Random Forest':>15}")
print("-"*44)
print(f"{'MAE':<12} ${lr_mae:>14,.0f} ${rf_mae:>14,.0f}")
print(f"{'RMSE':<12} ${lr_rmse:>14,.0f} ${rf_rmse:>14,.0f}")
print(f"{'R²':<12} {lr_r2:>15.4f} {rf_r2:>15.4f}")
winner = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
print(f"\n  Winner: {winner} 🏆")


# ══════════════════════════════════════════════════════════
# STEP 8 — PREDICT A NEW HOUSE
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  PREDICT A NEW HOUSE")
print("="*55)

new_house = pd.DataFrame([{
    'GrLivArea':       1500,   # 1500 sqft living area
    'OverallQual':     7,      # quality rating 7/10
    'HouseAge':        15,     # 15 years old
    'TotalBsmtSF':     800,    # 800 sqft basement
    'FullBath':        2,      # 2 bathrooms
    'BedroomAbvGr':    3,      # 3 bedrooms
    'GarageCars':      2,      # 2-car garage
    'LotArea':         8500,   # 8500 sqft lot
    'Neighborhood_enc': 3      # encoded neighborhood
}])

lr_price = lr.predict(new_house)[0]
rf_price = rf.predict(new_house)[0]

print(f"  House: 1500 sqft | 3 bed | 2 bath | 15yr old | 2-car garage")
print(f"  Linear Regression prediction : ${lr_price:>10,.0f}")
print(f"  Random Forest prediction     : ${rf_price:>10,.0f}")


# ══════════════════════════════════════════════════════════
# STEP 9 — PLOTS
# ══════════════════════════════════════════════════════════
print("\n[Generating plots...]\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('House Price Prediction — ML Project', fontsize=15, fontweight='bold')

# Plot 1: Price Distribution
axes[0,0].hist(df['SalePrice'], bins=40, color='steelblue', edgecolor='white', linewidth=0.5)
axes[0,0].set_title('Sale Price Distribution')
axes[0,0].set_xlabel('Sale Price ($)')
axes[0,0].set_ylabel('Count')
axes[0,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))

# Plot 2: Area vs Price scatter
axes[0,1].scatter(df['GrLivArea'], df['SalePrice'], alpha=0.3, color='steelblue', s=10)
axes[0,1].set_title('Living Area vs Sale Price')
axes[0,1].set_xlabel('GrLivArea (sqft)')
axes[0,1].set_ylabel('Sale Price ($)')
axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))

# Plot 3: Actual vs Predicted (Random Forest)
axes[1,0].scatter(y_test, rf_pred, alpha=0.4, color='seagreen', s=15)
mn = min(y_test.min(), rf_pred.min())
mx = max(y_test.max(), rf_pred.max())
axes[1,0].plot([mn,mx],[mn,mx],'r--', linewidth=1.5, label='Perfect prediction')
axes[1,0].set_title(f'Actual vs Predicted — Random Forest (R²={rf_r2:.2f})')
axes[1,0].set_xlabel('Actual Price ($)')
axes[1,0].set_ylabel('Predicted Price ($)')
axes[1,0].legend()
axes[1,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
axes[1,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))

# Plot 4: Feature Importance
feat_names, feat_imps = zip(*importances)
colors = ['steelblue' if i < 3 else 'lightsteelblue' for i in range(len(feat_names))]
axes[1,1].barh(feat_names[::-1], [x for x in feat_imps[::-1]], color=colors[::-1])
axes[1,1].set_title('Feature Importance (Random Forest)')
axes[1,1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('house_price_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as house_price_results.png")
print("\n✅ Project complete!")