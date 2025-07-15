import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load data
df = pd.read_csv('house_prices.csv')
X = df[['LotArea', 'BedroomAbvGr', 'YearBuilt']]
y = df['SalePrice']

# Training model
model = LinearRegression()
model.fit(X, y)

# Simpan model ke file model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
