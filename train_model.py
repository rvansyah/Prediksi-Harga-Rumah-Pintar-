import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv('house_prices.csv')   # Pastikan file house_prices.csv sudah ada di folder

# Pilih minimal 2 fitur dan 1 target
X = df[['LotArea', 'BedroomAbvGr', 'YearBuilt']]
y = df['SalePrice']

# Training model
model = LinearRegression()
model.fit(X, y)

# Simpan model ke file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
