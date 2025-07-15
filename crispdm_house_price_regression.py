import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Business Understanding: prediksi harga rumah

# 2. Data Understanding
df = pd.read_csv('house_prices.csv') # minimal 100 baris
print(df.describe())
print(df.isnull().sum())

# 3. Data Preparation
df = df.dropna() # atau imputasi
X = df[['LotArea', 'BedroomAbvGr', 'YearBuilt']] # minimal 2 fitur
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modelling
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Evaluation
r2 = r2_score(y_test, y_pred)
print(f'R2 Score: {r2:.2f}') # harus >= 0.6

# 6. Deployment
# Contoh deployment dengan Streamlit
# streamlit run app.py