import streamlit as st
import numpy as np
import pickle
import os

st.title("Prediksi Harga Rumah Pintar")

# Cek keberadaan file model.pkl
MODEL_PATH = 'model.pkl'
if not os.path.exists(MODEL_PATH):
    st.error(f"File model.pkl tidak ditemukan!\n\nPastikan Anda sudah melakukan training model dan file model.pkl ada di folder yang sama dengan app.py.")
    st.stop()

# Load model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Input fitur
lot_area = st.number_input('Luas Tanah (m2)', min_value=0)
bedrooms = st.number_input('Jumlah Kamar Tidur', min_value=0)
year_built = st.number_input('Tahun Dibangun', min_value=1800, max_value=2025)

if st.button('Prediksi Harga'):
    # Validasi input sederhana
    if lot_area <= 0 or bedrooms <= 0 or year_built <= 1800:
        st.warning("Isi semua data dengan benar!")
    else:
        # Prediksi harga rumah
        features = np.array([[lot_area, bedrooms, year_built]])
        predicted_price = model.predict(features)
        st.success(f"Harga Prediksi: Rp {predicted_price[0]:,.0f}")
