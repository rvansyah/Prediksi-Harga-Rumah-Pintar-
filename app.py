import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
st.title("Prediksi Harga Rumah")

lot_area = st.number_input('Luas Tanah')
bedrooms = st.number_input('Jumlah Kamar')
year_built = st.number_input('Tahun Dibangun')

if st.button('Prediksi'):
    pred = model.predict(np.array([[lot_area, bedrooms, year_built]]))
    st.write(f"Harga Prediksi: Rp {pred[0]:,.0f}")
