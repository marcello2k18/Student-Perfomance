# app.py
import streamlit as st
import numpy as np
import pickle

# Load model dari file
@st.cache_resource
def load_model():
    with open("xgb_optuna_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🎓 Prediksi IPK Mahasiswa")

# Input fitur
rata2_nilai = st.slider("Rata-rata Nilai Angka", 50.0, 100.0, 75.0)
rata2_hadir = st.slider("Rata-rata Kehadiran", 0.0, 16.0, 13.0)
jumlah_mk = st.number_input("Jumlah Mata Kuliah Diambil", 1, 20, 10)

# Prediksi
if st.button("Prediksi IPK"):
    features = np.array([[rata2_nilai, rata2_hadir, jumlah_mk]])
    prediction = model.predict(features)[0]
    
if prediction >= 3.7:
    st.success(f"🎯 Prediksi IPK: {prediction:.2f} — Sangat Memuaskan!")
    st.balloons()
elif prediction >= 3.0:
    st.info(f"✅ Prediksi IPK: {prediction:.2f} — Cukup Baik")
else:
    st.warning(f"⚠️ Prediksi IPK: {prediction:.2f} — Perlu Perhatian")
    st.snow()  
