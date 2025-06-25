import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import sklearn

# Load model dari file
@st.cache_resource
def load_model():
    model_path = "xgb_optuna_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ File model tidak ditemukan. Pastikan 'xgb_optuna_model.pkl' ada di root folder.")
        st.stop()
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🎓 Prediksi IPK Mahasiswa")

# Input fitur dengan batas yang sesuai Data
rata2_nilai = st.slider("Rata-rata Nilai Angka", 0.0, 100.0, 75.0)      # dari 0–100
rata2_hadir = st.slider("Rata-rata Kehadiran", 0.0, 14.0, 12.0)         # dari 0–14
jumlah_mk = st.number_input("Jumlah Mata Kuliah Diambil", 1, 20, 10)    # tetap


# Prediksi
if st.button("Prediksi IPK"):
    features = pd.DataFrame([{
        "rata2_nilai": rata2_nilai,
        "rata2_hadir": rata2_hadir,
        "jumlah_mk_diambil": jumlah_mk
    }])

    prediction = model.predict(features)[0]

    if prediction >= 3.7:
        st.balloons()
        st.success(f"🎯 Prediksi IPK: {prediction:.2f} — Sangat Memuaskan!")

    elif prediction >= 3.0:
        st.balloons()
        st.info(f"😊 Prediksi IPK: {prediction:.2f} — Cukup Baik")

    else:
        st.warning(f"⚠️ Prediksi IPK: {prediction:.2f} — Perlu Perhatian Lebih")
