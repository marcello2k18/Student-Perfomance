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

# Input fitur
rata2_nilai = st.slider("Rata-rata Nilai Angka", 50.0, 100.0, 75.0)
rata2_hadir = st.slider("Rata-rata Kehadiran", 0.0, 16.0, 13.0)
jumlah_mk = st.number_input("Jumlah Mata Kuliah Diambil", 1, 20, 10)

# Prediksi
if st.button("Prediksi IPK"):
    features = pd.DataFrame([{
        "rata2_nilai": rata2_nilai,
        "rata2_hadir": rata2_hadir,
        "jumlah_mk": jumlah_mk
    }])

    prediction = model.predict(features)[0]

    if prediction >= 3.7:
        st.balloons()
        st.success(f"🎯 Prediksi IPK: {prediction:.2f} — Sangat Memuaskan!")
        st.markdown("![Senang Banget](https://media.giphy.com/media/xT0BKmtQGLbumr5RCM/giphy.gif)")

    elif prediction >= 3.0:
        st.info(f"✅ Prediksi IPK: {prediction:.2f} — Cukup Baik")
        st.markdown("![Tidak Senang](https://media.giphy.com/media/3o7abAhvUQbXQvlPzW/giphy.gif)")

    else:
        st.warning(f"😢 Prediksi IPK: {prediction:.2f} — Perlu Perhatian")
        st.markdown("![Sedih Menangis](https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif)")
