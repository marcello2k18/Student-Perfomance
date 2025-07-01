import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# === Load Model ===
@st.cache_resource
def load_model():
    model_path = "xgb_optuna_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ File model tidak ditemukan. Pastikan 'xgb_optuna_model.pkl' ada di root folder.")
        st.stop()
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# === Sidebar Branding ===
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/79/Universitas_Multimedia_Nusantara.png", width=150)
    st.markdown("**Made by MBKM Research Team**")
    st.markdown("Model: `XGBoost (Optuna)`")
    st.caption("Versi Aplikasi: 1.1")

# === Judul Aplikasi ===
st.title("🎓 Prediksi Keberhasilan Akademik Mahasiswa")
st.subheader("📘 Tentang Aplikasi")
st.markdown(
    "Aplikasi ini memprediksi **keberhasilan akademik mahasiswa** berdasarkan pola kehadiran, nilai akademik, "
    "dan jumlah mata kuliah yang diambil. Prediksi dilakukan terhadap **IPK akhir mahasiswa** menggunakan model "
    "**XGBoost yang telah dituning dengan Optuna**."
)

# === Input User ===
st.markdown("### 📝 Masukkan Data Mahasiswa")

rata2_nilai = st.number_input("Rata-rata Nilai Angka", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
rata2_hadir = st.number_input("Rata-rata Kehadiran (per mata kuliah)", min_value=0.0, max_value=14.0, value=12.0, step=0.1)
jumlah_mk = st.number_input("Jumlah Mata Kuliah yang Diambil", min_value=1, max_value=60, value=48, step=1)

# === Prediksi IPK ===
if st.button("🎯 Prediksi IPK"):
    features = pd.DataFrame([{
        "rata2_nilai": rata2_nilai,
        "rata2_hadir": rata2_hadir,
        "jumlah_mk_diambil": jumlah_mk
    }])

    prediction = model.predict(features)[0]

    st.markdown("### 🎓 Hasil Prediksi Keberhasilan")

    if prediction >= 3.4:
        st.success(f"✅ Prediksi IPK: {prediction:.2f} — Mahasiswa diprediksi **BERHASIL** secara akademik.")
        st.info("Pola nilai dan kehadiran menunjukkan partisipasi belajar yang konsisten.")
    elif prediction >= 3.0:
        st.warning(f"⚠️ Prediksi IPK: {prediction:.2f} — Mahasiswa **CUKUP BERHASIL**, tetapi masih dapat ditingkatkan.")
        st.info("Tingkat kehadiran dan partisipasi tergolong moderat. Perlu dukungan dan pemantauan lanjutan.")
    else:
        st.error(f"❌ Prediksi IPK: {prediction:.2f} — Mahasiswa **KURANG BERHASIL** secara akademik.")
        st.info("Perlu perhatian lebih terhadap partisipasi, kehadiran, atau beban studi yang terlalu berat.")
