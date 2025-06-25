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
    st.caption("Versi Aplikasi: 1.0")

# === Judul Aplikasi ===
st.title("🎓 Prediksi IPK Mahasiswa")
st.subheader("📘 Tentang Aplikasi")
st.markdown(
    "Aplikasi ini memprediksi **IPK akhir mahasiswa** berdasarkan nilai angka, "
    "kehadiran, dan jumlah mata kuliah yang diambil menggunakan model **XGBoost hasil tuning Optuna**."
)

# === Input User ===
rata2_nilai = st.slider("Rata-rata Nilai Angka", 0.0, 100.0, 75.0, help="Rentang nilai 0–100.")
rata2_hadir = st.slider("Rata-rata Kehadiran", 0.0, 14.0, 12.0, help="Rata-rata kehadiran tiap mata kuliah (maks. 14 kali).")
jumlah_mk = st.number_input("Jumlah Mata Kuliah Diambil", min_value=1, max_value=20, value=10, step=1)

# === Prediksi IPK ===
if st.button("🎯 Prediksi IPK"):
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

# === Feedback ===
with st.expander("💬 Kirim Masukan untuk Aplikasi"):
    feedback = st.text_area("Masukkan pendapat atau saran kamu:")
    if st.button("Kirim Masukan"):
        st.success("🎉 Terima kasih! Masukan kamu sangat berarti.")
