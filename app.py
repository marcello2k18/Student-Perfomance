import streamlit as st
import pandas as pd
import os
import pickle

# === Load Model dan Dataset Preprocessed ===
@st.cache_resource
def load_model():
    model_path = "xgb_optuna_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model belum ditemukan. Pastikan file 'xgb_optuna_model.pkl' tersedia.")
        st.stop()
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    data_path = "data_mahasiswa_cleaned.csv"
    if not os.path.exists(data_path):
        st.error("❌ Dataset belum ditemukan. Pastikan file 'data_mahasiswa_cleaned.csv' tersedia.")
        st.stop()
    return pd.read_csv(data_path)

model = load_model()
df = load_data()

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
    "Masukkan **NIM mahasiswa** untuk menampilkan data akademik dan melakukan prediksi IPK menggunakan model "
    "*XGBoost yang telah dituning dengan Optuna*."
)

# === Input NIM ===
st.markdown("### 🔍 Masukkan NIM Mahasiswa")
input_nim = st.text_input("Contoh: 2301234567")

if input_nim:
    try:
        mahasiswa = df[df["NIM"].astype(str) == input_nim].iloc[0]

        # === Tampilkan Data Mahasiswa ===
        st.markdown("### 📄 Data Mahasiswa")
        if 'nama' in df.columns:
            st.write(f"**Nama:** {mahasiswa['nama']}")
        st.write(f"**Rata-rata Nilai:** {mahasiswa['rata2_nilai']:.2f}")
        st.write(f"**Rata-rata Kehadiran:** {mahasiswa['rata2_hadir']:.2f}")
        st.write(f"**Jumlah Mata Kuliah Diambil:** {int(mahasiswa['jumlah_mk_diambil'])}")

        # === Prediksi IPK ===
        st.markdown("### 🎯 Prediksi IPK Mahasiswa")

        fitur = pd.DataFrame([{
            "rata2_nilai": mahasiswa["rata2_nilai"],
            "rata2_hadir": mahasiswa["rata2_hadir"],
            "jumlah_mk_diambil": mahasiswa["jumlah_mk_diambil"]
        }])
        prediksi_ipk = model.predict(fitur)[0]

        if prediksi_ipk >= 3.4:
            st.success(f"✅ Prediksi IPK: {prediksi_ipk:.2f} — Mahasiswa diprediksi **BERHASIL** secara akademik.")
            st.info("Pola nilai dan kehadiran menunjukkan partisipasi belajar yang konsisten.")
        elif prediksi_ipk >= 3.0:
            st.warning(f"⚠️ Prediksi IPK: {prediksi_ipk:.2f} — Mahasiswa **CUKUP BERHASIL**, tetapi masih dapat ditingkatkan.")
            st.info("Tingkat kehadiran dan partisipasi tergolong moderat. Perlu dukungan dan pemantauan lanjutan.")
        else:
            st.error(f"❌ Prediksi IPK: {prediksi_ipk:.2f} — Mahasiswa **KURANG BERHASIL** secara akademik.")
            st.info("Perlu perhatian lebih terhadap partisipasi, kehadiran, atau beban studi yang terlalu berat.")
  except IndexError:
        st.error("❌ NIM tidak ditemukan dalam data.")
