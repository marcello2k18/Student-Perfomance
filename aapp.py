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
st.markdown("Aplikasi ini dapat menampilkan prediksi IPK mahasiswa berbasis NIM, baik satuan maupun massal (via file Excel).")

# === Input Satuan: NIM ===
st.header("🔍 Prediksi IPK Berdasarkan NIM")
input_nim = st.text_input("Masukkan NIM Mahasiswa:")

if input_nim:
    try:
        mahasiswa = df[df["NIM"].astype(str) == input_nim].iloc[0]
        st.subheader("📄 Detail Mahasiswa")
        if 'nama' in df.columns:
            st.write(f"**Nama:** {mahasiswa['nama']}")
        st.write(f"**Rata-rata Nilai:** {mahasiswa['rata2_nilai']:.2f}")
        st.write(f"**Rata-rata Kehadiran:** {mahasiswa['rata2_hadir']:.2f}")
        st.write(f"**Jumlah MK Diambil:** {int(mahasiswa['jumlah_mk_diambil'])}")

        fitur = pd.DataFrame([{
            "rata2_nilai": mahasiswa["rata2_nilai"],
            "rata2_hadir": mahasiswa["rata2_hadir"],
            "jumlah_mk_diambil": mahasiswa["jumlah_mk_diambil"]
        }])
        prediksi_ipk = model.predict(fitur)[0]

        st.subheader("🎯 Prediksi IPK")
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

# === Input Massal: Upload Excel ===
st.header("📥 Prediksi Massal dari File Excel")
uploaded_file = st.file_uploader("Upload file Excel (.xlsx) berisi kolom `NIM`", type=["xlsx", "xls"])

if uploaded_file:
    try:
        uploaded_df = pd.read_excel(uploaded_file)
        if "NIM" not in uploaded_df.columns:
            st.error("❌ Kolom 'NIM' tidak ditemukan dalam file.")
        else:
            result_rows = []

            for nim in uploaded_df["NIM"].astype(str):
                try:
                    mhs = df[df["NIM"].astype(str) == nim].iloc[0]
                    fitur = pd.DataFrame([{
                        "rata2_nilai": mhs["rata2_nilai"],
                        "rata2_hadir": mhs["rata2_hadir"],
                        "jumlah_mk_diambil": mhs["jumlah_mk_diambil"]
                    }])
                    pred_ipk = model.predict(fitur)[0]

                    if pred_ipk >= 3.4:
                        kategori = "✅ BERHASIL"
                        pesan = "Pola nilai dan kehadiran menunjukkan partisipasi belajar yang konsisten."
                    elif pred_ipk >= 3.0:
                        kategori = "⚠️ CUKUP BERHASIL"
                        pesan = "Perlu dukungan dan pemantauan lanjutan."
                    else:
                        kategori = "❌ KURANG BERHASIL"
                        pesan = "Perlu perhatian terhadap partisipasi dan kehadiran."

                    result_rows.append({
                        "NIM": nim,
                        "Nama": mhs["nama"] if "nama" in mhs else "-",
                        "Rata2 Nilai": round(mhs["rata2_nilai"], 2),
                        "Rata2 Kehadiran": round(mhs["rata2_hadir"], 2),
                        "Jumlah MK": int(mhs["jumlah_mk_diambil"]),
                        "Prediksi IPK": round(pred_ipk, 2),
                        "Kategori": kategori,
                        "Keterangan": pesan
                    })
                except:
                    result_rows.append({
                        "NIM": nim,
                        "Nama": "-",
                        "Rata2 Nilai": "-",
                        "Rata2 Kehadiran": "-",
                        "Jumlah MK": "-",
                        "Prediksi IPK": "-",
                        "Kategori": "❌ Tidak ditemukan",
                        "Keterangan": "-"
                    })

            hasil_df = pd.DataFrame(result_rows)
            st.markdown("### 📊 Hasil Prediksi Massal")
            st.dataframe(hasil_df, use_container_width=True)

            # Download link
            csv = hasil_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Hasil sebagai CSV", data=csv, file_name="hasil_prediksi_massal.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")
