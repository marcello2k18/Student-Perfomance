import streamlit as st
import pandas as pd
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# === Page Configuration ===
st.set_page_config(
    page_title="Academic Success Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS for Better Styling ===
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Success/Warning/Error cards */
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .danger-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Header styling */
    .big-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# === Load Model dan Dataset ===
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

# === Helper Functions ===
def get_category_and_message(pred_ipk):
    """Menentukan kategori dan pesan berdasarkan prediksi IPK"""
    if pred_ipk >= 3.51:
        return {
            "kategori": "🏆 CUM LAUDE",
            "color": "success",
            "emoji": "🎉",
            "pesan": "Mahasiswa menunjukkan kinerja akademik luar biasa dengan pola nilai dan kehadiran yang sangat konsisten.",
            "rekomendasi": "Pertahankan performa dan jadilah role model bagi mahasiswa lain."
        }
    elif pred_ipk >= 3.01:
        return {
            "kategori": "✅ SANGAT MEMUASKAN",
            "color": "success",
            "emoji": "👏",
            "pesan": "Kinerja akademik sangat baik dengan partisipasi belajar yang konsisten.",
            "rekomendasi": "Pertahankan performa dan tingkatkan keterlibatan di kegiatan akademik."
        }
    elif pred_ipk >= 2.76:
        return {
            "kategori": "⚠️ MEMUASKAN",
            "color": "warning",
            "emoji": "💪",
            "pesan": "Kinerja akademik cukup baik namun masih dapat ditingkatkan.",
            "rekomendasi": "Tingkatkan kehadiran dan partisipasi kelas untuk hasil yang lebih optimal."
        }
    else:
        return {
            "kategori": "❌ PERLU PERHATIAN",
            "color": "error",
            "emoji": "🚨",
            "pesan": "Kinerja akademik memerlukan perhatian khusus dan intervensi segera.",
            "rekomendasi": "Segera konsultasi dengan dosen pembimbing akademik dan manfaatkan program mentoring."
        }

def create_gauge_chart(value, title):
    """Membuat gauge chart untuk visualisasi metrik"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': 3.0, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 4.0], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 2.76], 'color': '#ffcccb'},
                {'range': [2.76, 3.0], 'color': '#ffffcc'},
                {'range': [3.0, 3.51], 'color': '#ccffcc'},
                {'range': [3.51, 4.0], 'color': '#90EE90'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 3.5
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_feature_comparison(mahasiswa):
    """Membuat bar chart perbandingan fitur mahasiswa vs rata-rata"""
    avg_nilai = df['rata2_nilai'].mean()
    avg_hadir = df['rata2_hadir'].mean()
    avg_mk = df['jumlah_mk_diambil'].mean()
    
    fig = go.Figure(data=[
        go.Bar(name='Mahasiswa Ini', x=['Rata-rata Nilai', 'Rata-rata Kehadiran', 'Jumlah MK'],
               y=[mahasiswa['rata2_nilai'], mahasiswa['rata2_hadir'], mahasiswa['jumlah_mk_diambil']],
               marker_color='#667eea'),
        go.Bar(name='Rata-rata Kampus', x=['Rata-rata Nilai', 'Rata-rata Kehadiran', 'Jumlah MK'],
               y=[avg_nilai, avg_hadir, avg_mk],
               marker_color='#764ba2')
    ])
    
    fig.update_layout(
        title='Perbandingan dengan Rata-rata Kampus',
        barmode='group',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkblue"}
    )
    
    return fig

# === Sidebar ===
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/79/Universitas_Multimedia_Nusantara.png", width=150)
    
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    st.info("""
    **Model:** XGBoost (Optuna-Optimized)
    
    **Performance:**
    - R² Score: 0.8637
    - RMSE: 0.1165
    - MAE: 0.0809
    
    **Features Used:**
    - Rata-rata Nilai
    - Rata-rata Kehadiran
    - Jumlah MK Diambil
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Dataset Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Students", f"{len(df):,}")
    with col2:
        st.metric("Avg GPA", f"{df['IPK'].mean():.2f}")
    
    st.markdown("---")
    st.markdown("**Made with ❤️ by**")
    st.markdown("**MBKM Research Team**")
    st.caption(f"Version 2.0 | {datetime.now().strftime('%Y')}")

# === Main Content ===
st.markdown('<p class="big-header">🎓 Sistem Prediksi Keberhasilan Akademik</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Machine Learning - XGBoost Optuna Optimization</p>', unsafe_allow_html=True)

# === Tabs for Different Sections ===
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Individual", "📊 Prediksi Massal", "📈 Dashboard Analytics"])

# === TAB 1: Individual Prediction ===
with tab1:
    st.markdown("### Prediksi IPK Berdasarkan NIM")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_nim = st.text_input("🔢 Masukkan NIM Mahasiswa:", placeholder="e.g., 00000012345")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🚀 Prediksi IPK", use_container_width=True)
    
    if input_nim and predict_button:
        try:
            mahasiswa = df[df["NIM"].astype(str) == input_nim].iloc[0]
            
            # Student Info Section
            st.markdown("---")
            st.markdown("### 👤 Informasi Mahasiswa")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{mahasiswa['rata2_nilai']:.2f}</h3>
                    <p>Rata-rata Nilai</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{mahasiswa['rata2_hadir']:.2f}</h3>
                    <p>Rata-rata Kehadiran</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{int(mahasiswa['jumlah_mk_diambil'])}</h3>
                    <p>Jumlah MK</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                if 'IPK' in mahasiswa:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{mahasiswa['IPK']:.2f}</h3>
                        <p>IPK Aktual</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Prediction Section
            fitur = pd.DataFrame([{
                "rata2_nilai": mahasiswa["rata2_nilai"],
                "rata2_hadir": mahasiswa["rata2_hadir"],
                "jumlah_mk_diambil": mahasiswa["jumlah_mk_diambil"]
            }])
            
            prediksi_ipk = model.predict(fitur)[0]
            result = get_category_and_message(prediksi_ipk)
            
            st.markdown("---")
            st.markdown("### 🎯 Hasil Prediksi")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Gauge chart
                fig_gauge = create_gauge_chart(prediksi_ipk, "Prediksi IPK")
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Result card
                if result['color'] == 'success':
                    card_class = 'success-card'
                elif result['color'] == 'warning':
                    card_class = 'warning-card'
                else:
                    card_class = 'danger-card'
                
                st.markdown(f"""
                <div class="{card_class}">
                    <h2>{result['emoji']} {result['kategori']}</h2>
                    <h3>Prediksi IPK: {prediksi_ipk:.2f}</h3>
                    <p style="margin-top: 1rem;">{result['pesan']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="info-box">
                    <strong>💡 Rekomendasi:</strong><br>
                    {result['rekomendasi']}
                </div>
                """, unsafe_allow_html=True)
            
            # Comparison chart
            st.markdown("---")
            fig_comparison = create_feature_comparison(mahasiswa)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
        except IndexError:
            st.error("❌ NIM tidak ditemukan dalam database.")
            st.info("💡 Pastikan NIM yang dimasukkan sudah terdaftar di sistem.")

# === TAB 2: Batch Prediction ===
with tab2:
    st.markdown("### 📥 Upload File untuk Prediksi Massal")
    
    st.markdown("""
    <div class="info-box">
        <strong>📋 Format File:</strong><br>
        • File harus berformat <code>.csv</code>, <code>.xlsx</code>, atau <code>.xls</code><br>
        • Harus memiliki kolom <code>NIM</code><br>
        • Contoh format:
    </div>
    """, unsafe_allow_html=True)
    
    # Example format
    example_df = pd.DataFrame({
        'NIM': ['00000012345', '00000012346', '00000012347']
    })
    st.dataframe(example_df, use_container_width=True)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📎 Pilih file untuk diupload",
        type=["csv", "xlsx", "xls"],
        help="Upload file berisi daftar NIM mahasiswa"
    )
    
    if uploaded_file:
        try:
            # Read file
            filename = uploaded_file.name.lower()
            if filename.endswith(".csv"):
                uploaded_df = pd.read_csv(uploaded_file)
            elif filename.endswith((".xlsx", ".xls")):
                uploaded_df = pd.read_excel(uploaded_file, engine="openpyxl")
            
            if "NIM" not in uploaded_df.columns:
                st.error("❌ Kolom 'NIM' tidak ditemukan dalam file.")
            else:
                st.success(f"✅ File berhasil diupload! Ditemukan {len(uploaded_df)} NIM.")
                
                if st.button("🚀 Mulai Prediksi Massal", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    result_rows = []
                    total = len(uploaded_df)
                    
                    for idx, nim in enumerate(uploaded_df["NIM"].astype(str)):
                        status_text.text(f"Processing {idx+1}/{total}: NIM {nim}")
                        progress_bar.progress((idx + 1) / total)
                        
                        try:
                            mhs = df[df["NIM"].astype(str) == nim].iloc[0]
                            fitur = pd.DataFrame([{
                                "rata2_nilai": mhs["rata2_nilai"],
                                "rata2_hadir": mhs["rata2_hadir"],
                                "jumlah_mk_diambil": mhs["jumlah_mk_diambil"]
                            }])
                            pred_ipk = model.predict(fitur)[0]
                            result = get_category_and_message(pred_ipk)
                            
                            result_rows.append({
                                "NIM": nim,
                                "Nama": mhs.get("nama", "-"),
                                "Rata2 Nilai": round(mhs["rata2_nilai"], 2),
                                "Rata2 Kehadiran": round(mhs["rata2_hadir"], 2),
                                "Jumlah MK": int(mhs["jumlah_mk_diambil"]),
                                "Prediksi IPK": round(pred_ipk, 2),
                                "Kategori": result['kategori'],
                                "Rekomendasi": result['rekomendasi']
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
                                "Rekomendasi": "Data tidak tersedia"
                            })
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    hasil_df = pd.DataFrame(result_rows)
                    
                    # Summary statistics
                    st.markdown("---")
                    st.markdown("### 📊 Ringkasan Hasil")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    valid_predictions = hasil_df[hasil_df['Prediksi IPK'] != '-']
                    
                    with col1:
                        st.metric("Total Mahasiswa", len(hasil_df))
                    with col2:
                        st.metric("Prediksi Berhasil", len(valid_predictions))
                    with col3:
                        if len(valid_predictions) > 0:
                            avg_pred = valid_predictions['Prediksi IPK'].mean()
                            st.metric("Rata-rata Prediksi IPK", f"{avg_pred:.2f}")
                    with col4:
                        cum_laude = len(valid_predictions[valid_predictions['Prediksi IPK'] >= 3.51])
                        st.metric("Cum Laude", cum_laude)
                    
                    # Distribution chart
                    if len(valid_predictions) > 0:
                        fig_dist = px.histogram(
                            valid_predictions,
                            x='Prediksi IPK',
                            nbins=20,
                            title='Distribusi Prediksi IPK',
                            color_discrete_sequence=['#667eea']
                        )
                        fig_dist.update_layout(
                            xaxis_title='Prediksi IPK',
                            yaxis_title='Jumlah Mahasiswa',
                            height=400
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Results table
                    st.markdown("---")
                    st.markdown("### 📋 Detail Hasil Prediksi")
                    st.dataframe(
                        hasil_df.style.apply(
                            lambda x: ['background-color: #d4edda' if '🏆' in str(v) or '✅' in str(v)
                                      else 'background-color: #fff3cd' if '⚠️' in str(v)
                                      else 'background-color: #f8d7da' if '❌' in str(v)
                                      else '' for v in x],
                            subset=['Kategori']
                        ),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download button
                    csv = hasil_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Hasil (CSV)",
                        data=csv,
                        file_name=f"hasil_prediksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")

# === TAB 3: Dashboard Analytics ===
with tab3:
    st.markdown("### 📈 Dashboard Analitik Dataset")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Mahasiswa", f"{len(df):,}")
    with col2:
        st.metric("Rata-rata IPK", f"{df['IPK'].mean():.2f}")
    with col3:
        st.metric("Std Dev IPK", f"{df['IPK'].std():.2f}")
    
    st.markdown("---")
    
    # GPA Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_gpa_dist = px.histogram(
            df,
            x='IPK',
            nbins=30,
            title='Distribusi IPK Mahasiswa',
            color_discrete_sequence=['#667eea']
        )
        fig_gpa_dist.add_vline(x=df['IPK'].mean(), line_dash="dash", line_color="red",
                              annotation_text="Mean")
        fig_gpa_dist.update_layout(height=400)
        st.plotly_chart(fig_gpa_dist, use_container_width=True)
    
    with col2:
        # GPA Categories
        categories = []
        for ipk in df['IPK']:
            if ipk >= 3.51:
                categories.append('Cum Laude')
            elif ipk >= 3.01:
                categories.append('Sangat Memuaskan')
            elif ipk >= 2.76:
                categories.append('Memuaskan')
            else:
                categories.append('Perlu Perhatian')
        
        df_cat = pd.DataFrame({'Kategori': categories})
        cat_counts = df_cat['Kategori'].value_counts()
        
        fig_pie = px.pie(
            values=cat_counts.values,
            names=cat_counts.index,
            title='Distribusi Kategori Kelulusan',
            color_discrete_sequence=['#11998e', '#38ef7d', '#f093fb', '#f5576c']
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Correlations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scatter1 = px.scatter(
            df,
            x='rata2_nilai',
            y='IPK',
            title='Hubungan Rata-rata Nilai vs IPK',
            trendline="ols",
            color_discrete_sequence=['#667eea']
        )
        fig_scatter1.update_layout(height=400)
        st.plotly_chart(fig_scatter1, use_container_width=True)
    
    with col2:
        fig_scatter2 = px.scatter(
            df,
            x='rata2_hadir',
            y='IPK',
            title='Hubungan Rata-rata Kehadiran vs IPK',
            trendline="ols",
            color_discrete_sequence=['#764ba2']
        )
        fig_scatter2.update_layout(height=400)
        st.plotly_chart(fig_scatter2, use_container_width=True)

# === Footer ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Academic Success Prediction System</strong></p>
    <p>Powered by XGBoost & Optuna | Developed by MBKM Research Team</p>
    <p>© 2024 Universitas Multimedia Nusantara. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
