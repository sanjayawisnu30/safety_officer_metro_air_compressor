import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import pdfplumber
import io

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AI Safety Officer Dashboard",
    page_icon="🤖",
    layout="wide"
)

# URL API FastAPI Anda
API_URL = "http://127.0.0.1:8000/predict"

# --- 2. FUNGSI PEMBACA FILE ---
def load_data(file):
    """Membaca file CSV, Excel, atau mengekstrak tabel dari PDF."""
    file_extension = file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            df = pd.read_csv(file)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file)
        elif file_extension == 'pdf':
            # Ekstraksi tabel dari PDF (Sangat bergantung pada format asli PDF)
            with pdfplumber.open(file) as pdf:
                all_tables = []
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        all_tables.extend(table)
                
                if not all_tables:
                    st.error("Tidak ada tabel yang terdeteksi di dalam PDF ini.")
                    return None
                
                # Asumsi baris pertama adalah header
                df = pd.DataFrame(all_tables[1:], columns=all_tables[0])
        else:
            st.error("Format file tidak didukung.")
            return None
        
        # Bersihkan kolom timestamp jika ada
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Pastikan kolom angka dibaca sebagai float (Penting untuk hasil dari PDF)
        numeric_cols = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df

    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

# --- 3. TAMPILAN UI STREAMLIT ---
st.title("🤖 AI Safety Officer: Predictive Maintenance")
st.markdown("Unggah log sensor kompresor udara Anda (CSV, Excel, atau PDF) untuk dianalisis oleh AI.")

# Sidebar untuk Upload
with st.sidebar:
    st.header("📂 Input Data Sensor")
    uploaded_file = st.file_uploader(
        "Pilih file log sensor (Min. 2500 baris/data historis yang cukup)", 
        type=['csv', 'xlsx', 'pdf']
    )
    st.markdown("---")
    st.info("💡 **Tips:** Sistem membutuhkan data historis yang cukup (minimal 30 menit setelah di-resample) untuk melihat pola.")

# --- 4. LOGIKA UTAMA ---
if uploaded_file is not None:
    # Load data
    with st.spinner('Membaca file...'):
        df = load_data(uploaded_file)
    
    if df is not None:
        # Tampilkan Raw Data & Grafik
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Tren Sensor Utama (TP2 & TP3)")
            if 'timestamp' in df.columns and 'TP2' in df.columns and 'TP3' in df.columns:
                fig = px.line(df, x='timestamp', y=['TP2', 'TP3'], title="Tekanan Input vs Output")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Grafik tidak dapat ditampilkan. Pastikan kolom 'timestamp', 'TP2', dan 'TP3' ada.")
                
        with col2:
            st.subheader("📄 Preview Data")
            st.dataframe(df.tail(10), height=300)

        st.markdown("---")
        
        # Tombol Analisis
        if st.button("🔍 Jalankan Diagnosa AI", type="primary", use_container_width=True):
            with st.spinner("AI sedang menganalisis pola anomali..."):
                
                # Format Payload ke JSON
                payload = []
                for _, row in df.iterrows():
                    try:
                        payload.append({
                            "timestamp": str(row['timestamp']),
                            "TP2": float(row['TP2']),
                            "TP3": float(row['TP3']),
                            "H1": float(row['H1']),
                            "DV_pressure": float(row['DV_pressure']),
                            "Reservoirs": float(row['Reservoirs']),
                            "Oil_temperature": float(row['Oil_temperature']),
                            "Motor_current": float(row['Motor_current'])
                        })
                    except KeyError:
                        st.error("Kolom sensor tidak lengkap! Pastikan format file sesuai dengan standar.")
                        st.stop()
                
                # Tembak ke API FastAPI
                try:
                    response = requests.post(API_URL, json={"readings": payload})
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # --- TAMPILKAN HASIL DIAGNOSA ---
                        st.header("📋 Hasil Diagnosa AI")
                        
                        # Kotak Metrik
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Status Mesin", result['status'].split(' ')[1] if len(result['status'].split(' ')) > 1 else result['status'])
                        m2.metric("Risk Score", f"{result['risk_score']:.4f}")
                        m3.metric("Severity Level", f"Level {result['severity_level']}")
                        
                        # Banner Status Berwarna
                        if result['severity_level'] == 2:
                            st.error(f"**{result['status']}**")
                        elif result['severity_level'] == 1:
                            st.warning(f"**{result['status']}**")
                        else:
                            st.success(f"**{result['status']}**")
                            
                        # Laporan XAI (Chatbot)
                        st.subheader("🕵️ Analisis Detail")
                        st.info(result['analysis_text'])
                        
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Tidak dapat terhubung ke Sistem Inti (API). Pastikan Uvicorn FastAPI sudah berjalan di background.")