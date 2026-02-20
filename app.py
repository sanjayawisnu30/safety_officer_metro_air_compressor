import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import pdfplumber

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Safety Officer", page_icon="🤖", layout="wide")

API_PREDICT_URL = "http://127.0.0.1:8000/predict"
API_HEALTH_URL = "http://127.0.0.1:8000/health"

# --- 2. FUNGSI PEMBACA FILE ---
@st.cache_data
def load_data(file):
    file_extension = file.name.split('.')[-1].lower()
    try:
        if file_extension == 'csv': df = pd.read_csv(file)
        elif file_extension in ['xls', 'xlsx']: df = pd.read_excel(file)
        elif file_extension == 'pdf':
            with pdfplumber.open(file) as pdf:
                all_tables = [page.extract_table() for page in pdf.pages if page.extract_table()]
                if not all_tables: return None
                df = pd.DataFrame(all_tables[0][1:], columns=all_tables[0][0])
        else: return None
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        numeric_cols = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
        for col in numeric_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna()
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        return None

# --- 3. AMBIL THRESHOLD DARI API ---
@st.cache_data
def get_thresholds():
    try:
        res = requests.get(API_HEALTH_URL).json()
        return res['config']['threshold_critical'], res['config']['threshold_warning']
    except:
        return 0.33, 0.23 # Fallback default

thresh_critical, thresh_warning = get_thresholds()

# --- 4. TAMPILAN UI STREAMLIT ---
st.title("🤖 AI Safety Officer: Predictive Maintenance")
st.markdown("Analisis tren degradasi mesin dan prediksi sisa umur pakai (*Remaining Useful Life*).")

with st.sidebar:
    st.header("📂 Input Data Sensor")
    uploaded_file = st.file_uploader("Upload log sensor (Min. 200 baris)", type=['csv', 'xlsx', 'pdf'])
    st.markdown("---")
    st.info(f"**Threshold Sistem:**\n- 🔴 Critical: {thresh_critical:.4f}\n- 🟡 Warning: {thresh_warning:.4f}")

# --- 5. LOGIKA PREDIKTIF ---
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.subheader("📄 Preview Data")
        st.dataframe(df.tail(5), height=200)
        st.markdown("---")
        
        if st.button("🚀 Mulai Analisis Prediktif (RUL)", type="primary", use_container_width=True):
            
            MIN_ROWS = 200 
            if len(df) < MIN_ROWS:
                st.error(f"Data terlalu pendek ({len(df)} baris). Butuh minimal {MIN_ROWS} baris untuk simulasi tren historis.")
                st.stop()

            progress_text = "Menganalisis tren masa lalu menuju masa kini..."
            my_bar = st.progress(0, text=progress_text)
            
            num_steps = 10
            step_size = (len(df) - MIN_ROWS) // num_steps
            if step_size == 0: step_size = 1
            
            timestamps_history = []
            risk_scores_history = []
            latest_result = None

            for i, end_idx in enumerate(range(MIN_ROWS, len(df) + 1, step_size)):
                chunk = df.iloc[:end_idx]
                
                payload = [{"timestamp": str(r['timestamp']), "TP2": r['TP2'], "TP3": r['TP3'], 
                            "H1": r['H1'], "DV_pressure": r['DV_pressure'], "Reservoirs": r['Reservoirs'], 
                            "Oil_temperature": r['Oil_temperature'], "Motor_current": r['Motor_current']} 
                           for _, r in chunk.iterrows()]
                
                try:
                    res = requests.post(API_PREDICT_URL, json={"readings": payload})
                    if res.status_code == 200:
                        data = res.json()
                        risk_scores_history.append(data['risk_score'])
                        timestamps_history.append(chunk.iloc[-1]['timestamp'])
                        latest_result = data
                except:
                    st.error("Gagal menghubungi API.")
                    st.stop()
                    
                my_bar.progress((i + 1) / (num_steps + 1))
            
            my_bar.empty()

            # =========================================================
            # PERUBAHAN UI DIMULAI DI SINI
            # =========================================================

            # --- 6. HASIL DIAGNOSA AKHIR (Kondisi Saat Ini diletakkan di atas) ---
            if latest_result:
                st.header("🕵️ Kondisi Mesin Saat Ini")
                m1, m2, m3 = st.columns(3)
                m1.metric("Status Akhir", latest_result['status'].split(' ')[1] if len(latest_result['status'].split(' ')) > 1 else latest_result['status'])
                m2.metric("Skor Risiko Saat Ini", f"{latest_result['risk_score']:.4f}")
                m3.metric("Severity Level", f"Level {latest_result['severity_level']}")
                
                # Banner dinamis berdasarkan tingkat keparahan
                if latest_result['severity_level'] == 2:
                    st.error(f"🚨 **ALASAN CRITICAL:** {latest_result['analysis_text']}")
                elif latest_result['severity_level'] == 1:
                    st.warning(f"⚠️ **ALASAN WARNING:** {latest_result['analysis_text']}")
                else:
                    st.success(f"✅ **ANALISIS NORMAL:** {latest_result['analysis_text']}")

            st.markdown("---")

            # --- 7. KALKULASI RUL (EKSTRAPOLASI LINIER) ---
            x_vals = np.arange(len(risk_scores_history))
            slope, intercept = np.polyfit(x_vals, risk_scores_history, 1)
            
            st.header("📈 Analisis Lanjutan: Kurva Degradasi & Prediksi RUL")
            
            rul_text = ""
            latest_score = risk_scores_history[-1]

            if latest_score >= thresh_critical:
                rul_text = "🚨 **MESIN SUDAH BERADA DI ZONA CRITICAL!** Segera matikan unit untuk mencegah kerusakan fatal."
                st.error(rul_text)
            elif latest_score >= thresh_warning:
                rul_text = "🟡 **MESIN DI ZONA WARNING!** Performa menurun, jadwalkan inspeksi."
                if slope > 0: 
                    steps_to_critical = (thresh_critical - intercept) / slope
                    remaining_steps = max(0, steps_to_critical - x_vals[-1])
                    rul_text += f" Tren menunjukkan batas CRITICAL akan tercapai dalam estimasi **{int(remaining_steps * 5)} menit**."
                st.warning(rul_text)
            else:
                if slope > 0.005: 
                    steps_to_critical = (thresh_critical - intercept) / slope
                    remaining_steps = max(0, steps_to_critical - x_vals[-1])
                    rul_text = f"⚠️ **INDIKASI DEGRADASI:** Mesin saat ini aman, namun tren naik. Estimasi menyentuh batas CRITICAL dalam **{int(remaining_steps * 5)} menit**."
                    st.info(rul_text)
                else:
                    rul_text = "🟢 **MESIN STABIL.** Tidak terdeteksi anomali atau tren kerusakan dalam waktu dekat."
                    st.success(rul_text)

            # --- 8. GAMBAR GRAFIK DENGAN PLOTLY ---
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=timestamps_history, y=risk_scores_history, 
                                     mode='lines+markers', name='Risk Score Aktual', line=dict(color='blue', width=3)))

            if slope > 0.005 and risk_scores_history[-1] < thresh_critical:
                future_x = len(risk_scores_history) + 3 
                future_y = slope * future_x + intercept
                
                fig.add_trace(go.Scatter(
                    x=[timestamps_history[-1], timestamps_history[-1] + pd.Timedelta(minutes=15)], 
                    y=[risk_scores_history[-1], future_y], 
                    mode='lines', name='Prediksi Tren (RUL)', line=dict(color='orange', width=3, dash='dash')
                ))

            fig.add_hline(y=thresh_critical, line_dash="solid", line_color="red", annotation_text="CRITICAL", annotation_position="top left")
            fig.add_hline(y=thresh_warning, line_dash="dash", line_color="orange", annotation_text="WARNING", annotation_position="top left")

            fig.update_layout(title="Perjalanan Risk Score vs Threshold", xaxis_title="Waktu", yaxis_title="Risk Score (MAE)", height=400)
            st.plotly_chart(fig, use_container_width=True)