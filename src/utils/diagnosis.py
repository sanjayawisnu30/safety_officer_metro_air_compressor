import numpy as np
import pandas as pd

# Knowledge Base Diagnosa
DIAGNOSIS_MAP = {
    'TP2': {
        'RENDAH': "Indikasi **KEBOCORAN UDARA (Air Leak)**, Load Valve macet, atau kegagalan kompresi.",
        'TINGGI': "Kemungkinan **Penyumbatan (Blockage)** jalur output atau sensor error.",
        'NORMAL': "Pola tekanan tidak stabil."
    },
    'TP3': {
        'RENDAH': "Tekanan Output drop. Cek kebocoran jaringan distribusi atau Aftercooler.",
        'TINGGI': "Back-pressure tinggi. Cek kran output tertutup atau dryer mampet.",
        'NORMAL': "Fluktuasi tekanan output."
    },
    'Oil_temperature': {
        'RENDAH': "Suhu oli dingin. Cek Thermostatic Bypass Valve.",
        'TINGGI': "⚠️ **OVERHEAT**. Cek Oil Cooler, kipas mati, atau level oli rendah.",
        'NORMAL': "Anomali termal."
    },
    'Motor_current': {
        'RENDAH': "Amper drop. Indikasi **V-Belt Putus** atau kompresor Unloaded.",
        'TINGGI': "⚠️ **OVERLOAD**. Cek Bearing macet atau tegangan drop.",
        'NORMAL': "Gangguan kelistrikan motor."
    },
    'Efficiency_Index': {
        'RENDAH': "⚠️ **INEFISIENSI**. Energi terbuang tapi tekanan tidak dihasilkan (Bocor/Aus).",
        'TINGGI': "Anomali Kalkulasi (False Positive).",
        'NORMAL': "Rasio Input/Output tidak wajar."
    }
    # ... (Tambahkan fitur lain jika perlu)
}

def analyze_direction(actual, predicted, threshold=0.05):
    diff = actual - predicted
    if diff > threshold: return "TINGGI"
    elif diff < -threshold: return "RENDAH"
    else: return "NORMAL"

def generate_report(input_seq, reconstruction, feature_names, threshold_critical, threshold_warning):
    # 1. Hitung Error
    mae_per_feature = np.mean(np.abs(reconstruction - input_seq), axis=1)[0]
    risk_score = np.mean(mae_per_feature)
    
    # 2. Tentukan Status
    if risk_score > threshold_critical:
        status = "🔴 BAHAYA (CRITICAL)"
        severity = 2
    elif risk_score > threshold_warning:
        status = "🟡 WARNING (PERINGATAN)"
        severity = 1
    else:
        status = "🟢 AMAN (NORMAL)"
        severity = 0

    # 3. Analisis Penyebab (Hanya jika Warning/Bahaya)
    analysis = []
    top_features = []
    
    if severity > 0:
        avg_actual = np.mean(input_seq, axis=1)[0]
        avg_pred = np.mean(reconstruction, axis=1)[0]
        
        # Urutkan error terbesar
        contribution = pd.DataFrame({'Feature': feature_names, 'Error': mae_per_feature})
        contribution = contribution.sort_values(by='Error', ascending=False)
        
        # Ambil Top 2
        top_1 = contribution.iloc[0]['Feature']
        idx_1 = feature_names.index(top_1)
        
        direction = analyze_direction(avg_actual[idx_1], avg_pred[idx_1])
        explanation = DIAGNOSIS_MAP.get(top_1, {}).get(direction, "Anomali pola sensor.")
        
        analysis.append(f"Penyebab Utama: {top_1} ({direction})")
        analysis.append(f"Analisis: {explanation}")
        
        # Simpan detail untuk JSON response
        top_features = contribution.head(3).to_dict(orient='records')

    return {
        "status": status,
        "risk_score": float(risk_score),
        "severity_level": severity,
        "analysis_text": "\n".join(analysis) if analysis else "Sistem Beroperasi Normal.",
        "top_contributing_features": top_features
    }