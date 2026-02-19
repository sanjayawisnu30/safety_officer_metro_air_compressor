import requests
import pandas as pd
import json

# ==========================================
# 🎛️ SETTING SKENARIO (PILIH SALAH SATU)
# ==========================================
# Pilih 'aman' untuk melihat respon NORMAL
# Pilih 'rusak' untuk melihat respon BAHAYA (Air Leak)
SKENARIO = 'aman' 
# ==========================================

print(f"📥 Loading dataset untuk skenario: [{SKENARIO.upper()}]...")

# 1. Load data
df = pd.read_csv("data/raw/MetroPT3(AirCompressor).csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# 2. Potong Data Sesuai Skenario Historis
if SKENARIO == 'aman':
    # Ambil 45 menit data di bulan Februari (Periode Sehat)
    sample_data = df['2020-02-05 10:00:00':'2020-02-05 10:45:00'].copy()
elif SKENARIO == 'rusak':
    # Ambil 45 menit data tepat di tengah-tengah jadwal kerusakan "Air Leak"
    # Sesuai Ground Truth: 18 April 2020
    sample_data = df['2020-04-18 12:00:00':'2020-04-18 12:45:00'].copy()
else:
    print("Skenario tidak dikenal, mengambil data default...")
    sample_data = df.head(2500).copy()

# Reset index agar timestamp kembali menjadi kolom
sample_data = sample_data.reset_index()

# 3. Format ke JSON sesuai Schema Pydantic API
payload = []
for _, row in sample_data.iterrows():
    payload.append({
        "timestamp": str(row['timestamp']),
        "TP2": row['TP2'],
        "TP3": row['TP3'],
        "H1": row['H1'],
        "DV_pressure": row['DV_pressure'],
        "Reservoirs": row['Reservoirs'],
        "Oil_temperature": row['Oil_temperature'],
        "Motor_current": row['Motor_current']
    })

# 4. Kirim Request ke Model API Terbaru
url = "http://127.0.0.1:8000/predict"
print(f"🚀 Mengirim {len(payload)} baris data sensor ke API...")

try:
    response = requests.post(url, json={"readings": payload})
    
    if response.status_code == 200:
        result = response.json()
        print("\n=====================================")
        print(f"🤖 HASIL DIAGNOSA AI SAFETY OFFICER")
        print("=====================================")
        print(f"Status       : {result['status']}")
        print(f"Risk Score   : {result['risk_score']:.4f}")
        print(f"Tingkat Bahaya: Level {result['severity_level']}")
        
        print(f"\n📝 Analisis Chatbot:")
        print(result['analysis_text'])
        
        # Jika Anda ingin melihat JSON aslinya, uncomment baris di bawah:
        # print("\n[Raw JSON]")
        # print(json.dumps(result, indent=2))
        print("=====================================")
    else:
        print(f"\n❌ API MENGEMBALIKAN ERROR ({response.status_code}):")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("❌ ERROR KONEKSI: API belum menyala!")
    print("   Pastikan Anda sudah menjalankan perintah: uvicorn api.main:app")