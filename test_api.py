import requests
import pandas as pd
import json

# Setting manual skenario
SKENARIO = 'aman' 

print(f"Loading dataset untuk skenario: [{SKENARIO.upper()}]...")

# Load data
df = pd.read_csv("data/raw/MetroPT3(AirCompressor).csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Potong Data Sesuai Skenario Historis
if SKENARIO == 'aman':
    # Ambil 45 menit data di bulan Februari
    sample_data = df['2020-02-05 10:00:00':'2020-02-05 10:45:00'].copy()
elif SKENARIO == 'rusak':
    # Ambil 45 menit data tepat di tengah-tengah jadwal kerusakan "Air Leak"
    sample_data = df['2020-04-18 12:00:00':'2020-04-18 12:45:00'].copy()
else:
    print("Skenario tidak dikenal, mengambil data default...")
    sample_data = df.head(2500).copy()

sample_data = sample_data.reset_index()

# Format ke JSON
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

# Send request
url = "http://127.0.0.1:8000/predict"
print(f"🚀 Mengirim {len(payload)} baris data sensor ke API...")

try:
    response = requests.post(url, json={"readings": payload})
    
    if response.status_code == 200:
        result = response.json()
        print(f"HASIL DIAGNOSA AI SAFETY OFFICER")
        print(f"Status       : {result['status']}")
        print(f"Risk Score   : {result['risk_score']:.4f}")
        print(f"Tingkat Bahaya: Level {result['severity_level']}")
        
        print(f"\nAnalisis Chatbot:")
        print(result['analysis_text'])
    else:
        print(f"\nAPI MENGEMBALIKAN ERROR ({response.status_code}):")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("ERROR KONEKSI: API belum menyala!")
    print("Pastikan Anda sudah menjalankan perintah: uvicorn api.main:app")