import pandas as pd
import os
from pathlib import Path

print("⏳ Memulai pembuatan file testing...")

# --- 1. SETUP PATH ---
# Mengarah ke folder root
BASE_DIR = Path(__file__).resolve().parent

# Path ke raw data
RAW_CSV_PATH = BASE_DIR / "data" / "raw" / "MetroPT3(AirCompressor).csv"

# Path ke folder output
OUTPUT_DIR = BASE_DIR / "data" / "test_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Buat foldernya jika belum ada

# --- 2. LOAD DATA MENTAH ---
if not RAW_CSV_PATH.exists():
    print(f"❌ File tidak ditemukan di: {RAW_CSV_PATH}")
    exit()

print("📥 Sedang membaca dataset mentah (ini mungkin memakan waktu sebentar)...")
df = pd.read_csv(RAW_CSV_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Jumlah baris yang diambil (2500 baris = ~41 menit data per detik)
# Cukup untuk resample menjadi 30 menit bersih
NUM_ROWS = 2500 

# --- 3. BUAT SKENARIO 1: AMAN (NORMAL) ---
print("🟢 Membuat test_aman.csv ...")
# Mengambil data dari bulan Februari (Periode Sehat)
df_aman = df['2020-02-05 10:00:00':'2020-02-05 11:00:00'].head(NUM_ROWS).copy()
df_aman.to_csv(OUTPUT_DIR / "test_aman.csv")

# --- 4. BUAT SKENARIO 2: BAHAYA (CRITICAL) ---
print("🔴 Membuat test_bahaya.csv ...")
# Mengambil data tepat di tengah jadwal kerusakan (Air Leak) - 18 April 2020
df_bahaya = df['2020-04-18 12:00:00':'2020-04-18 13:00:00'].head(NUM_ROWS).copy()
df_bahaya.to_csv(OUTPUT_DIR / "test_bahaya.csv")

# --- 5. BUAT SKENARIO 3: WARNING (WASPADA) ---
print("🟡 Membuat test_warning.csv ...")
# Karena jarang ada data "Warning" yang murni stabil di dataset, 
# kita buat secara sintetis dengan mengambil data AMAN, namun kita beri 
# sedikit kebocoran (Tekanan TP2 & TP3 turun 20%).
df_warning = df_aman.copy()
df_warning['TP2'] = df_warning['TP2'] * 0.80 
df_warning['TP3'] = df_warning['TP3'] * 0.80 
df_warning.to_csv(OUTPUT_DIR / "test_warning.csv")

print(f"\n✅ SUKSES! 3 File testing telah berhasil dibuat di folder: {OUTPUT_DIR}")
print("1. data/test_samples/test_aman.csv")
print("2. data/test_samples/test_warning.csv")
print("3. data/test_samples/test_bahaya.csv")