# Gunakan versi Python yang ringan
FROM python:3.10-slim

# Set direktori kerja di dalam kontainer
WORKDIR /app

# Copy daftar library yang dibutuhkan
COPY requirements.txt .

# Install semua library (tanpa menyimpan cache agar image lebih kecil)
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode proyek ke dalam kontainer
COPY . .