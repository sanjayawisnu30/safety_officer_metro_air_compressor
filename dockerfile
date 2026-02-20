FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libbz2-dev \
    liblzma-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY src ./src
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]