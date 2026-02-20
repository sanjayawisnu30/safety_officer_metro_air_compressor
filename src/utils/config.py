import os
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
UTILS_DIR = CURRENT_FILE.parent     
SRC_DIR = UTILS_DIR.parent          
ROOT_DIR = SRC_DIR.parent           

MODELS_DIR = SRC_DIR / "models_store"
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"

# Gunakan .as_posix() karena /a pada naming projek saya
MODEL_PATH = (MODELS_DIR / "metropt_lstm_model.h5").as_posix()
SCALER_PATH = (MODELS_DIR / "metropt_scaler.pkl").as_posix()
CONFIG_PATH = (MODELS_DIR / "metropt_config.pkl").as_posix()
MLFLOW_DB_PATH = (MODELS_DIR / "mlflow.db").as_posix() 

# Path ke dataset CSV
RAW_CSV_PATH = (DATA_RAW_DIR / "MetroPT3(AirCompressor).csv").as_posix()

# Konfigurasi kolom
FEATURE_COLS = [
    'TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs',
    'Oil_temperature', 'Motor_current',
    'TP2_grad', 'Oil_temp_grad', 'Efficiency_Index', 'TP3_roll_std'
]

RAW_SENSOR_COLS = [
    'TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 
    'Oil_temperature', 'Motor_current'
]

TIME_STEPS = 30