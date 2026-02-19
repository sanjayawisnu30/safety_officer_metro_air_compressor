import pandas as pd
import numpy as np
from src.utils.config import RAW_SENSOR_COLS, FEATURE_COLS

def process_input_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df[RAW_SENSOR_COLS].copy()
    
    if isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean = df_clean.resample('1min').mean().dropna()
    
    df_clean['TP2_grad'] = df_clean['TP2'].diff().fillna(0)
    df_clean['Oil_temp_grad'] = df_clean['Oil_temperature'].diff().fillna(0)
    df_clean['Efficiency_Index'] = df_clean['TP3'] / (df_clean['Motor_current'] + 0.1)
    df_clean['TP3_roll_std'] = df_clean['TP3'].rolling(window=5).std().fillna(0)
    
    return df_clean[FEATURE_COLS].dropna()

def prepare_lstm_sequence(scaled_data: np.ndarray, time_steps: int = 30):
    output = []
    if len(scaled_data) < time_steps:
        raise ValueError(f"Data kurang panjang! Butuh min {time_steps} baris, punya {len(scaled_data)}.")
        
    for i in range(len(scaled_data) - time_steps + 1):
        output.append(scaled_data[i : (i + time_steps)])
    
    return np.array(output)