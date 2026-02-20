import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import mlflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
from src.utils.config import MODEL_PATH, SCALER_PATH, CONFIG_PATH, MLFLOW_DB_PATH, RAW_CSV_PATH, FEATURE_COLS, TIME_STEPS
from src.data.preprocessing import process_input_data, prepare_lstm_sequence

# Sesuaikan jadwal berdasarkan pdf
FAILURE_PERIODS = [
    ('2020-04-18 00:00:00', '2020-04-18 23:59:00', 'Air Leak (High Stress)'),
    ('2020-05-29 23:30:00', '2020-05-30 06:00:00', 'Air Leak (High Stress)'),
    ('2020-06-05 10:00:00', '2020-06-07 14:30:00', 'Air Leak (High Stress)'),
    ('2020-07-15 14:30:00', '2020-07-15 19:00:00', 'Air Leak (High Stress)')
]

def apply_failure_labels(df):
    """Memberikan label 1 pada data yang masuk jadwal rusak, 0 untuk aman."""
    df['y_true_manual'] = 0
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    print("   [LABELING] Menerapkan Ground Truth...")
    for start, end, label in FAILURE_PERIODS:
        mask = (df.index >= start) & (df.index <= end)
        df.loc[mask, 'y_true_manual'] = 1
        count = mask.sum()
        if count > 0:
            print(f"{label}: Ditandai {count} baris data.")
    return df

def build_autoencoder(input_shape):
    """Membangun Arsitektur Jaringan Saraf LSTM"""
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu', return_sequences=False),
        RepeatVector(input_shape[0]),
        LSTM(32, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(input_shape[1]))
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def run_training():
    print("\n[TRAINING] Memulai proses training dengan MLflow & Kalibrasi F1")
    
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("MetroPT3_Anomaly_Detection")

    with mlflow.start_run(run_name="LSTM_F1_Optimized_Run"):
        
        # Load data
        if not os.path.exists(RAW_CSV_PATH):
            print(f"[TRAINING] Error: File tidak ditemukan di {RAW_CSV_PATH}")
            return False
            
        print("[TRAINING] Membaca data mentah...")
        df = pd.read_csv(RAW_CSV_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        print("[TRAINING] Memproses Data Sehat (Feb-Mar)...")
        df_train_raw = df['2020-02-01':'2020-03-01']
        df_train_clean = process_input_data(df_train_raw)
        
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(df_train_clean)
        joblib.dump(scaler, SCALER_PATH)
        
        X_train = prepare_lstm_sequence(train_scaled, TIME_STEPS)
        
        epochs = 15
        batch_size = 64
        mlflow.log_params({
            "time_steps": TIME_STEPS, "epochs": epochs, "batch_size": batch_size,
            "validation_split": 0.1, "shuffle": False
        })

        # Train
        print("[TRAINING] Melatih model LSTM...")
        model = build_autoencoder((X_train.shape[1], X_train.shape[2]))
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        history = model.fit(
            X_train, X_train, epochs=epochs, batch_size=batch_size, 
            validation_split=0.1, callbacks=[early_stop], shuffle=False, verbose=1
        )
        model.save(MODEL_PATH)
        mlflow.log_metric("final_train_mae", history.history['loss'][-1])

        # Evaluasi dan Kalibrasi
        print("[EVALUASI] Menyiapkan Data Uji (April-Agustus)...")
        df_test_raw = df['2020-04-01':] 
        df_test_clean = process_input_data(df_test_raw)
        
        test_scaled = scaler.transform(df_test_clean)
        X_test = prepare_lstm_sequence(test_scaled, TIME_STEPS)
        
        test_timestamps = df_test_clean.index[TIME_STEPS - 1:]
        
        print("[EVALUASI] Menghitung Risk Score Data Uji...")
        test_pred = model.predict(X_test, verbose=0)
        test_mae = np.mean(np.abs(test_pred - X_test), axis=1)
        test_risk = np.mean(test_mae, axis=1)
        
        # DataFrame Evaluasi
        eval_df = pd.DataFrame({'Risk_Score': test_risk}, index=test_timestamps)
        eval_df = apply_failure_labels(eval_df)
        
        print("[EVALUASI] Mengoptimasi Threshold dengan F1-Score...")
        precisions, recalls, thresholds = precision_recall_curve(eval_df['y_true_manual'], eval_df['Risk_Score'])
        
        # Cegah pembagian dengan nol
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_idx])
        
        # Hitung metrik akhir menggunakan threshold terbaik
        eval_df['y_pred'] = (eval_df['Risk_Score'] >= best_threshold).astype(int)
        
        final_acc = accuracy_score(eval_df['y_true_manual'], eval_df['y_pred'])
        final_prec = precision_score(eval_df['y_true_manual'], eval_df['y_pred'], zero_division=0)
        final_rec = recall_score(eval_df['y_true_manual'], eval_df['y_pred'], zero_division=0)
        final_f1 = f1_scores[best_idx]
        
        # Simpan
        config_data = {
            'threshold_critical': best_threshold,
            'threshold_warning': best_threshold * 0.7, # Warning diset di 70% dari Critical
            'features': FEATURE_COLS,
            'time_steps': TIME_STEPS
        }
        joblib.dump(config_data, CONFIG_PATH)
        
        # Log Metrics ke MLflow
        mlflow.log_metrics({
            "threshold_critical": best_threshold,
            "threshold_warning": best_threshold * 0.7,
            "eval_accuracy": final_acc,
            "eval_precision": final_prec,
            "eval_recall": final_rec,
            "eval_f1_score": final_f1
        })
        
        mlflow.log_artifact(CONFIG_PATH) 
        mlflow.log_artifact(SCALER_PATH) 
        
        print(f"\n[HASIL FINAL MLOPS]")
        print(f"   - Threshold Critical : {best_threshold:.6f}")
        print(f"   - Accuracy           : {final_acc:.4f}")
        print(f"   - Precision          : {final_prec:.4f}")
        print(f"   - Recall             : {final_rec:.4f}")
        print(f"   - F1-Score           : {final_f1:.4f}")
        print("[TRAINING] Selesai! Threshold dan Metrics telah dicatat di MLflow.")
        
        return True