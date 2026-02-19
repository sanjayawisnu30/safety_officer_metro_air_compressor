import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import disesuaikan dengan tree folder baru
from src.utils.config import MODEL_PATH, SCALER_PATH, CONFIG_PATH, FEATURE_COLS, TIME_STEPS
from src.data.preprocessing import process_input_data, prepare_lstm_sequence
from src.utils.diagnosis import generate_report

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.config = {}
        self.load_artifacts()

    def load_artifacts(self):
        print("⏳ Loading artifacts...")
        try:
            self.config = joblib.load(CONFIG_PATH)
            self.thresh_critical = self.config.get('threshold_critical', 0.33)
            self.thresh_warning = self.config.get('threshold_warning', 0.23)
            
            self.scaler = joblib.load(SCALER_PATH)
            
            try:
                self.model = load_model(MODEL_PATH, custom_objects={'mae': tf.keras.losses.MeanAbsoluteError})
            except:
                self.model = load_model(MODEL_PATH, compile=False)
                
            print("✅ System Loaded Successfully!")
        except Exception as e:
            print(f"❌ Error loading artifacts: {e}")
            raise e

    def predict(self, df_input):
        df_clean = process_input_data(df_input)
        
        if len(df_clean) < TIME_STEPS:
            return {"error": f"Data kurang. Butuh {TIME_STEPS} baris data bersih, punya {len(df_clean)}."}

        X_scaled = self.scaler.transform(df_clean)
        X_seq = np.array([X_scaled[-TIME_STEPS:]]) 
        
        reconstruction = self.model.predict(X_seq, verbose=0)
        
        result = generate_report(
            X_seq, reconstruction, FEATURE_COLS, 
            self.thresh_critical, self.thresh_warning
        )
        return result

# Inisialisasi Singleton
detector = AnomalyDetector()