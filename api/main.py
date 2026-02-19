import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Import disesuaikan
from api.schemas import PredictionRequest, PredictionResponse
from src.inference import detector 
from src.models.train import run_training

app = FastAPI(
    title="MetroPT-3 AI Safety Officer",
    description="API untuk deteksi anomali pada Air Compressor Unit.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "AI Safety Officer is Online! 🟢", "docs": "/docs"}

@app.get("/health")
def health_check():
    if detector.model is None or detector.scaler is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    return {"status": "healthy", "model_loaded": True, "config": detector.config}

@app.post("/predict", response_model=PredictionResponse)
def predict_anomaly(payload: PredictionRequest):
    try:
        data = [item.dict() for item in payload.readings]
        df_input = pd.DataFrame(data)
        df_input['timestamp'] = pd.to_datetime(df_input['timestamp'])
        df_input.set_index('timestamp', inplace=True)
        
        result = detector.predict(df_input)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/train")
def trigger_training(background_tasks: BackgroundTasks):
    """Endpoint untuk me-retrain model secara asinkron di background."""
    background_tasks.add_task(run_training)
    return {
        "status": "Success",
        "message": "Proses training model telah dimulai di background. Cek terminal untuk progress."
    }

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)