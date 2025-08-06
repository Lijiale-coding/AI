from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import numpy as np
import joblib

app = FastAPI()
modele = joblib.load("modele_xgb.joblib")

class PredictInput(BaseModel):
    ticker: str = "TSLA"
    jours: int = 10   # 取最近多少天数据

@app.post("/predict_auto")
def predict_auto(data: PredictInput):
    df = yf.download(data.ticker, period=f"{data.jours}d")
    sma_5 = df["Close"].rolling(window=5).mean().iloc[-1]
    sma_10 = df["Close"].rolling(window=10).mean().iloc[-1]
    X = np.array([[sma_5, sma_10]])
    pred = float(modele.predict(X)[0])
    return {"prediction": round(pred, 2), "sma_5": round(sma_5,2), "sma_10": round(sma_10,2)}
