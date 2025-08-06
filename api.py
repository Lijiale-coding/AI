from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# 1. 加载模型
modele = joblib.load("modele_xgb.joblib")

# 2. 定义输入数据结构
class InputData(BaseModel):
    sma_5: float
    sma_10: float

# 3. 路由
@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.sma_5, data.sma_10]])
    pred = float(modele.predict(X)[0])
    return {"prediction": round(pred, 2)}
