from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import io
import base64
from xgboost import XGBRegressor

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def predict_tesla(jours_futurs=5, period="1y"):
    data = yf.download('TSLA', period=period)
    df = data[['Close']].reset_index().rename(columns={'Close': 'Clôture', 'Date': 'Date'})
    df['SMA_5'] = df['Clôture'].rolling(window=5).mean()
    df['SMA_10'] = df['Clôture'].rolling(window=10).mean()
    df = df.dropna().copy()
    X = df[['SMA_5', 'SMA_10']]
    y = df['Clôture'].shift(-1).dropna()
    X = X.iloc[:-1]
    df = df.iloc[:-1]
    modele = XGBRegressor(objective='reg:squarederror')
    modele.fit(X, y)
    prix_futurs = []
    clotures_recentes = [float(x) for x in df['Clôture'].values[-10:]]
    for i in range(jours_futurs):
        sma_5 = np.mean(clotures_recentes[-5:])
        sma_10 = np.mean(clotures_recentes[-10:])
        entrees = np.array([[sma_5, sma_10]])
        prediction = float(modele.predict(entrees)[0])
        prix_futurs.append(prediction)
        clotures_recentes.append(prediction)
    dates_futures = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=jours_futurs)
    resultats = pd.DataFrame({
        'Date': dates_futures,
        'Prix_prédit (€)': [round(p, 2) for p in prix_futurs]
    })
    # 推荐建议
    tendance = np.diff(prix_futurs)
    jours_hausse = sum(tendance > 0)
    jours_baisse = sum(tendance < 0)
    if jours_hausse >= 3:
        conseil = "📈 Tendance haussière détectée – Opportunité d'achat à envisager."
    elif jours_baisse >= 3:
        conseil = "📉 Tendance baissière – Il est recommandé d’attendre ou de réduire les positions."
    else:
        conseil = "⚖️ Tendance neutre – Soyez prudent, observez le marché."
    # 画图并转为base64嵌入网页
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'], df['Clôture'], label="Cours de clôture", color='black')
    ax.plot(df['Date'], df['SMA_5'], label='SMA 5 jours', linestyle='--', color='blue')
    ax.plot(df['Date'], df['SMA_10'], label='SMA 10 jours', linestyle='--', color='green')
    ax.plot(resultats['Date'], resultats["Prix_prédit (€)"], label="Prédiction IA", linestyle='-', marker='o', color='orange')
    ax.legend()
    ax.set_title("Prévision du cours Tesla et indicateurs techniques")
    ax.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return resultats, conseil, img_base64


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "resultats": None, "conseil": None, "img_data": None})

@app.post("/", response_class=HTMLResponse)
async def main_post(request: Request, jours_futurs: int = Form(5), period: str = Form("1y")):
    resultats, conseil, img_base64 = predict_tesla(jours_futurs, period)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "resultats": resultats.to_html(index=False),
        "conseil": conseil,
        "img_data": img_base64
    })
