import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import yfinance as yf

# 1. Génération des données simulées

data = yf.download('TSLA', period='1y')  
df = data[['Close']].reset_index().rename(columns={'Close': 'Clôture', 'Date': 'Date'})


df['SMA_5'] = df['Clôture'].rolling(window=5).mean()
df['SMA_10'] = df['Clôture'].rolling(window=10).mean()
df.dropna(inplace=True)

# 2. Préparation des données d'entraînement
X = df[['SMA_5', 'SMA_10']]
y = df['Clôture'].shift(-1).dropna()
X = X.iloc[:-1]
df = df.iloc[:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. Entraînement du modèle
from xgboost import XGBRegressor
modele = XGBRegressor(objective='reg:squarederror')
modele.fit(X_train, y_train)
import joblib
joblib.dump(modele, "modele_xgb.joblib")

# 4. Prédiction des 5 prochains jours
jours_futurs = 5
prix_futurs = []
#clotures_recentes = list(df['Clôture'].values[-10:])
clotures_recentes = [float(x) for x in df['Clôture'].values[-10:]]

for i in range(jours_futurs):
    sma_5 = np.mean(clotures_recentes[-5:])
    sma_10 = np.mean(clotures_recentes[-10:])
    entrees = np.array([[sma_5, sma_10]])
    prediction = float(modele.predict(entrees)[0])   # 强制转为 float
    prix_futurs.append(prediction)
    clotures_recentes.append(prediction)

# 5. Construction du tableau de résultats
dates_futures = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=jours_futurs)
resultats = pd.DataFrame({
    'Date': dates_futures,
    'Prix_prédit (€)': [round(p, 2) for p in prix_futurs]
})
resultats.to_csv("prediction_result.csv", index=False, encoding='utf-8')
# 6. Génération du conseil
tendance = np.diff(prix_futurs)
jours_hausse = sum(tendance > 0)
jours_baisse = sum(tendance < 0)

if jours_hausse >= 3:
    conseil = " Tendance haussière détectée – Opportunité d'achat à envisager."
elif jours_baisse >= 3:
    conseil = " Tendance baissière – Il est recommandé d’attendre ou de réduire les positions."
else:
    conseil = " Tendance neutre – Soyez prudent, observez le marché."

# 7. Affichage
print(" Prévision du prix de l'action Tesla (5 jours) :")
print(resultats.to_string(index=False))
print("\n Conseil automatique :")
print(conseil)

# 8. Visualisation

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Clôture'], label='Historique (Clôture)', color='black')
plt.plot(df['Date'], df['SMA_5'], label='SMA 5 jours', linestyle='--', color='blue')
plt.plot(df['Date'], df['SMA_10'], label='SMA 10 jours', linestyle='--', color='green')
plt.plot(resultats['Date'], resultats['Prix_prédit (€)'], label='Prévision IA', linestyle='-', marker='o', color='orange')

plt.title("Prévision du prix de l'action Tesla avec indicateurs techniques")
plt.xlabel("Date")
plt.ylabel("Prix (€)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
