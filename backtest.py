# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 01:33:58 2025

@author: SETUP GAME
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 1. Charger le fichier CSV
Gold = pd.read_csv("XAU_USD Historical Data (1).csv")

# 2. Nettoyage des colonnes inutiles
Gold.drop(columns=["Vol.", "Change %"], inplace=True, errors='ignore')

# 3. Nettoyer les noms de colonnes (en cas d’espaces)
Gold.columns = Gold.columns.str.strip()

# 4. Nettoyer les colonnes numériques
for col in ["Open", "High", "Low", "Price"]:
    Gold[col] = Gold[col].astype(str).str.replace(",", "").astype(float)

# 5. Renommer 'Price' en 'Close'
Gold.rename(columns={"Price": "Close"}, inplace=True)

# 6. Convertir la colonne Date en datetime et la mettre en index
Gold["Date"] = pd.to_datetime(Gold["Date"], format="%m/%d/%Y")
Gold.sort_values("Date", inplace=True)
Gold.set_index("Date", inplace=True)

# 7. Calcul des indicateurs techniques
def calculate_indicators(Gold):
    # EMA
    Gold["ema_50"] = Gold["Close"].ewm(span=50, adjust=False).mean()
    Gold["ema_200"] = Gold["Close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta = Gold["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    Gold["rsi"] = 100 - (100 / (1 + rs))

    # OBV (sans volume, on simule avec les prix)
    obv = [0]
    for i in range(1, len(Gold)):
        if Gold["Close"].iloc[i] > Gold["Close"].iloc[i - 1]:
            obv.append(obv[-1] + Gold["Close"].iloc[i])
        elif Gold["Close"].iloc[i] < Gold["Close"].iloc[i - 1]:
            obv.append(obv[-1] - Gold["Close"].iloc[i])
        else:
            obv.append(obv[-1])
    Gold["obv"] = obv

    # ATR
    high_low = Gold["High"] - Gold["Low"]
    high_close = abs(Gold["High"] - Gold["Close"].shift())
    low_close = abs(Gold["Low"] - Gold["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    Gold["atr"] = tr.rolling(14, min_periods=1).mean()

    return Gold

# Appliquer la fonction
Gold = calculate_indicators(Gold)

# Aperçu final
print(Gold.tail())

#backtest : 
 # Paramètres
capital = 100000
risk_pct = 0.005   # SL : 0.3%
reward_pct = 0.01  # TP : 0.8%

# Seuils RSI
rsi_min = 48
rsi_max = 62

trades = []
equity_curve = [capital]
equity_dates = []

for i in range(15, len(Gold)):
    row = Gold.iloc[i]

    # Conditions de filtre ATR
    atr_filter = row["atr"] < Gold["atr"].rolling(30).mean().iloc[i]

    # Divergences haussières
    bullish_divergence = (
        row["Low"] < Gold["Low"].iloc[i - 14:i].min() and
        row["rsi"] > Gold["rsi"].iloc[i - 14:i].min() and
        row["rsi"] < rsi_min
    )
    bullish_confirmation = row["ema_50"] > row["ema_200"]
    obv_trend_up = row["obv"] > Gold["obv"].iloc[i - 2]

    # Divergences baissières
    bearish_divergence = (
        row["High"] > Gold["High"].iloc[i - 14:i].max() and
        row["rsi"] < Gold["rsi"].iloc[i - 14:i].max() and
        row["rsi"] > rsi_max
    )
    bearish_confirmation = row["ema_50"] < row["ema_200"]
    obv_trend_down = row["obv"] < Gold["obv"].iloc[i - 2]

    if atr_filter:

        ### ✅ SIGNAL ACHAT ###
        if bullish_divergence and bullish_confirmation and obv_trend_up:
            entry = row["Close"]
            tp = entry * (1 + reward_pct)  # Take Profit
            sl = entry * (1 - risk_pct)    # Stop Loss

            # Rechercher le SL ou TP dans les 10 jours suivants
            exit_price = row["Close"]
            for j in range(i+1, min(i+10, len(Gold))):
                future_row = Gold.iloc[j]
                if future_row["High"] >= tp:
                    exit_price = tp
                    break
                elif future_row["Low"] <= sl:
                    exit_price = sl
                    break
                else:
                    exit_price = future_row["Close"]

            change_pct = (exit_price - entry) / entry
            pnl = capital * change_pct
            capital += pnl

            trades.append({
                "Date": row.name,
                "Type": "long",
                "Entry": entry,
                "Exit": exit_price,
                "PnL": pnl,
                "Return": round(change_pct * 100, 2)
            })

        ### ✅ SIGNAL VENTE ###
        elif bearish_divergence and bearish_confirmation and obv_trend_down:
            entry = row["Close"]
            tp = entry * (1 - reward_pct)  # Take Profit
            sl = entry * (1 + risk_pct)    # Stop Loss

            exit_price = row["Close"]
            for j in range(i+1, min(i+10, len(Gold))):
                future_row = Gold.iloc[j]
                if future_row["Low"] <= tp:
                    exit_price = tp
                    break
                elif future_row["High"] >= sl:
                    exit_price = sl
                    break
                else:
                    exit_price = future_row["Close"]

            change_pct = (entry - exit_price) / entry
            pnl = capital * change_pct
            capital += pnl

            trades.append({
                "Date": row.name,
                "Type": "short",
                "Entry": entry,
                "Exit": exit_price,
                "PnL": pnl,
                "Return": round(change_pct * 100, 2)
            })

    equity_curve.append(capital)
    equity_dates.append(row.name)

# Résultats du backtest
results = pd.DataFrame(trades)

# Statistiques
total_trades = len(results)
win_trades = results[results["PnL"] > 0]
lose_trades = results[results["PnL"] < 0]
winrate = len(win_trades) / total_trades * 100 if total_trades > 0 else 0
avg_gain = win_trades["PnL"].mean() if not win_trades.empty else 0
avg_loss = lose_trades["PnL"].mean() if not lose_trades.empty else 0
reward_risk = abs(avg_gain / avg_loss) if avg_loss != 0 else None

print("\n📊 Résultats du Backtest final :")
print(f"Nombre total de trades     : {total_trades}")
print(f"Taux de réussite (winrate) : {winrate:.2f}%")
print(f"Gain moyen                 : {avg_gain:.2f} $")
print(f"Perte moyenne              : {avg_loss:.2f} $")
if reward_risk is not None:
    print(f"Reward/Risk Ratio          : {reward_risk:.2f}")
else:
    print("Reward/Risk Ratio          : N/A (pas de trades gagnants ou perdants)")
print(f"Capital final              : {capital:.2f} $")

print("\n📄 Derniers trades exécutés :")
print(results.tail())

# 📈 Courbe d’équity
plt.figure(figsize=(12, 5))
plt.plot(equity_dates, equity_curve[1:], label="Equity Curve", linewidth=2)
plt.title("📈 Courbe d’Équity - Performance de la stratégie")
plt.xlabel("Date")
plt.ylabel("Capital ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
