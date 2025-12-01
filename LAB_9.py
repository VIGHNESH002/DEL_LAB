
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout


ticker = "AAPL"
data = yf.download(ticker, start="2010-01-01", end="2023-01-01")[["Open"]]
data = data.sort_index()
dataset = data.values


train_len = int(np.ceil(len(dataset) * 0.8))


scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)


def create_sequences(series, steps=60):
    x, y = [], []
    for i in range(steps, len(series)):
        x.append(series[i - steps:i, 0])
        y.append(series[i, 0])
    return np.array(x), np.array(y)


train_scaled = scaled[:train_len]
x_train, y_train = create_sequences(train_scaled, steps=60)

test_scaled = scaled[train_len - 60:]
x_test, y_test_scaled = create_sequences(test_scaled, steps=60)


y_test = dataset[train_len:]


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))


model = Sequential([
    GRU(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    GRU(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")


model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=1)


pred_scaled = model.predict(x_test)


pred = scaler.inverse_transform(pred_scaled)


rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")


valid = data.iloc[train_len:].copy()
valid["Predictions"] = pred

plt.figure(figsize=(12, 6))
plt.plot(data["Open"], label="Train")
plt.plot(valid[["Open", "Predictions"]])
plt.xlabel("Date")
plt.ylabel("Open Price USD ($)")
plt.legend(["Train", "Val", "Predictions"])
plt.title(f"{ticker} – GRU Stock Price Prediction")
plt.show()

# -----------------------------
# Printing output
# -----------------------------
print(valid)
