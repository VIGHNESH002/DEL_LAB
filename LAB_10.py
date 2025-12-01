# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------
# Loading data
# -----------------------------
ticker = "AAPL"
data = yf.download(ticker, start="2010-01-01", end="2023-01-01")[["Open"]]
data = data.sort_index()
dataset = data.values

# Splitting into train/test
train_len = int(np.ceil(len(dataset) * 0.8))

# -----------------------------
# Scaling / preprocessing data
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)

# -----------------------------
# Creating sequences
# -----------------------------
def create_sequences(series, steps=60):
    x, y = [], []
    for i in range(steps, len(series)):
        x.append(series[i - steps:i, 0])
        y.append(series[i, 0])
    return np.array(x), np.array(y)

# Train sequences
train_scaled = scaled[:train_len]
x_train, y_train = create_sequences(train_scaled, steps=60)

# Test sequences
test_scaled = scaled[train_len - 60:]
x_test, y_test_scaled = create_sequences(test_scaled, steps=60)

# Ground-truth test values in original scale
y_test = dataset[train_len:]

# -----------------------------
# Reshaping
# -----------------------------
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# -----------------------------
# Building the neural network (LSTM)
# -----------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# -----------------------------
# Training the model
# -----------------------------
model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=1)

# -----------------------------
# Making predictions
# -----------------------------
pred_scaled = model.predict(x_test)

# -----------------------------
# Inverse scaling
# -----------------------------
pred = scaler.inverse_transform(pred_scaled)

# -----------------------------
# Calculating metrics (RMSE, MAE)
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")

# -----------------------------
# Plotting predictions
# -----------------------------
valid = data.iloc[train_len:].copy()
valid["Predictions"] = pred

plt.figure(figsize=(12, 6))
plt.plot(data["Open"], label="Train")
plt.plot(valid[["Open", "Predictions"]])
plt.xlabel("Date")
plt.ylabel("Open Price USD ($)")
plt.legend(["Train", "Val", "Predictions"])
plt.title(f"{ticker} – LSTM Stock Price Prediction")
plt.show()

# -----------------------------
# Printing output
# -----------------------------
print(valid)
