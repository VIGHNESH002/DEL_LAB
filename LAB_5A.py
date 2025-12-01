# Importing modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Loading data (synthetic classification)
data, labels = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    random_state=42
)
dataset = data

# Splitting into train/test
x_train, x_test, y_train, y_test = train_test_split(
    dataset, labels, test_size=0.2, random_state=42
)

# Scaling / preprocessing data (skipped as in original)

# Building the neural network – Dropout model
def create_dropout_model(dropout_rate=0.2):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(32, activation="relu"),
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid")
    ])
    return model

# Building the neural network – Gradient Clipping model
def create_clipped_model(clip_norm=1.0):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(clipnorm=clip_norm),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Training the Dropout model
dropout_model = create_dropout_model()
dropout_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
history_dropout = dropout_model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_test, y_test),
    verbose=0
)

# Training the Gradient Clipping model
clipped_model = create_clipped_model()
history_clipped = clipped_model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_test, y_test),
    verbose=0
)

# Plotting results – Accuracy
plt.plot(history_dropout.history["accuracy"], "--", label="Dropout Train Acc")
plt.plot(history_dropout.history["val_accuracy"], "--", label="Dropout Val Acc")
plt.plot(history_clipped.history["accuracy"], label="Clipped Train Acc")
plt.plot(history_clipped.history["val_accuracy"], label="Clipped Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Dropout vs Gradient Clipping")
plt.show()
