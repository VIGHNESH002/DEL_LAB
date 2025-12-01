# Importing modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Loading data
max_words = 10000
maxlen = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
data = x_train
dataset = x_train

# Scaling / preprocessing data – padding sequences
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Building the neural network (LSTM)
model = Sequential([
    Embedding(max_words, 50, input_length=maxlen),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Training the model
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=2
)

# Calculating metrics
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy:", test_acc)

# Simple sentiment prediction helper
word_index = imdb.get_word_index()

def predict_sentiment(text):
    tokens = text.lower().split()
    seq = [word_index.get(w, 0) for w in tokens if word_index.get(w, 0) < max_words]
    seq = pad_sequences([seq], maxlen=maxlen)
    score = model.predict(seq)[0][0]
    return score

print("Positive:", predict_sentiment("This movie was fantastic and beautifully made"))
print("Negative:", predict_sentiment("This movie was boring, slow and a waste of time"))

# Plotting results – Accuracy and Loss
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()
