# Importing modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Layer

# Loading data (Fashion MNIST, subset)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train[:1000] / 255.0
y_train = y_train[:1000]
x_test = x_test[:200] / 255.0
y_test = y_test[:200]
data = x_train
dataset = x_train

# Building the baseline model
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Creating adversarial examples
def generate_adversarial(x, eps=0.1):
    perturb = np.sign(np.random.randn(*x.shape))
    x_adv = x + eps * perturb
    return np.clip(x_adv, 0, 1)

# TangentProp layer
class TangentProp(Layer):
    def call(self, x):
        return x + tf.random.normal(tf.shape(x), stddev=0.1)

# Training model with adversarial data
model_adv = create_model()
x_adv = generate_adversarial(x_train)
x_mix = np.concatenate([x_train, x_adv], axis=0)
y_mix = np.concatenate([y_train, y_train], axis=0)

history_adv = model_adv.fit(
    x_mix, y_mix,
    epochs=5,
    validation_data=(x_test, y_test),
    verbose=0
)

# Training model with TangentProp
model_tp = create_model()
model_tp.add(TangentProp())
history_tp = model_tp.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test),
    verbose=0
)

# Tangent distance helper
def tangent_distance(a, b):
    return np.linalg.norm(a - b)

# Tangent distance classifier
def tangent_classifier(train_x, train_y, test_x):
    train_flat = train_x.reshape(len(train_x), -1)
    test_flat = test_x.reshape(len(test_x), -1)
    preds = []
    for x in test_flat:
        dists = np.linalg.norm(train_flat - x, axis=1)
        preds.append(train_y[np.argmin(dists)])
    return np.array(preds)

# Evaluating TangentProp model
tp_acc = model_tp.evaluate(x_test, y_test, verbose=0)[1] * 100
print(f"TangentProp Model Accuracy: {tp_acc:.2f}%")

# Evaluating Tangent Distance classifier
td_pred = tangent_classifier(x_train, y_train, x_test)
td_acc = np.mean(td_pred == y_test) * 100
print(f"Tangent Distance Classifier Accuracy: {td_acc:.2f}%")

# Plotting results – Loss comparison
plt.plot(history_adv.history["loss"], "--", label="Adv Training Loss")
plt.plot(history_tp.history["loss"], "--", label="TangentProp Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss: Adversarial vs TangentProp")
plt.show()
