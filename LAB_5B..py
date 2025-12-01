# Importing modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Loading data (MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
data = x_train
dataset = x_train

# Creating extra labels – parity of digit
y_train_parity = np.array([sum(map(int, str(d))) % 2 for d in y_train])
y_test_parity = np.array([sum(map(int, str(d))) % 2 for d in y_test])

# Plotting sample images (same idea, same output)
plt.figure(figsize=(6, 6))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

# Building the neural network – multi-output
inp = Input(shape=(28, 28))
x = layers.Flatten()(inp)
x = layers.Dense(128, activation="relu")(x)

digit_out = layers.Dense(10, activation="softmax", name="digit_output")(x)
parity_out = layers.Dense(1, activation="sigmoid", name="parity_output")(x)

model = Model(inputs=inp, outputs=[digit_out, parity_out])

model.compile(
    optimizer="adam",
    loss=["sparse_categorical_crossentropy", "binary_crossentropy"],
    metrics=[["accuracy"], ["accuracy"]]
)

# Training the model with Early Stopping
es = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    x_train,
    [y_train, y_train_parity],
    epochs=20,
    validation_split=0.2,
    callbacks=[es],
    verbose=0
)

# Determine early stop epoch
stop_ep = np.argmin(history.history["val_loss"])
print("\nEarly stopped at epoch:", stop_ep + 1)

# Plotting results – Loss with early stop marker
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.axvline(stop_ep, linestyle="--", label="Early Stop")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()
