
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()
data = x_train
dataset = x_train


x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train[..., None]
x_test = x_test[..., None]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential([
    Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Dropout(0.25),

    Conv2D(64, 3, activation="relu"),
    MaxPooling2D(),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=128,
    verbose=2
)


pred_probs = model.predict(x_test)
pred = tf.argmax(pred_probs, axis=1)


test_acc = model.evaluate(x_test, y_test, verbose=0)[1]
print("Test Accuracy:", test_acc)


plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()


for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    true_label = tf.argmax(y_test[i]).numpy()
    plt.title(f"P:{pred[i].numpy()}, T:{true_label}")
    plt.axis("off")
plt.tight_layout()
plt.show()
