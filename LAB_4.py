
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


iris = load_iris()
data = iris.data
labels = iris.target
dataset = data


X = data
y = labels

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))


x_train, x_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


def build_model():
    model = Sequential([
        Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax")
    ])
    return model


model_gd = build_model()
model_gd.compile(
    optimizer=SGD(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model_sgd = build_model()
model_sgd.compile(
    optimizer=SGD(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history_gd = model_gd.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_test, y_test),
    verbose=0
)

history_sgd = model_sgd.fit(
    x_train, y_train,
    epochs=50,
    batch_size=1,
    validation_data=(x_test, y_test),
    verbose=0
)


plt.figure()
plt.plot(history_gd.history["loss"], label="GD Train Loss")
plt.plot(history_gd.history["val_loss"], label="GD Val Loss")
plt.plot(history_sgd.history["loss"], label="SGD Train Loss")
plt.plot(history_sgd.history["val_loss"], label="SGD Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Comparison: GD vs SGD")
plt.show()

# Plotting results – Accuracy
plt.figure()
plt.plot(history_gd.history["accuracy"], label="GD Train Acc")
plt.plot(history_gd.history["val_accuracy"], label="GD Val Acc")
plt.plot(history_sgd.history["accuracy"], label="SGD Train Acc")
plt.plot(history_sgd.history["val_accuracy"], label="SGD Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Comparison: GD vs SGD")
plt.show()
