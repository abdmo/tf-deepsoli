import numpy as np
from tensorflow import keras

N_CLASSES = 13

with open("dataset/soli_X_valid.npy", "rb") as f:
    X_valid = np.load(f)
with open("dataset/soli_y_valid.npy", "rb") as f:
    y_valid = np.load(f)

y_valid = keras.utils.to_categorical(y_valid, N_CLASSES)
model = keras.models.load_model("model")

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(X_valid, y_valid, batch_size=128)
print("test loss, test acc:", results)
