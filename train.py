import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import transpose
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, LSTM, SpatialDropout2D, TimeDistributed
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

BATCH_SIZE = 16
EPOCHS = 50
SEQ_SIZE = 40
N_CLASSES = 13

#def load_data():
#    X = None
#    y = None
#    with open(DATA_FILE, "rb") as f:
#        X = np.load(f)
#    with open(LABEL_FILE, "rb") as f:
#        y = np.load(f)
#    return X, y

# lr is decreased to a tenth every 20 epochs
def scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 20
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

def main():
    #tf.debugging.set_log_device_placement(True)
    #tf.keras.backend.set_image_data_format('channels_first')
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    #X, y = load_data()
    #print(f"X.shape={X.shape}, y.shape={y.shape}")

    # Split train/test data 50/50
    #half = int(y.shape[0]/2)
    #indices = np.random.permutation(X.shape[0])
    #tr_idx, val_idx = indices[:half], indices[half:]
    #X_train, X_valid = X[tr_idx,:], X[val_idx,:]
    #print(f"X_train.shape={X_train.shape}, X_valid.shape={X_valid.shape}")
    #y_train, y_valid = y[tr_idx,:], y[val_idx]
    #print(f"y_train.shape={y_train.shape}, y_valid.shape={y_valid.shape}")

    # Load data
    with open("dataset/soli_X_train.npy", "rb") as f:
        X_train = np.load(f)
    with open("dataset/soli_X_valid.npy", "rb") as f:
        X_valid = np.load(f)
    with open("dataset/soli_y_train.npy", "rb") as f:
        y_train = np.load(f)
    with open("dataset/soli_y_valid.npy", "rb") as f:
        y_valid = np.load(f)

    # Randomize data
    indices = np.random.permutation(y_train.shape[0])
    X_train = X_train[indices[:],:]
    y_train = y_train[indices[:],:]
    print(f"X_train.shape={X_train.shape}, X_valid.shape={X_valid.shape}")
    print(f"y_train.shape={y_train.shape}, y_valid.shape={y_valid.shape}")

    y_train = to_categorical(y_train, N_CLASSES)
    y_valid = to_categorical(y_valid, N_CLASSES)

    # Build model
    model = Sequential()
    # conv1
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation="relu"), input_shape=(SEQ_SIZE, 32, 32, 3)))
    model.add(TimeDistributed(BatchNormalization()))
    # conv2
    model.add(TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation="relu")))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(SpatialDropout2D(0.4)))
    # conv3
    model.add(TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation="relu")))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(SpatialDropout2D(0.4)))
    model.add(TimeDistributed(Flatten()))
    # fc4
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # fc5
    model.add(Dense(512, activation="relu"))
    # lstm6
    model.add(LSTM(512, dropout=0.5))
    # fc7 - softmax
    model.add(Dense(N_CLASSES, activation="softmax"))
    model.summary()

    # Configure model
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01), metrics=["accuracy"])
    callback = LearningRateScheduler(scheduler)

    # Train model
    model.fit(X_train, y_train, epochs=EPOCHS, verbose=1, validation_data=(X_valid, y_valid), callbacks=[callback])

    # Save model
    model.save(f"soli_model_{time.time()}")

if __name__ == "__main__":
    main()

