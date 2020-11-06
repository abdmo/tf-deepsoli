import os
import sys
import time

import h5py
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Embedding, Flatten, Input, LSTM, Masking, SpatialDropout2D, TimeDistributed
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATASET_DIR = "./dsp"
CH = "ch0"
SEQ_SIZE = 40
BATCH_SIZE = 16
EPOCHS = 50
N_CLASSES = 13

# lr is decreased to a tenth every 20 epochs
def scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 20
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

def load_data(filenames, pad=False):
    workdir = os.getcwd()
    os.chdir(DATASET_DIR)
    X = []
    y = []
    cnt = 0
    for filename in filenames:
        X_seq = []
        with h5py.File(f"{filename}.h5", "r") as h5f:
            print(f"{cnt}: {filename}")
            ch = h5f[CH][()]
            label = h5f["label"][()]
            assert ch.shape[0] == label.shape[0]
            # I kinda cheated here. Ignoring data where the sequences < SEQ_SIZE
            # Proper way to do this is using pad sequences and masking
            # which I'm having trouble to implement
            if ch.shape[0] < SEQ_SIZE:
                continue
            ch = ch[0:SEQ_SIZE]
            ch = ch.reshape(-1, 32, 32, 1)
            y.append(label[0])
            X.append(ch)
            cnt = cnt + 1
            #if cnt == 100:
            #    break
    X = np.array(X)
    if pad:
        X = pad_sequences(X, maxlen=SEQ_SIZE, padding="post", truncating="post")
        print(X.shape)
    y = np.array(y)
    os.chdir(workdir)
    return X, y

def main():
    #tf.debugging.set_log_device_placement(True)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Load data
    with open("file_half.json", "rb") as f:
        filenames = json.load(f)
        train_files = filenames["train"]
        valid_files = filenames["eval"]

    X_train, y_train = load_data(train_files)
    X_valid, y_valid = load_data(valid_files)

    # Randomize data
    indices = np.random.permutation(y_train.shape[0])
    X_train = X_train[indices[:],:]
    y_train = y_train[indices[:],:]
    print(f"X_train.shape={X_train.shape}, X_valid.shape={X_valid.shape}")
    print(f"y_train.shape={y_train.shape}, y_valid.shape={y_valid.shape}")

    y_train = to_categorical(y_train, N_CLASSES)
    y_valid = to_categorical(y_valid, N_CLASSES)

    #X_train = tf.convert_to_tensor(X_train, dtype="float32")
    #X_train = Masking(mask_value=0.0)(X_train)

    # Build model
    model = Sequential()
    # conv1
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation="relu"), input_shape=(SEQ_SIZE, 32, 32, 1)))
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
    #model.add(Embedding(input_dim=512, output_dim=512, mask_zero=True))
    model.add(LSTM(512, dropout=0.5))
    # fc7 - softmax
    model.add(Dense(N_CLASSES, activation="softmax"))
    model.summary()

    # Configure model
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01), metrics=["accuracy"])
    
    # Callbacks
    lr_cb = LearningRateScheduler(scheduler)

    # Train model
    model.fit(X_train, y_train, epochs=EPOCHS, verbose=1, validation_data=(X_valid, y_valid), callbacks=[lr_cb])

    # Save model
    model.save(f"soli_tf1.keras_model_{time.time()}.h5")

if __name__ == "__main__":
    main()

