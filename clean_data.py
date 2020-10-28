import os
import sys
import glob
from random import randrange

import h5py
import json
import numpy as np

DATASET_DIR = "./dsp"
SEQ_SIZE = 40

def __load_data(filenames):
    X = []
    y = []
    cnt = 0
    for filename in filenames:
        X_seq = []
        with h5py.File(f"{filename}.h5", "r") as h5f:
            print(f"{cnt}: {filename}")
            ch0 = h5f["ch0"][()]
            ch1 = h5f["ch1"][()]
            ch2 = h5f["ch2"][()]
            label = h5f["label"][()]
            assert ch0.shape[0] == ch1.shape[0] == ch2.shape[0] == label.shape[0]
            #print(f"ch0={ch0.shape}, ch1={ch1.shape}, ch2={ch2.shape}, label={label.shape}")
            # Randomly sample SEQ_SIZE frames from the sequence
            if label.shape[0] < SEQ_SIZE:
                continue
            diff = label.shape[0] - SEQ_SIZE
            if diff > 0:
                start_idx = randrange(diff + 1)
            else:
                start_idx = 0
            ch0 = ch0[start_idx:start_idx + SEQ_SIZE]
            ch1 = ch1[start_idx:start_idx + SEQ_SIZE]
            ch2 = ch2[start_idx:start_idx + SEQ_SIZE]

            #print(f"ch0={ch0.shape}, ch1={ch1.shape}, ch2={ch2.shape}, label={label.shape}")
            assert ch0.shape[0] == ch1.shape[0] == ch2.shape[0] == SEQ_SIZE

            y.append(label[0])
            for i in range(SEQ_SIZE):
                X_seq.append([ch0[i], ch1[i], ch2[i]])
        
            X.append(X_seq)
            cnt = cnt + 1
            #if cnt == 50:
            #    break
    X = np.array(X).reshape(-1, SEQ_SIZE, 3, 32, 32)
    print(X.shape)
    X = np.moveaxis(X, 2, 4)
    print(X.shape)
    y = np.array(y)
    print(y.shape)
    assert X.shape[0] == y.shape[0]
    return X, y

def load_data():
    log = open("readme.txt", "w")

    train_files = []
    valid_files = []
    with open("file_half.json", "rb") as f:
        filenames = json.load(f)
        train_files = filenames["train"]
        valid_files = filenames["eval"]

    log.write(f"n train data from json?: {len(train_files)}\n")
    log.write(f"n valid data from json?: {len(valid_files)}\n")

    os.chdir(DATASET_DIR)
    X_train, y_train = __load_data(train_files)
    X_valid, y_valid =__load_data(valid_files)

    log.write(f"n cleaned train data?: {y_train.shape[0]}\n")
    log.write(f"n cleaned valid data?: {y_valid.shape[0]}\n")
    log.close()

    os.chdir("..")
    with open("soli_X_train.npy", "wb") as f:
        np.save(f, X_train)
    with open("soli_X_valid.npy", "wb") as f:
        np.save(f, X_valid)
    with open("soli_y_train.npy", "wb") as f:
        np.save(f, y_train)
    with open("soli_y_valid.npy", "wb") as f:
        np.save(f, y_valid)

load_data()

