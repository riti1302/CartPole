import numpy as np
import os
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Activation
from tensorflow.keras.callbacks import TensorBoard

file_path = 'Data/training_data/saved-500-5000-mean-4-median-4-complete.npy'

def Get_model(input_size):

    model = Sequential()

    model.add(Dense(128,input_shape=(input_size,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.6))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.6))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.6))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.6))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.6))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    opt = tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

    return model

def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data])
    Y = np.array([i[1] for i in training_data])

    name = int(time.time())
    tensorboard = TensorBoard(log_dir='logs/{}'.format(name))
    model = Get_model(input_size = len(X[0]))
    
    model.fit(X, Y, epochs = 5, callbacks = [tensorboard])
    
    return model


def save_model(model):
    if not os.path.exists('Data/model'):
        os.makedirs('Data/model')

    model.save('Data/model/new_model.model')
    print("Model saved")
    return

training_data = np.load(file_path)
save_model(train_model(training_data))