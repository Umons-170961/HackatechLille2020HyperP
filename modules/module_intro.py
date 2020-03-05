#!/usr/bin/python3

import csv
import time
import os
import math
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

# Disabling the warnings about AVX/FMA CPU extension for Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#------------------------------------#
#------------DATA LOADING------------#
#------------------------------------#
def load_csv_with_header(filename, features_dtype, labels_dtype):
    """From https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/base.py"""
    
    with open(filename, 'r') as csv_file:
        # Counting the number of lines of the file.
        reader = csv.reader(csv_file, delimiter=' ')
        n_samples = sum(1 for line in reader) - 1
        csv_file.seek(0)
        
        # First line of the file contains the number of features and the number of labels.
        line = next(reader)
        n_features = int(line[0])
        n_labels = int(line[1])
        
        # Following lines contain (features, labels) samples.
        features = np.zeros((n_samples, n_features), dtype=features_dtype)
        labels = np.zeros((n_samples, n_labels), dtype=labels_dtype)
        for i, line in enumerate(reader):
            labels[i,0:n_labels] = np.asarray(line[n_features:], dtype=labels_dtype)
            features[i] = np.asarray(line[0:n_features], dtype=features_dtype)

        # Training data is stored into a dictionary.
        dict_data = {}
        dict_data["features"] = features
        dict_data["labels"] = labels
            
    return dict_data

def load_datasets():
    training_filename = "./datasets/training_set.csv"
    training_set = load_csv_with_header(filename=training_filename, features_dtype=np.float, labels_dtype=np.float)
    x_train = training_set["features"]
    y_train = training_set["labels"]
    y_train = y_train.astype(int)
    y_train = np_utils.to_categorical(y_train)

    test_filename = "./datasets/test_set.csv"
    test_set = load_csv_with_header(filename=test_filename, features_dtype=np.float, labels_dtype=np.float)
    x_test = test_set["features"]
    y_test = test_set["labels"]
    y_test = y_test.astype(int)
    y_test = np_utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test


#----------------------------------------------------------#
#----------------------ANN DEFINITION----------------------#
#----------------------------------------------------------#
def create_model(nb_unit, nb_hidden_layer, nb_feature, hidden_activation, dropout_prob, weight_decay_factor):
    model = Sequential()
    for i in range(0,nb_hidden_layer):
        model.add(Dense(units=nb_unit, activation=hidden_activation, input_dim=nb_feature, kernel_initializer='zeros', kernel_regularizer=l2(weight_decay_factor)))
        model.add(Dropout(dropout_prob))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


#--------------------------------------------------------#
#----------------------ANN TRAINING----------------------#
#--------------------------------------------------------#
def train_ANN(model, x_train, y_train, x_test, y_test, batch_size):
    t_start = time.time()
    my_histo = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=1000, verbose=0, validation_data=(x_test, y_test), shuffle=False, callbacks=[EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=50, verbose=0, mode='min', baseline=None, restore_best_weights=True)])
    t_end = time.time()
    training_mse = model.evaluate(x_train, y_train, verbose=0)
    test_mse = model.evaluate(x_test, y_test, verbose=0)
    return model, (t_end-t_start), training_mse, test_mse
