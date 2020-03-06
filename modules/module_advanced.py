import struct
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers import Dropout, Dense
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import time
import os
import tensorflow as tf

# Disabling the warnings about AVX/FMA CPU extension for Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)




#------------------------------------#
#------------DATA LOADING------------#
#------------------------------------#
train_files = ['datasets/train-images-idx3-ubyte',
               'datasets/train-labels-idx1-ubyte']
test_files = ['datasets/t10k-images-idx3-ubyte',
              'datasets/t10k-labels-idx1-ubyte']


def read(image_file, label_file, size=-1):
    """
    Inspired from loadlocal_mnist function from mlxtend python package
    """
    with open(label_file, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        y = np.fromfile(lbpath, dtype=np.uint8, count=size)
        size = len(y)

    with open(image_file, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        X = np.fromfile(imgpath, dtype=np.uint8, count=size*784)
        X = X.reshape(size, 784)

    print('Dimensions of the extracted file : %s x %s' %
                                (X.shape[0], X.shape[1]))
    print('labels: %s' % np.unique(y))
    distribution = np.bincount(y)/y.shape[0]*100
    print('Class distribution: %s' % np.array_str(distribution, precision=3))

    return X, np_utils.to_categorical(y)


def initialize_datasets(n_train, n_test):
    x_train, y_train = read(*train_files, n_train)
    x_test, y_test = read(*test_files, n_test)
    return [x_train, y_train, x_test, y_test]




#----------------------------------------------------------#
#----------------------ANN DEFINITION----------------------#
#----------------------------------------------------------#
def create_model(nb_unit, nb_feature, hidden_activation, dropout_prob, weight_decay_factor):
    nb_hidden_layer=len(nb_unit)
    if not ((nb_hidden_layer==(len(hidden_activation))) and (nb_hidden_layer==len(dropout_prob))):
        print("Neurons number list and dropout rates list must be of same length")
        print("Neurons number list should be one less than activation list")
        return None

    model = Sequential()
    for i in range(0,nb_hidden_layer):
        model.add(Dense(units=nb_unit[i], activation=hidden_activation[i], input_dim=nb_feature, kernel_initializer='random_uniform', kernel_regularizer=l2(weight_decay_factor)))
        model.add(Dropout(dropout_prob[i]))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model




#--------------------------------------------------------#
#----------------------ANN TRAINING----------------------#
#--------------------------------------------------------#
def train_ANN(model, x_train, y_train, x_test, y_test, batch_size):
    t_start = time.time()
    my_histo = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=1000, verbose=0, validation_data=(x_test, y_test), shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=50, verbose=0, mode='min', baseline=None, restore_best_weights=True)])
    t_end = time.time()
    training_mse = model.evaluate(x_train, y_train, verbose=0)
    test_mse = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)
    y_pred = np_utils.to_categorical( np.argmax(y_pred, axis=1), num_classes=10)
    accuracy = np.sum( np.sum( np.abs(y_pred-y_test), axis=1) == 0 )/y_test.shape[0]
    return model, (t_end-t_start), training_mse, test_mse, accuracy
