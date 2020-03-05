#!/usr/bin/python3
from modules.module_intro import *
from itertools import product

#------------------------------------------------------------------------------#
#----------------------SERIAL HYPERPARAMETERS GRID SEARCH----------------------#
#------------------------------------------------------------------------------#

x_train, y_train, x_test, y_test = load_datasets()

# Hyperparameters:
# Here we define the different values that each hyperparameter might take

nb_units=[128]
nb_hidden_layers=[1]
hidden_activations=['relu','tanh']
dropout_probs=[0.05]
batch_sizes=[y_train.shape[0]]
weight_decay_factors=[0.01]

# nb_units=[128, 256]
# nb_hidden_layers=[1, 2]
# hidden_activations=['relu', 'tanh']
# dropout_probs=[0.05, 0.1]
# batch_sizes=[y_train.shape[0], int(y_train.shape[0]/2)]
# weight_decay_factors=[0.01, 0.05]


# Computing all possible configurations
configs = product(nb_units, nb_hidden_layers, hidden_activations, dropout_probs, batch_sizes, weight_decay_factors)

best_loss=math.inf
best_config=[]

# Grid Search loop
for config in configs :
    nb_unit, nb_hidden_layer, hidden_activation, dropout_prob, batch_size, weight_decay_factor = config
    print("_________________\n Testing config : \n number of units=",nb_unit,"\n number of hidden layers=", nb_hidden_layer,"\n activation function=",hidden_activation,"\n dropout probability=",dropout_prob,"\n batch size=",batch_size,"\n weight decay factor=",weight_decay_factor)

    #Creating and training the Neural network with the config
    model = create_model(nb_unit, nb_hidden_layer, x_train.shape[1], hidden_activation, dropout_prob, weight_decay_factor)
    model, training_time, training_loss, test_loss = train_ANN(model, x_train, y_train, x_test, y_test, batch_size)
    print("Results : ", test_loss, " in ", training_time, " seconds ")
    # Saving if better result than before
    if test_loss<best_loss:
        print("===== NEW BEST:-) =====")
        best_loss=test_loss
        best_config=config

print("________________________")
print("Final results : ")
print("Best MSE = ",best_loss)
print("with config ",best_config)
