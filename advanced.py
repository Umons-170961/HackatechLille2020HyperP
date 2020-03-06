from modules.module_advanced import *




#-------------------------------------------------------#
#------------READING THE WHOLE MNIST DATASET------------#
#-------------------------------------------------------#
# Number of training samples:
# n_train = 10000 # up to 60 000
n_train=500
# Number of test samples:
# n_test = 2000 # up to 10 000
n_test=500
x_train, y_train, x_test, y_test = initialize_datasets(n_train, n_test)

## Use here your previous ANN & hyperparameter search method.
## have a break during the computations ;-)




#-----------------------------------------------------#
#------------ANN WITH HETEROGENEOUS LAYERS------------#
#-----------------------------------------------------#

# Hyperparameters
nb_unit=[128, 256, 128] # 1st layer made of 128 neurones, 2nd layer made of 256 neurones and 3rd layer made of 128 neurones
hidden_activation=['relu','tanh','relu'] # 1st layer activation: relu, 2nd layer activation: tanh, 3rd layer activation: relu
dropout_prob=[0.05, 0.1, 0.05] # 1st layer dropout probability: 0.05, 2nd layer dropout probability: 0.1, 3rd layer dropout probability: 0.05
batch_size=y_train.shape[0]
weight_decay_factor=0.01

# Creating and training the ANN
model = create_model(nb_unit, x_train.shape[1], hidden_activation, dropout_prob, weight_decay_factor)
model, training_time, training_loss, test_loss, accuracy = train_ANN(model, x_train, y_train, x_test, y_test, batch_size)
print("---------------Results : \n training MSE=", training_loss, "\n test MSE=", test_loss, "\n training time=", training_time, " seconds \n accuracy=", accuracy,"%")
# Use accuracy to compare your results with the literature results
