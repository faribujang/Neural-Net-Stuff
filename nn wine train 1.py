# import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# weights and biases initialize for reproduceable results
np.random.seed(3)

# number of wine classes
classifications = 3

# load dataset
dataset = np.loadtxt("C:\\Users\\Imam\\Documents\\PROGRAMMING PROJECTS\\wine.csv", delimiter = ",")

# split dataset into sets for testing and training (attributes and their classifications)
X = dataset[:,1:14] #from csv, second column to 13th column (last)
Y = dataset[:,0:1] #from csv, first column
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5) #
# test_size is how much of data used to evaluate model (80/20 split is conventional)
# random_state is seating value

# convert output values to one-hot (convert values to vectors that update cost function/weights & biases)
y_train = keras.utils.to_categorical(y_train-1, classifications)
y_test = keras.utils.to_categorical(y_test-1, classifications)

# creating model, multiple layers
model = Sequential()
model.add(Dense(10, input_dim = 13, activation = "relu")) # layer 1, 13 attributes that make up predictive values
model.add(Dense(8, activation = "relu")) # relu maps to value between 0 and 1 after it is multiplied by weights and added to biases
model.add(Dense(8, activation = "relu")) # relu maps to value between 0 and 1 after it is multiplied by weights and added to biases
model.add(Dense(6, activation = "relu")) # Y = X*W + B (W= weight, B = biases)
model.add(Dense(6, activation = "relu")) # Y = X*W + B (W= weight, B = biases)
model.add(Dense(4, activation = "relu")) # Dense means densely connected network, every neuron is connected to every other neuron in next layer
model.add(Dense(4, activation = "relu")) # Dense means densely connected network, every neuron is connected to every other neuron in next layer
model.add(Dense(2, activation = "relu")) # Dense(# of neurons, activation type)
# model.add(Dense(0, activation = "relu"))
model.add(Dense(classifications, activation = "softmax")) #final layer, neurons equal to potential classifications, softmax to predict probability of classification (not binary)

# compile and fit model

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(x_train, y_train, batch_size = 25, epochs = 5000, validation_data = (x_test, y_test))
# loss function says given networks prediction on training set, how well does it match up with prediction, adjust weights and biases
# adam optimizer used to apply gradient descent to better predict in future
# metrics tell how well network evaluates on training set and test set

# batch_size means only update weights after 25 examples 
# 5000 epochs means full pass over training set 
# validation data is how well network is performing based on how well it can predict


# hello = tf.constant('Tensorflow up in this bitch')
# sess = tf.Session()

# print(sess.run(hello))

# print(np.__version__)