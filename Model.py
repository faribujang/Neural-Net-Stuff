import os
import tensorflow as tf

from Dataset_1 import get_data, next_minibatch

import pandas as pd

import numpy as np


#Change Log
#N/A

Script_tag = 'Siamesev1.0.py'

#Load and Save Functions
Save_load = False
Save_save = False
Save_name = 'Version1.0'
Save_dir = 'C:\\Users\\Imam\\Documents\\PROGRAMMING PROJECTS\\LaunchX'
Save_location = Save_dir+Script_tag+Save_name
if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)

#Open Data Set
# trainingData = Dataset_1.dataAssembler(0,10)
# testData = Dataset_1.dataAssembler(10,90)

from sklearn.model_selection import train_test_split

# data, labels = np.arange(10).reshape((5, 2)), range(5)

# data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20, random_state=42) #split data into training/testing sets


# load dataset
dataset = np.loadtxt("C:\\Users\\Imam\\Documents\\PROGRAMMING PROJECTS\\LaunchX\\ESC-50-master\\meta\\esc50.csv", delimiter = ",")

# dataset = get_data()

# split dataset into sets for testing and training (attributes and their classifications)
X = dataset[:,0:1] #from csv, second column to 13th column (last)
Y = dataset[:,3:4] #from csv, first column
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5) #\47

trainingData = x_train, y_train
testData = x_test, y_test

#Training step count varible
training_steps = 1

#Batch Size
batch_size = 256

#Epoch Ammount
epoch_amm =30

#Input Data Params
data_dimension = [None, 784]

#I/O Placeholders 
x1 = tf.placeholder(tf.float32, data_dimension)
x2 = tf.placeholder(tf.float32, data_dimension)
y = tf.placeholder(tf.float32)

#Learning Rate For Network
flat_rate   = False
base_Rate   = .001
decay_steps = 64
decay_rate  = .97
Staircase   = True

#Learning Rate Definition
if False==flat_rate:
    Learning_Rate = tf.train.exponential_decay(base_Rate, training_steps, decay_steps, decay_rate, staircase='Staircase', name='Exp_decay' )
else:
    Learning_Rate = base_Rate

#Convolution Function returns neurons that act on a section of prev. layer
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#Pooling function returns max value in 2 by 2 sections    
def maxpool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#RELU (rectified linear unit)
def relu(x):
    return tf.nn.relu(x,'relu')

#Matrix broadcasting addition    
def add(x, b):
    return tf.add(x,b)

#Weight initializer    
def weight_def(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#Bias intializer   
def bias_def(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#'Main' method, contains the Neural Network    
def convolutional_neural_network(x1, x2):
    weights = {'W_fc':weight_def([784,512]),
               'W_fc2':weight_def([1024,1024]),
               'W_out':weight_def([1024,2]),}
      
    biases = {'B_fc':bias_def([512]),
              'B_fc2':bias_def([1024]),
              'B_out':bias_def([2])}
    
    #Siamese 1
    fc1 = tf.matmul(x1,weights['W_fc'])
    fc1 = add(fc1,biases['B_fc'])
    fc1 = relu(fc1)

    #siamese 2
    fc2 = tf.matmul(x2,weights['W_fc'])
    fc2 = add(fc2,biases['B_fc'])
    fc2 = relu(fc2)
    
    #Conector Op
    fc = tf.concat([fc1, fc2], 1)

    #Fc Layer
    fc = tf.matmul(fc,weights['W_fc2'])
    fc = add(fc,biases['B_fc2'])
    fc = relu(fc)

    #final layer
    output = tf.matmul(fc,weights['W_out'])
    output = add(output,biases['B_out'])
    
    return output

#Trains The neural Network
def train_neural_network(x1,x2):

    training_steps = 0
    #Initiate The Network
    prediction = convolutional_neural_network(x1, x2)
    
    #Define the Cost and Cost function
    #tf.reduce_mean averages the values of a tensor into one value
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction) )

    #Optimizer + Learning_Rate passthrough
    optimizer = tf.train.AdamOptimizer(Learning_Rate).minimize(cost)
    
    #Get Epoch Ammount 
    hm_epochs = epoch_amm
    
    #Save_saver = tf.train.Saver({'W_conv1':weights[W_conv1] ,'W_conv2':weights[W_conv2] ,'W_fc':weights[W_fc] ,'W_out':weights[W_out] ,'B_conv1':biases[B_conv1] ,'B_conv2':biases[B_conv2] ,'B_fc':biases[B_fc] ,'B_out':biases[B_out]})
    Save_saver = tf.train.Saver()
   
    #Starts C++ Training session
    print('Session Started, ', Script_tag)
    with tf.Session() as sess:
    
        #Initiate and Load all Variables
        sess.run(tf.global_variables_initializer())
        if Save_load: 
            Save_saver.restore(sess , Save_location)
            
        #Begin Logs
        summary_writer = tf.summary.FileWriter('/tmp/logs',sess.graph)
        
        #Start Training
        for epoch in range(hm_epochs):
            
            #Holds Data for loss and accuracy functions
            epoch_loss = 0
            acc_total = 0
            for count in range(int(trainingData.num_examples/batch_size)):
     
                #Training code
                training_steps = (training_steps+1)
                epoch_x1, epoch_x2, epoch_y = trainingData.generateSiameseBatch(batch_size)
                count, c = sess.run([optimizer, cost], feed_dict={x1: epoch_x1, x2: epoch_x2, y: epoch_y})
                epoch_loss += c
                
                #Find Accuracy While Training
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                
                print('Epoch', epoch, 'current epoch loss', epoch_loss, 'batch loss', c,'ts',training_steps,'    ', end='\r')
            #Log the loss per epoch
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss,'    ','batch loss', c,'ts',training_steps,'    ' )
            
            ''' acc_total = 0
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            for _ in range(int(trainingData.num_examples/batch_size)):
                test_x1, test_x2, test_y = trainingData.generateSiameseBatch(batch_size)
                acc = accuracy.eval(feed_dict={x1: test_x1,x2: test_x2, y: test_y})
                acc_total += acc
                print('Train Accuracy:',acc_total*batch_size/float(trainingData.num_examples),end='\r')
            print('Epoch', epoch, 'current train set accuracy : ',acc_total*batch_size/float(trainingData.num_examples))
            
            acc_total = 0
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            for _ in range(int(testData.num_examples/batch_size)):
                test_x1, test_x2, test_y = testData.generateSiameseBatch(batch_size)
                acc = accuracy.eval(feed_dict={x1: test_x1,x2: test_x2, y: test_y})
                acc_total += acc
                print('Test Accuracy:',acc_total*batch_size/float(testData.num_examples),end='\r')
            print('Epoch', epoch, 'current test set accuracy : ',acc_total*batch_size/float(trainingData.num_examples))
 '''
        acc_total = 0
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        for _ in range(int(trainingData.num_examples/batch_size)):
            test_x1, test_x2, test_y = trainingData.generateSiameseBatch(batch_size)
            acc = accuracy.eval(feed_dict={x1: test_x1,x2: test_x2, y: test_y})
            acc_total += acc
            print('Train Accuracy:',acc_total*batch_size/float(trainingData.num_examples),end='\r')
        print('Epoch', epoch, 'current train set accuracy : ',acc_total*batch_size/float(trainingData.num_examples))
        
        acc_total = 0
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        for _ in range(int(testData.num_examples/batch_size)):
            test_x1, test_x2, test_y = testData.generateSiameseBatch(batch_size)
            acc = accuracy.eval(feed_dict={x1: test_x1,x2: test_x2, y: test_y})
            acc_total += acc
            print('Test Accuracy:',acc_total*batch_size/float(testData.num_examples),end='\r')  
        print('Epoch', epoch, 'current test set accuracy : ',acc_total*batch_size/float(testData.num_examples))

        print('Batch - ', batch_size)
        print('Epochs - ', epoch_amm)
        print('Learning Rate:')
        print('flat rate', flat_rate )
        print('Base Rate',base_Rate)
        print('Decay Steps',decay_steps)
        print('Decay Rate',decay_rate)
        print('staircase',Staircase)
        if Save_save:
            Save_path = Save_saver.save(sess , Save_location)
            print("Model saved in file: %s" % Save_path)

    sess.close()
train_neural_network(x1,x2)