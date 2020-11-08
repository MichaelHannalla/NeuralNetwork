# CSE488 - Computational Intelligence
# Faculty of Engineering - Ain Shams University
# Neural Network Project - MNIST Dataset
# Haidy Sorial Samy, 16P8104 | Michael Samy Hannalla, 16P8202
# Building of a modular neural network and training it using gradient descent optimization

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy 
from scipy.special import softmax 

train_data_raw = np.loadtxt('gdrive/My Drive/mnist_train.csv', 
                        delimiter=",")

test_data_raw = np.loadtxt('gdrive/My Drive/mnist_test.csv', 
                        delimiter=",")

# Adding one to the train_data_raw
a= len(train_data_raw)
c = np.ones((a , 1))   
new_train_data = np.hstack((train_data_raw,c))

# Adding one to the test_data_raw
a= len(test_data_raw)
c = np.ones((a , 1))   
new_test_data = np.hstack((test_data_raw,c))


# converting train and test to numpy array
# and mapping from 0-255 to 0.01 to 0.99
train_imgs = np.asfarray(new_train_data)
train_imgs[:,1:785] = np.asfarray(new_train_data[:, 1:785]) * 0.99 / 255 + 0.01
test_imgs = np.asfarray(new_test_data)
test_imgs[:,1:785] = np.asfarray(new_test_data[:, 1:785]) * 0.99 / 255 + 0.01

# Splitting the label from the train_imags array 
train_imgs_labels, train_imgs_pixels = train_imgs[:,0], train_imgs[:,1:].T 
print("Shape of train_imgs_labels = " + str(np.shape(train_imgs_labels)))
print("Shape of train_imgs_pixels = " + str(np.shape(train_imgs_pixels)))
print(train_imgs_labels)


# Splitting the label from the test_imags array 
test_imgs_labels, test_imgs_pixels = test_imgs[:,0], test_imgs[:,1:].T 
print("Shape of test_imgs_labels = " + str(np.shape(test_imgs_labels)))
print("Shape of test_imgs_pixels = " + str(np.shape(test_imgs_pixels)))
print(test_imgs_labels)

#ONEHOT TRAIN ARRAY
train_labels_onehot = np.zeros((10,60000))
print("Shape of thet train_lables_onehot = " + str(np.shape(train_labels_onehot)))

for idx, value in enumerate(train_imgs_labels):
  train_labels_onehot[int(value),int(idx)] = 1 

print(train_labels_onehot)

#ONEHOT TEST ARRAY
test_labels_onehot = np.zeros((10,10000))
print("Shape of thet test_lables_onehot = " + str(np.shape(test_labels_onehot)))

for idx, value in enumerate(test_imgs_labels):
  test_labels_onehot[int(value),int(idx)] = 1 

print(test_labels_onehot)

#intialization
class Layer:
   def __init__(self,n_inputs,n_outputs,activation):
     self.xavier = math.sqrt(6)/ math.sqrt(n_inputs+n_outputs)
     self.weights = np.random.uniform(-1*self.xavier,self.xavier,[n_outputs,n_inputs])
     self.activation = activation

   def compute_forward_propagation(self,x_input):
     self.x_input = x_input
     self.outnotactivated = self.weights.dot(self.x_input)
     if self.activation=='NULL':
       self.outactivated = self.outnotactivated
     elif self.activation=='tanh':
       self.outactivated = np.tanh(self.outnotactivated)
     elif self.activation=='relu':
       self.outactivated = np.maximum(0, self.outnotactivated)
      

   def compute_back_propagation(self,back_flowing_gradient):
     if self.activation=='NULL':
       self.inner_gradient = back_flowing_gradient
     elif self.activation=='tanh':
       self.inner_gradient = (1-(np.tanh(self.outnotactivated))**2)*back_flowing_gradient
     elif self.activation=='relu':
       self.inner_gradient = (self.outnotactivated>0).astype(np.uint8)*back_flowing_gradient 
     
     self.weight_gradient = np.dot(self.inner_gradient,(self.x_input).T)

     self.input_gradient = np.dot((self.weights).T,self.inner_gradient)


class NeuralNetworks:
  def __init__(self, *args):
    self.layers = []
    for arg in args:
      self.layers.append(Layer(*arg))
      print(arg)

  def predict(self, x_input):
    self.layers[0].compute_forward_propagation(x_input)
    for i,_ in enumerate(self.layers[1:],start=1):
      self.layers[i].compute_forward_propagation(self.layers[i-1].outactivated)
    self.softmax_output = softmax(self.layers[-1].outactivated,axis = 0)
    self.softmax_output = np.where(self.softmax_output<0.000001,0.000001,self.softmax_output)
    return self.softmax_output 


  def accuracy_loss(self, y_label, y_hat):
    self.loss = np.mean(-y_label*np.log(y_hat))
    temp_max_y_label = np.argmax(y_label, axis=0)
    temp_max_y_hat = np.argmax(y_hat, axis=0) 
    temp_success = (temp_max_y_label == temp_max_y_hat).astype(np.uint8)
    self.accuracy = np.mean(temp_success)
    print(self.loss)
    print(self.accuracy*100)
    return self.loss, self.accuracy 


  def train(self, x_input, y_label, learning_rate=0.1, iterations=100):
    self.loss_list = []
    self.accuracy_list = []
    for i in range(iterations):
      self.predict(x_input)
      self.dsce =  (self.softmax_output-y_label) /len(y_label.T)
      self.layers[-1].compute_back_propagation(self.dsce)
      for i in reversed(range(len(self.layers)-1)):
        self.layers[i].compute_back_propagation(self.layers[i+1].input_gradient)
      
      for i in range(len(self.layers)):
        self.layers[i].weights -= learning_rate*self.layers[i].weight_gradient
      
      self.loss_list.append(self.accuracy_loss(y_label, self.predict(x_input))[0])
      self.accuracy_list.append(self.accuracy_loss(y_label, self.predict(x_input))[1])

      
HeidiNet = NeuralNetworks((785,74,'relu'),(74,74,'relu'),(74,74,'relu'),(74,10,'NULL'))
HeidiNet.train(train_imgs_pixels, train_labels_onehot, 0.1, 100)
print("Accuracy of testing dataset")
HeidiNet.accuracy_loss(test_labels_onehot,HeidiNet.predict(test_imgs_pixels))

# Plotting data
plt.plot(range(len(HeidiNet.loss_list)),HeidiNet.loss_list)
plt.xlabel("Iterations")
plt.ylabel("CE Loss")
plt.figure()
plt.plot(range(len(HeidiNet.accuracy_list)),HeidiNet.accuracy_list)
plt.xlabel("Iterations")
plt.ylabel("Training Accuracy")
plt.show()