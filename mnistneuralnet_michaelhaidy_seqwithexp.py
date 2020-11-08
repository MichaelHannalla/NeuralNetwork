# CSE488 - Computational Intelligence
# Faculty of Engineering - Ain Shams University
# Neural Network Project - MNIST Dataset
# Haidy Sorial Samy, 16P8104 | Michael Samy Hannalla, 16P8202
# Building of a modular neural network and training it using gradient descent optimization

import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import time
import scipy as sp
from scipy.special import softmax

class NetLayer():
    def __init__(self, weight_shape, activation_fn):
        self.activation_fn = activation_fn
        xavier_init = math.sqrt(6.0) / math.sqrt(weight_shape[0] + weight_shape[1])  # Xavier initialization.
        self.weight = np.random.uniform(low= -xavier_init, high= xavier_init, size=weight_shape).T
        #self.weight = np.random.rand(*weight_shape).T
    
    def feedforward(self, input_x):
        self.input_x = input_x
        self.output = self.weight.dot(self.input_x)
        if self.activation_fn == 'relu':
            self.out_activated = np.where(self.output <= 0, 0, self.output) # ReLU activation function.
        elif self.activation_fn == 'sigmoid':
            self.out_activated = 1 / (1 + np.exp(self.output))              # Sigmoid activation function.
        elif self.activation_fn == 'tanh':
            self.out_activated = np.tanh(self.output)                       # tanh activation function.
        elif self.activation_fn == 'none':
            self.out_activated = self.output
        else:
            sys.exit('Aborting code, unidentified activation function.')

    def backpropagate(self, backflowgradient):
        if self.activation_fn == 'relu':                                    # ReLU local gradient.
            backflowgradient[self.output<=0] = 0                            
        elif self.activation_fn == 'sigmoid':                               # Sigmoid local gradient
            backflowgradient = backflowgradient * self.out_activated * np.subtract(1, self.out_activated)
        elif self.activation_fn == 'tanh':                                  # tanh local gradient
            backflowgradient = backflowgradient * np.subtract(1, np.square(self.out_activated))
        self.weight_grad = backflowgradient.dot(self.input_x.T)
        self.backpropagated_grad = self.weight.T.dot(backflowgradient)

class NeuralNetwork():
    def __init__(self, name, input_neurons, output_neurons, output_activation_fn):
        self.name = name
        self.nodes_list = [input_neurons, output_neurons]
        self.activations_list = [output_activation_fn]
        self.layer_iterator = 1
    
    def add_layer(self, nodes, activation_fn):
        self.nodes_list.insert(self.layer_iterator, nodes)
        self.activations_list.insert(0, activation_fn)
        self.layer_iterator += 1

    def compile_network(self):
        self.weight_shapes = zip(self.nodes_list, self.nodes_list[1:])
        self.layers = []
        temp_i = 0
        for weight_shape in self.weight_shapes:
            self.layers.append(NetLayer(weight_shape, self.activations_list[temp_i]))
            temp_i += 1
        
    def predict(self, input_x):
        self.layers[0].feedforward(input_x)

        for i,_ in enumerate(self.layers[1:], start = 1):
            self.layers[i].feedforward(self.layers[i-1].out_activated)

        self.softmax_out = softmax(self.layers[-1].out_activated, axis = 0)
        self.softmax_out = np.where(self.softmax_out<0.0001, 0.0001, self.softmax_out) # To avoid zero in log(y_hat)
        return self.softmax_out

    def check_accuracy(self, predicted_y, labels_y):
        temp_y = np.argmax(labels_y, axis = 0)
        temp_yhat = np.argmax(predicted_y, axis = 0)
        temp_success = (temp_y == temp_yhat).astype(np.uint8)
        return np.mean(temp_success)

    def lossfunc(self, predicted_y, labels_y): # Categorical CE.
        return np.sum(-1* labels_y * np.log(predicted_y))/np.size(predicted_y, axis =1) 
    
    def lossgradfunc(self, predicted_y, labels_y): # Delta Categorical CE with Softmax.
        return (predicted_y - labels_y)/np.size(predicted_y, axis =1) #ref(https://deepnotes.io/softmax-crossentropy)
        
    def train(self, input_x, labels_y, learning_rate = 0.1, max_epochs = 100):
        self.loss_values = []
        train_start_time = time.time()
        for epoch in range(max_epochs):

            self.predicted_y = self.predict(input_x)
            self.layers[-1].backpropagate(self.lossgradfunc(self.predicted_y, labels_y))
            for i in range(len(self.layers)-2, -1, -1):
                self.layers[i].backpropagate(self.layers[i+1].backpropagated_grad)
        
            # Gradient descent learning.
            for i in range(len(self.layers)):
                self.layers[i].weight -= learning_rate * self.layers[i].weight_grad
        
            # Print metrics.
            current_loss = self.lossfunc(self.predicted_y, labels_y)
            self.loss_values.append(current_loss)
            print(" Epoch: " + str(epoch) + "        Cross Entropy Loss: " + str(current_loss), end="\r")
        print("\n")
        train_end_time = time.time()
        return self.check_accuracy(self.predict(input_x), labels_y), (train_end_time - train_start_time)
        # Network training returns accuracy and training time after training finishes

# Importing of dataset.
print("Started importing of datasets.")
train_data = np.loadtxt('dataset/mnist_train.csv', dtype = np.float32, delimiter=",")
test_data = np.loadtxt('dataset/mnist_test.csv', dtype = np.float32, delimiter=",")
print("Finished importing of datasets and starting processing.")

# Dataset processing
train_imgs_x = train_data[:,1:].T
test_imgs_x = test_data[:,1:].T
train_labels_y = train_data[:,0]
test_labels_y = test_data[:,0]

del train_data
del test_data

train_imgs_x = np.asfarray(train_imgs_x, np.float32) * 0.99 / 255 + 0.01
train_imgs_x = np.vstack((train_imgs_x, np.ones(np.size(train_imgs_x, axis= 1), np.uint8)))
test_imgs_x = np.asfarray(test_imgs_x, np.float32) * 0.99 / 255 + 0.01
test_imgs_x = np.vstack((test_imgs_x, np.ones(np.size(test_imgs_x, axis= 1), np.uint8)))

num_different_labels = 10

train_labels_onehot = np.zeros((num_different_labels, len(train_labels_y)))  # Transforming to one-hot representation
for i, val in enumerate(train_labels_y):
    train_labels_onehot[int(val),int(i)] = 1
del train_labels_y

test_labels_onehot = np.zeros((num_different_labels, len(test_labels_y)))    # Transforming to one-hot representation
for i, val in enumerate(test_labels_y):
    test_labels_onehot[int(val),int(i)] = 1
del test_labels_y

print("Finished dataset processing.")

input_pixels = 28 * 28
output_categories = 10

# Demonstration of dataset images.
#get_img = np.random.randint(low=0, high=60000, size=(15,))
#for image in train_imgs_x.T[get_img,:-1]:
#    plot = plt.figure()
#    temp_img = np.reshape(image, (28,28))
#    plt.imshow(temp_img)
#    plt.draw()
    
########################################
## Experiments for Project Submission ##
########################################

# Experiment 1a - single relu activated layer
Iters = []
TrainingAccuracies = []
TestingAccuracies = []
TrainingTimes = []
for current_iteration_limit in range(1,201,20):   # Start 1, End 200, Step 20
    print("Experiment 1a (relu) with epoch limit = " + str(current_iteration_limit - 1))
    Iters.append(current_iteration_limit) 
    MyNeuralNetwork = NeuralNetwork("MNIST NN", input_pixels+1, output_categories, output_activation_fn= 'relu') 
    MyNeuralNetwork.compile_network()
    current_tr_acc, current_time = MyNeuralNetwork.train(train_imgs_x,train_labels_onehot, learning_rate = 0.1, max_epochs= current_iteration_limit)
    TrainingTimes.append(current_time)
    TrainingAccuracies.append(current_tr_acc)
    current_ts_out = MyNeuralNetwork.predict(test_imgs_x)
    TestingAccuracies.append(MyNeuralNetwork.check_accuracy(current_ts_out, test_labels_onehot))
    del MyNeuralNetwork
fig, axs = plt.subplots(3)
fig.suptitle('Single relu activated layer network')
axs[0].plot(Iters, TrainingTimes, 'tab:red')
axs[1].plot(Iters, TrainingAccuracies, 'tab:blue')
axs[2].plot(Iters, TestingAccuracies , 'tab:green')
axs.flat[0].set(ylabel='Training Time')
axs.flat[1].set(ylabel='Training Accuracy')
axs.flat[2].set(ylabel='Testing Accuracy')
axs.flat[2].set(xlabel='Number of Iterations')
plt.draw()
del Iters, TrainingAccuracies, TestingAccuracies, TrainingTimes

# Experiment 1b - single tanh activated layer
Iters = []
TrainingAccuracies = []
TestingAccuracies = []
TrainingTimes = []
for current_iteration_limit in range(1,201,20):   # Start 1, End 200, Step 20
    print("Experiment 1b (tanh) with epoch limit = " + str(current_iteration_limit - 1))
    Iters.append(current_iteration_limit) 
    MyNeuralNetwork = NeuralNetwork("MNIST NN", input_pixels+1, output_categories, output_activation_fn= 'tanh') 
    MyNeuralNetwork.compile_network()
    current_tr_acc, current_time = MyNeuralNetwork.train(train_imgs_x,train_labels_onehot, learning_rate = 0.1, max_epochs= current_iteration_limit)
    TrainingTimes.append(current_time)
    TrainingAccuracies.append(current_tr_acc)
    current_ts_out = MyNeuralNetwork.predict(test_imgs_x)
    TestingAccuracies.append(MyNeuralNetwork.check_accuracy(current_ts_out, test_labels_onehot))
    del MyNeuralNetwork
fig, axs = plt.subplots(3)
fig.suptitle('Single tanh activated layer network')
axs[0].plot(Iters, TrainingTimes, 'tab:red')
axs[1].plot(Iters, TrainingAccuracies , 'tab:blue')
axs[2].plot(Iters, TestingAccuracies, 'tab:green')
axs.flat[0].set(ylabel='Training Time')
axs.flat[1].set(ylabel='Training Accuracy')
axs.flat[2].set(ylabel='Testing Accuracy')
axs.flat[2].set(xlabel='Number of Iterations')
plt.draw()
del Iters, TrainingAccuracies, TestingAccuracies, TrainingTimes

# Using the following formula 
# n_hidden = n_samples / (alpha * (n_input + n_output)) 
# where alpha is a scaling factor from 2 to 10
# n_hidden = 60000 / (alpha *  (785 + 10))
# Experiment 2 - One hidden layer (relu activated)

Nodes = []
TrainingErrors = []
TestingErrors = []
for current_alpha in range(2,11): # Start 2, End 10, Step 1
    current_node_per_layer = int(len(train_imgs_x.T) / (current_alpha * (input_pixels + 1 + output_categories)))
    print("Experiment 2 (single hidden relu layer) with hidden layer nodes = " + str(current_node_per_layer))
    Nodes.append(current_node_per_layer) 
    MyNeuralNetwork = NeuralNetwork("MNIST NN", input_pixels+1, output_categories, output_activation_fn= 'none') 
    MyNeuralNetwork.add_layer(current_node_per_layer, 'relu')
    MyNeuralNetwork.compile_network()
    current_tr_acc, _ = MyNeuralNetwork.train(train_imgs_x,train_labels_onehot, learning_rate = 0.1, max_epochs= 100)
    TrainingErrors.append(1 - current_tr_acc)
    current_ts_out = MyNeuralNetwork.predict(test_imgs_x)
    TestingErrors.append(1 - MyNeuralNetwork.check_accuracy(current_ts_out, test_labels_onehot))
    del MyNeuralNetwork
plt.figure()
plt.plot(Nodes, TrainingErrors, 'r')
plt.plot(Nodes, TestingErrors, 'b')
plt.xlabel("Nodes Per Layer")
plt.ylabel("Error Rates")
plt.legend(('Training','Testing'))
plt.draw()
del Nodes, TrainingErrors, TestingErrors

# Experiment 3 - Two hidden layers (relu activated)
Nodes = []
TrainingErrors = []
TestingErrors = []
for current_alpha in range(2,11):   # Start 2, End 10, Step 1
    current_node_per_layer = int(len(train_imgs_x.T) / (current_alpha * (input_pixels + 1 + output_categories)))
    print("Experiment 3 (double hidden relu layer) with hidden layer nodes = " + str(current_node_per_layer))
    Nodes.append(current_node_per_layer) 
    MyNeuralNetwork = NeuralNetwork("MNIST NN", input_pixels+1, output_categories, output_activation_fn= 'none') 
    MyNeuralNetwork.add_layer(current_node_per_layer, 'relu')
    MyNeuralNetwork.add_layer(current_node_per_layer, 'relu')
    MyNeuralNetwork.compile_network()
    current_tr_acc, _ = MyNeuralNetwork.train(train_imgs_x,train_labels_onehot, learning_rate = 0.1, max_epochs= 100)
    TrainingErrors.append(1 - current_tr_acc)
    current_ts_out = MyNeuralNetwork.predict(test_imgs_x)
    TestingErrors.append(1 - MyNeuralNetwork.check_accuracy(current_ts_out, test_labels_onehot))
    del MyNeuralNetwork
plt.figure()
plt.plot(Nodes, TrainingErrors, 'r')
plt.plot(Nodes, TestingErrors, 'b')
plt.xlabel("Nodes Per Layer")
plt.ylabel("Error Rates")
plt.legend(('Training','Testing'))
plt.draw()
del Nodes, TrainingErrors, TestingErrors

# Experiment 4 - Three hidden layers (relu activated)
Nodes = []
TrainingErrors = []
TestingErrors = []
for current_alpha in range(2,11):   # Start 2, End 10, Step 1
    current_node_per_layer = int(len(train_imgs_x.T) / (current_alpha * (input_pixels + 1 + output_categories)))
    print("Experiment 4 (triple hidden relu layer) with hidden layer nodes = " + str(current_node_per_layer))
    Nodes.append(current_node_per_layer) 
    MyNeuralNetwork = NeuralNetwork("MNIST NN", input_pixels+1, output_categories, output_activation_fn= 'none') 
    MyNeuralNetwork.add_layer(current_node_per_layer, 'relu')
    MyNeuralNetwork.add_layer(current_node_per_layer, 'relu')
    MyNeuralNetwork.add_layer(current_node_per_layer, 'relu')
    MyNeuralNetwork.compile_network()
    current_tr_acc, _ = MyNeuralNetwork.train(train_imgs_x,train_labels_onehot, learning_rate = 0.1, max_epochs= 100)
    TrainingErrors.append(1 - current_tr_acc)
    current_ts_out = MyNeuralNetwork.predict(test_imgs_x)
    TestingErrors.append(1 - MyNeuralNetwork.check_accuracy(current_ts_out, test_labels_onehot))
    del MyNeuralNetwork
plt.figure()
plt.plot(Nodes, TrainingErrors, 'r')
plt.plot(Nodes, TestingErrors, 'b')
plt.xlabel("Nodes Per Layer")
plt.ylabel("Error Rates")
plt.legend(('Training','Testing'))
plt.draw()
del Nodes, TrainingErrors, TestingErrors


# Experiment 5 - One hidden layer (tanh activated)
Nodes = []
TrainingErrors = []
TestingErrors = []
for current_alpha in range(2,11):   # Start 2, End 10, Step 1
    current_node_per_layer = int(len(train_imgs_x.T) / (current_alpha * (input_pixels + 1 + output_categories)))
    print("Experiment 5 (single hidden tanh layer) with hidden layer nodes = " + str(current_node_per_layer))
    Nodes.append(current_node_per_layer) 
    MyNeuralNetwork = NeuralNetwork("MNIST NN", input_pixels+1, output_categories, output_activation_fn= 'none') 
    MyNeuralNetwork.add_layer(current_node_per_layer, 'tanh')
    MyNeuralNetwork.compile_network()
    current_tr_acc, _ = MyNeuralNetwork.train(train_imgs_x,train_labels_onehot, learning_rate = 0.1, max_epochs= 100)
    TrainingErrors.append(1 - current_tr_acc)
    current_ts_out = MyNeuralNetwork.predict(test_imgs_x)
    TestingErrors.append(1 - MyNeuralNetwork.check_accuracy(current_ts_out, test_labels_onehot))
    del MyNeuralNetwork
plt.figure()
plt.plot(Nodes, TrainingErrors, 'r')
plt.plot(Nodes, TestingErrors, 'b')
plt.xlabel("Nodes Per Layer")
plt.ylabel("Error Rates")
plt.legend(('Training','Testing'))
plt.draw()
del Nodes, TrainingErrors, TestingErrors

# Experiment 6 - Two hidden layers (tanh activated)
Nodes = []
TrainingErrors = []
TestingErrors = []
for current_alpha in range(2,11):   # Start 2, End 10, Step 1
    current_node_per_layer = int(len(train_imgs_x.T) / (current_alpha * (input_pixels + 1 + output_categories)))
    print("Experiment 6 (double hidden tanh layer) with hidden layer nodes = " + str(current_node_per_layer))
    Nodes.append(current_node_per_layer) 
    MyNeuralNetwork = NeuralNetwork("MNIST NN", input_pixels+1, output_categories, output_activation_fn= 'none') 
    MyNeuralNetwork.add_layer(current_node_per_layer, 'tanh')
    MyNeuralNetwork.add_layer(current_node_per_layer, 'tanh')
    MyNeuralNetwork.compile_network()
    current_tr_acc, _ = MyNeuralNetwork.train(train_imgs_x,train_labels_onehot, learning_rate = 0.1, max_epochs= 100)
    TrainingErrors.append(1 - current_tr_acc)
    current_ts_out = MyNeuralNetwork.predict(test_imgs_x)
    TestingErrors.append(1 - MyNeuralNetwork.check_accuracy(current_ts_out, test_labels_onehot))
    del MyNeuralNetwork
plt.figure()
plt.plot(Nodes, TrainingErrors, 'r')
plt.plot(Nodes, TestingErrors, 'b')
plt.xlabel("Nodes Per Layer")
plt.ylabel("Error Rates")
plt.legend(('Training','Testing'))
plt.draw()
del Nodes, TrainingErrors, TestingErrors

# Experiment 7 - Three hidden layers (tanh activated)
Nodes = []
TrainingErrors = []
TestingErrors = []
for current_alpha in range(2,11):   # Start 2, End 10, Step 1
    current_node_per_layer = int(len(train_imgs_x.T) / (current_alpha * (input_pixels + 1 + output_categories)))
    print("Experiment 7 (triple hidden tanh layer) with hidden layer nodes = " + str(current_node_per_layer))
    Nodes.append(current_node_per_layer) 
    MyNeuralNetwork = NeuralNetwork("MNIST NN", input_pixels+1, output_categories, output_activation_fn= 'none') 
    MyNeuralNetwork.add_layer(current_node_per_layer, 'tanh')
    MyNeuralNetwork.add_layer(current_node_per_layer, 'tanh')
    MyNeuralNetwork.add_layer(current_node_per_layer, 'tanh')
    MyNeuralNetwork.compile_network()
    current_tr_acc, _ = MyNeuralNetwork.train(train_imgs_x,train_labels_onehot, learning_rate = 0.1, max_epochs= 100)
    TrainingErrors.append(1 - current_tr_acc)
    current_ts_out = MyNeuralNetwork.predict(test_imgs_x)
    TestingErrors.append(1 - MyNeuralNetwork.check_accuracy(current_ts_out, test_labels_onehot))
    del MyNeuralNetwork
plt.figure()
plt.plot(Nodes, TrainingErrors, 'r')
plt.plot(Nodes, TestingErrors, 'b')
plt.xlabel("Nodes Per Layer")
plt.ylabel("Error Rates")
plt.legend(('Training','Testing'))
plt.draw()
del Nodes, TrainingErrors, TestingErrors

plt.show()