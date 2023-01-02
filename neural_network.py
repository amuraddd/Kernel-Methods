"""
Two Layer Neural Network
Ali Murad
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

X, label = load_svmlight_file('Data/mnist.scale.bz2')
X = X.toarray()
labels = label.astype(int)

# append 1 as bias to X
X = np.append(X, np.ones((X.shape[0], 1)), axis=1)

# make sure to append 1s to X and one hot encode the labels
x_train, x_test, y_train, y_test = train_test_split(X, labels,
                                                   test_size=0.2,
                                                   random_state=0)

# one hot encode Y
def one_hot_encoder(y_actuals):
    """one hot encode y for loss calculations as: [0 , 1] columns"""
    total_classes = len(np.unique(y_actuals))
    encoded_y = np.zeros((len(y_actuals), total_classes))
    for i in range(len(y_actuals)):
        encoded_y[i, y_actuals[i]] = 1
    return encoded_y

#encode training and test y labels
y_train_encoded = one_hot_encoder(y_train)
y_test_encoded = one_hot_encoder(y_test)

class NeuralNetwork:
    """
    Two layer Neural Network with Back Propagation.
    """
    def __init__(self, num_neurons=100, num_output_neurons=10, num_inputs=5, learning_rate=0.02):
        self.num_neurons = num_neurons
        self.num_output_neurons = num_output_neurons
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        
        # initialize the weight vectors for the hidden layer
        self.w = np.random.normal(loc = 0, scale = 1, size=(num_neurons, num_inputs))
        
        # initialize weight vectors for the output layer - there will be as many weight vectors as the number of output neurons 
        # and the number of outputs in the hidden layer(hidden neurons plus 1 for bias)
        self.v = np.random.normal(loc = 0, scale = 1, size=(num_output_neurons, num_neurons+1))#add 1 to account for the bias #np.append(), np.ones((self.num_output_neurons, 1)), axis=1)
    
    # define the activation function for the hidden layer - sigmoid
    def sigmoid(self, val):
        """
        Sigmoid for Activation of the hidden layer and/or output layer.        
        """
        return (1/(1+np.exp(-val)))
        
    # activation function for the output layer
    def softmax(self, h_vals):
        """
        Softmax for the activation of the output layer.
        """
        exponentiated_values = np.exp(h_vals - np.max(h_vals, axis=0, keepdims=True))
        probs = exponentiated_values/np.sum(exponentiated_values, axis=0, keepdims=True)
        return probs
    
    #forward propagation
    def forward_propagation(self, x):
        """
        Forward propagation to propagate a sample trhough two layers.
        """
        w_dot_x = np.dot(self.w, x) #this dot product will give one output per hidden neuron - in the case of 100 neurons it will be a vector of 100, 1
        h = np.append(self.sigmoid(w_dot_x), 1) #apply activation function and append 1 as bias        

        v_dot_h = np.dot(self.v, h)
        y_hat = self.sigmoid(v_dot_h)
        
        return w_dot_x, h, y_hat
    
    # derivative of sigmoid
    def gradient_sigmoid(self, vals):
        """
        Derivative of the sigmoid functiion
        """
        return np.exp(-vals)/((1+np.exp(-vals))**2)
    
    # back propagation
    def fit(self, x, y):
        """
        Backpropagation to learn weights for each, the hidden and the output layer.
        """
        for sample in range(len(x)):
            
            # get predicted values for the output neurons
            w_dot_x, h, y_hat = self.forward_propagation(x[sample])

            # back propagation
            # derivative for updating weights for the output layer
            dl_dv = -np.dot((y[sample] - y_hat).reshape(self.num_output_neurons, 1), h.reshape(self.num_neurons+1, 1).T)
            
            #select all but the weight for the bias for the output layer
            v = np.array([i[:-1] for i in self.v])
            
            for i in range(len(self.w)):
                
                #calculate loss
                loss = y[sample] - y_hat

                dl_dh = -loss.reshape(1, self.num_output_neurons) @ v
                dh_dw = self.gradient_sigmoid(np.dot(self.w[i], x[sample])).reshape(1,1) @ x[sample].reshape(1, self.num_inputs)

                #derivative for the hidden layer
                dl_dw = dl_dh.T @ dh_dw

                #update the weights of the first layer first 
                self.w = self.w - self.learning_rate*dl_dw #update the weight vector for the first layer
                #update the weights for the output layer using the 
            self.v = self.v - self.learning_rate*dl_dv 
            
    #predict
    def predict(self, x_test):
        """
        Predict using the learned weights in back propagation.
        """
        # get predicted values for the output neurons
        y_hat = list()
        
        for i in range(len(x_test)):
            _, _, yi_hat = self.forward_propagation(x_test[i])
            y_hat.append(yi_hat)
        
        return y_hat
    
# num output neurons should be the same as the number of classes
# initialize the model with the number of classes and num_ouput_neurons, num_inputs as number of features, and the learning rate for back propagation.
model = NeuralNetwork(num_output_neurons=10, num_inputs=x_train.shape[1], learning_rate=0.02)

#make predictions
test_predictions = model.predict(x_test)
train_predictions = model.predict(x_train)

# reverse one hot encoded predictions and actuals
test_predictions_reversed = np.argmax(np.array(test_predictions).round(), axis=1)
y_test_reversed = np.argmax(y_test_encoded, axis=1)

# reverse training y labels
train_predictions_reversed = np.argmax(np.array(train_predictions).round(), axis=1)
y_train_reversed = np.argmax(y_train_encoded, axis=1)

def neural_net_accuracy(actuals, test):
    accuracy = list()
    for i in range(len(test)):
        if test[i]==actuals[i]:
            accuracy.append(1)
        else:
            accuracy.append(0)
    accuracy_rate = sum(accuracy)/len(accuracy)*100
    
    return accuracy_rate

print("Kernel Perceptron Testing Accuracy: ", neural_net_accuracy(y_test_reversed, test_predictions_reversed ), "%")
print("Kernel Perceptron Training Accuracy: ", neural_net_accuracy(y_train_reversed , train_predictions_reversed), "%")

print("Test Confusion Matrix")
print(pd.crosstab(test_predictions_reversed, y_test_reversed))
