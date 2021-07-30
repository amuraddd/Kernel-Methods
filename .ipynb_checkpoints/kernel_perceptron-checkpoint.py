"""
Kernel Perceptron 
Ali Murad
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from mini2_utils import binary_converter, decimal_converter

mnist_X, mnist_label = load_svmlight_file('Data/mnist.scale.bz2')
mnist_X = mnist_X.toarray()
mnist_label = mnist_label.astype(int)

X = mnist_X[:]
label = mnist_label[:]

class Kernel_Perceptron:
    def __init__(self, x, a=0, b=0, d=1, iterations=10, sample_size=1.0):
        self.a = a
        self.b = b
        self.d = d
        self.alpha = np.zeros(x.shape[0], dtype=np.float64)
        self.iterations = iterations
        self.sample_size = int(len(x)*sample_size)
        
    def fit(self, x, y):
        """
        The function takes as input:
        x: data
        y: class labels for the data
        """
        a = self.a
        b = self.b
        d = self.d
        alpha = self.alpha[:self.sample_size]                
        x = x[:self.sample_size]
        y = y[:self.sample_size]
        
        #gram matrix
        x_gram = x @ x.T
        def change_y(vals):
            for v in range(len(vals)):                
                if vals[v]==1:
                    vals[v]=1
                if vals[v]==0:
                    vals[v]=-1
            return vals

        y = change_y(y)              
        for i in range(self.iterations): #outer loop for multiple itrations over the entire data     
            for j in range(len(y)): #iterations over samples
                b_i = 1/((j+1)**2) #initialize the stepsize as 1/t^2 where t is each iteration(add 1 to avoid division by 0)                
                k = (a + b*x_gram[:, j])**d #polynomial kernel is given by: (a + b*np.dot(x,x_i))**(d)                
                if np.dot(alpha, k) > 0:
                    y_hat = 1                                    
                if np.dot(alpha, k) <= 0:
                    y_hat = -1                    
                if y[j] == y_hat:
                    pass
                else:
                    alpha[j] = alpha[j] + b_i*y[j]  #update the alpha vector                
            self.alpha = alpha
    
    

    def predict(self, x_train, x_test):
        """
        The function takes in:
        x_train: data to map to
        x_test: data to make predictions on
        Returns:
        y_hat: predicted classes for each sample
        """      
        x_train = x_train[:self.sample_size]
        test_samples = x_test.shape[0] #number of rows in the data        
        test_gram = x_train @ x_test.T
        
        y_hat = list()
        for i in range(test_samples):
            kernel = (self.a + self.b*test_gram[:, i])**self.d
            if np.dot(self.alpha, kernel) > 0:
                y_hat.append(1)
            if np.dot(self.alpha, kernel) <= 0:
                y_hat.append(0)
        
        return y_hat
    
# train test split
x_train, x_test, y_train, y_test = train_test_split(X, label,
                                                   test_size=0.3,
                                                   random_state=0)

y_train1, y_train2, y_train3, y_train4 = binary_converter(y_train) #get labels for ECOC
y_test1, y_test2, y_test3, y_test4 = binary_converter(y_test) #get labels for ECOC

#instantiate 4 models - 1 for each bit
model_1 = Kernel_Perceptron(x_train, a=1, b=1, d=4, iterations=150, sample_size=0.10) #quadratic kernel
model_2 = Kernel_Perceptron(x_train, a=1, b=1, d=4, iterations=150, sample_size=0.10) #quadratic kernel
model_3 = Kernel_Perceptron(x_train, a=1, b=1, d=4, iterations=150, sample_size=0.10) #quadratic kernel
model_4 = Kernel_Perceptron(x_train, a=1, b=1, d=4, iterations=150, sample_size=0.10) #quadratic kernel

# train models for 4 bits in parallel
model_1.fit(x_train, y_train1)
model_2.fit(x_train, y_train2)
model_3.fit(x_train, y_train3)
model_4.fit(x_train, y_train4)

# test predictions
pred_1 = model_1.predict(x_train, x_test)
pred_2 = model_2.predict(x_train, x_test)
pred_3 = model_3.predict(x_train, x_test)
pred_4 = model_4.predict(x_train, x_test)

# training predictions
train_pred_1 = model_1.predict(x_train, x_train)
train_pred_2 = model_2.predict(x_train, x_train)
train_pred_3 = model_3.predict(x_train, x_train)
train_pred_4 = model_4.predict(x_train, x_train)


# training predictions
sample_size = int(len(y_train)*0.10)
y_train1, y_train2, y_train3, y_train4 = binary_converter(y_train) #get labels for ECOC
meta_train = [y_train1[: sample_size], y_train2[: sample_size], y_train3[: sample_size], y_train4[: sample_size]]
train_array = [''.join(str(item) for item in column) for column in zip(*meta_train)]
train_array = decimal_converter(train_array)

#testing predictions
meta_pred = [pred_1, pred_2, pred_3, pred_4]
pred_array = [''.join(str(item) for item in column) for column in zip(*meta_pred)]
pred_array = decimal_converter(pred_array)

#actual data
meta_test = [y_test1, y_test2, y_test3, y_test4]
test_array = [''.join(str(item) for item in column) for column in zip(*meta_test)]
test_array = decimal_converter(test_array)

def kernel_perceptron_accuracy(actuals, test):
    accuracy = list()
    for i in range(len(test)):
        if test[i]==actuals[i]:
            accuracy.append(1)
        else:
            accuracy.append(0)
    accuracy_rate = sum(accuracy)/len(accuracy)*100
    
    return accuracy_rate

print("Kernel Perceptron Testing Accuracy: ", kernel_perceptron_accuracy(test_array, pred_array), "%")
print("Kernel Perceptron Training Accuracy: ", kernel_perceptron_accuracy(test_array, train_array), "%")

print("Test Confusion Matrix")
print(pd.crosstab(test_predictions_reversed, y_test_reversed))


























