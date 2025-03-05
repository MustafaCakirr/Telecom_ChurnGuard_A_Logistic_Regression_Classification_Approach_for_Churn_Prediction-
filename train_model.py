import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score
from train_test_preprocessing import X_train,X_test,y_train,y_test
import matplotlib.pyplot as plt 


def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b

def sigmoid(z):
    z = np.clip(z, -500, 500)  
    return 1 / (1 + np.exp(-z))

# Perform forward and backward propagation to compute cost and gradients
def forward_backward_propagation(w,b,X_train,y_train):
    # forward propagation
    z = np.dot(np.transpose(w), X_train.T) + b 
    y_head = sigmoid(z)
    y_train = y_train.values.reshape(-1)  # Pandas Series to NumPy array + reshape

    epsilon = 1e-9  # Small constant to prevent log(0)

    loss = -y_train.T * np.log(y_head + epsilon) - (1 - y_train.T) * np.log(1 - y_head + epsilon)

    cost = np.sum(loss) / X_train.shape[0]

    # backward propagation
    derivative_weight = np.dot(X_train.T,(y_head.T - y_train)) / X_train.shape[0]
    derivative_bias = np.sum(y_head.T-y_train) / X_train.shape[0]
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients

# Update parameters using gradient descent
def update(w,b,X_train,y_train,learning_rate,number_of_iteration):
   
    for i in range (number_of_iteration):  # NUMBER OF ITERATION KAÇ KEZ FORWARD-BACKWARD YAPILACAĞINI SEÇİYORUZ.
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,X_train,y_train)
        
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
    
    parameters = {"weight" : w, "bias" : b}
  
    return parameters,gradients

# Predict binary labels for test data using learned parameters
def predict (w,b,X_test):
    z = sigmoid(np.dot(w.T,X_test.T)+b)
    y_prediction = np.zeros((1,X_test.shape[0]))
    for i in range(z.shape[1]):
        y_prediction[0,i] = 1
        if z[0,i] <= 0.5:
            y_prediction[0,i] = 0 
        else:
            y_prediction[0,i] = 1
    return y_prediction

# Train a custom logistic regression model using gradient descent and evaluate test accuracy 
def logistic_regression(X_train,y_train,X_test,y_test,learning_rate,number_of_iterations):
    # initialize 
    dimension = X_train.shape[1]
    w,b = initialize_weights_and_bias(dimension)

    parameters,gradients = update(w,b,X_train,y_train,learning_rate,number_of_iterations)

    y_predictions_test = predict(parameters["weight"],parameters["bias"],X_test)
    y_predictions_test = y_predictions_test.flatten()  
    y_test = y_test.values.reshape(-1)  # Pandas Series to NumPy array

    print("Test accuracy: {} %".format(100 - np.mean(np.abs(y_predictions_test - y_test)) * 100))


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Train and evaluate logistic regression model using pipeline and display test accuracy
def sklearn_logistic_regression():
    
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400))
    model.fit(X_train, y_train)
    
    y_predict = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_predict)
    print(f"Sklearn LRC Test Accuracy:{accuracy}")

    """ or """    

    # print("Sklearn LRC Test accuracy: ",model.score(X_test,y_test))



