import pandas as pd
from data_preprocessing import data
from train_test_preprocessing import *
from train_model import logistic_regression,sklearn_logistic_regression

# Set display options for pandas DataFrames
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

# Run scikit-learn logistic regression model
sklearn_logistic_regression()

# Run custom logistic regression model 
logistic_regression(X_train,y_train,X_test,y_test,learning_rate=0.01,number_of_iterations=2000)