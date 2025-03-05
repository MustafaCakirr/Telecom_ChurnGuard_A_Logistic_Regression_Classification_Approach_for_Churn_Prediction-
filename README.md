# ChurnGuard: A Logistic Regression Classification Approach for Churn Prediction

ChurnGuard is a machine learning project that implements a logistic regression approach to predict customer churn in the telecom industry. The project is written in Python, follows PEP 8 coding standards, and offers both a custom implementation using gradient descent as well as a scikit-learn based implementation for comparison.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Modules](#modules)
- [Results](#results)
- [Acknowledgements](#Acknowledgements)


## Overview

The goal of this project is to predict whether a customer will churn using logistic regression. The project includes:
- Data preprocessing (handling missing values, type conversion, one-hot encoding, and outlier removal).
- Splitting the dataset into training and test sets.
- Implementing a custom logistic regression model using gradient descent.
- Benchmarking against scikit-learn's logistic regression using a pipeline.

## Features

- **Data Preprocessing:**  
  Handles missing values, converts data types, applies one-hot encoding, and removes outliers using the IQR method.
  
- **Custom Logistic Regression:**  
  Implements gradient descent from scratch to optimize model parameters and evaluate test accuracy.
  
- **Scikit-learn Integration:**  
  Uses a pipeline with `StandardScaler` and `LogisticRegression` to provide a benchmark model.
  
- **Modular Code Structure:**  
  The code is organized into distinct modules for preprocessing, model training, and evaluation, enhancing maintainability and scalability.
  
- **PEP 8 Compliant:**  
  The project follows Python's style guidelines and includes inline documentation for better code readability.




# Modules 

## Data Preprocessing
This module performs data preprocessing for the Telco customer churn dataset. It prepares the raw data for modeling by applying several transformation steps.

### Key Features

- **Drop Unnecessary Columns:**  
  Removes columns that are not needed for model training (e.g., CustomerID, location details, payment info, etc.).

- **Data Type Conversion:**  
  Converts the "Total Charges" column to a numeric type and fills any missing values using the median.

- **One-Hot Encoding:**  
  Applies one-hot encoding to categorical columns such as "Multiple Lines", "Online Security", and others, to convert them into a machine-readable format.

- **Binary Conversion:**  
  Transforms Yes/No and True/False string values into binary (1/0) for columns like "Senior Citizen", "Partner", "Dependents", "Phone Service", and "Paperless Billing".

- **Gender Encoding:**  
  Encodes the "Gender" column into binary values (Female = 1, Male = 0).

- **Feature-Target Split:**  
  Separates the preprocessed data into feature matrix `X` and target vector `y` for subsequent modeling.

## Train Model
This module trains logistic regression models on the Telco customer churn dataset using both custom and scikit-learn implementations.

### Key Features

- **Custom Logistic Regression:**  
  Implements logistic regression with gradient descent, including weight initialization, forward/backward propagation, and parameter updates.

- **Gradient Descent Optimization:**  
  Iteratively minimizes the cost function by updating model parameters based on computed gradients.

- **Prediction Functionality:**  
  Generates binary predictions for test data using the learned weights and bias.

- **Scikit-learn Pipeline:**  
  Provides an alternative model training approach using a pipeline with StandardScaler and LogisticRegression, streamlining preprocessing and model fitting.

- **Model Evaluation:**  
  Calculates and displays test accuracy for both the custom and scikit-learn logistic regression models.

## Train-Test Preprocessing Module
This module splits the Telco customer churn dataset into training and test sets. It also includes optional code for applying data scaling and oversampling techniques.

### Key Features

- **Data Splitting:**  
  Uses `train_test_split` to partition the dataset into training (80%) and testing (20%) subsets.

- **Optional Oversampling:**  
  Contains commented code to apply `BorderlineSMOTE` for balancing the training set when the minority class is below 25%.

- **Optional Scaling:**  
  Provides commented code for feature scaling using `StandardScaler` for both training and test sets.

## Remove Outliers Module
This module provides a function to remove outliers from a DataFrame using the Interquartile Range (IQR) method.

### Key Features

- **IQR Calculation:**  
  Computes the first (Q1) and third (Q3) quartiles and determines the IQR for specified columns.

# Results

## Custom Logistic Regression
- **Cost Progression:**  
  - Iteration 0: 10.345449  
  - Iteration 10: 42240.695316  
  - Iteration 20: 42636.600320  
  - …  
  - Iteration 1990: 43996.477286

- **Test Accuracy:** 71.61%

## Scikit-learn Logistic Regression
- **Test Accuracy:** 80.27%



# Acknowledgements
I would like to thank the Udemy channel **dataiteam** for the inspiration and guidance provided through their tutorials, which greatly contributed to the development of this project.





