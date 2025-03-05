from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from data_preprocessing import X,y
from imblearn.over_sampling import BorderlineSMOTE
import numpy as np


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# SMOTE (VALUE 1 DEĞERİ %25'in altında)  Counter({0: 4165, 1: 1469})
# smote = BorderlineSMOTE(random_state=42, k_neighbors=1)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test_ = scaler.transform(X_test)  