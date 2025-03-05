import pandas as pd 
from remove_outliers import remove_outliers

data = pd.read_excel("C:/Telco_customer_churn.xlsx")

# Drop columns that are not required for modeling
data.drop(["CustomerID","Lat Long","Country","State","City"
           ,"Zip Code","Internet Service","Payment Method"
           ,"Churn Reason","Churn Score","Churn Label"
           ,"Count","Latitude","Longitude","CLTV"],axis=1,inplace=True)

# Data Type Control 
# data.info()

# to Numeric 
data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors='coerce')

# data.isnull().sum()
data["Total Charges"].fillna(data["Total Charges"].median(), inplace=True)

# One-Hot Encodding
onehot_encoding_list = ["Multiple Lines","Online Security","Online Backup","Device Protection","Tech Support","Streaming TV","Streaming Movies","Contract"]
encoded_cols = pd.get_dummies(data[onehot_encoding_list],drop_first=True)
data_remaining = data.drop(columns=onehot_encoding_list)
data = pd.concat([data_remaining,encoded_cols],axis=1)

# String to Numeric(1-0) Yes-No
str_cols_Y_N = ["Senior Citizen","Partner", "Dependents","Phone Service","Paperless Billing"]
for col in str_cols_Y_N:
    data[col] = data[col].map({"Yes":1,"No":0})

# String to Numeric(1-0) True-False
str_cols_T_F = ["Multiple Lines_No phone service","Multiple Lines_Yes","Online Security_No internet service",
                "Online Security_Yes","Online Backup_No internet service","Online Backup_Yes","Device Protection_No internet service",
                "Device Protection_Yes","Tech Support_No internet service","Tech Support_Yes","Streaming TV_No internet service",
                "Streaming TV_Yes","Streaming Movies_No internet service","Streaming Movies_Yes","Contract_One year","Contract_Two year"]
for col in str_cols_T_F:
    data[col] = data[col].map({True:1,False:0})

# String to Numeric(1-0) Male(0)-Female(1)
data["Gender"] = data["Gender"].map({"Female":1,"Male":0})

# Outliers
outliers_list = ["Total Charges","Monthly Charges","Tenure Months"]
data = remove_outliers(data,outliers_list)

# Features and Target Vector
X = data.drop(["Churn Value"],axis=1)
y = data["Churn Value"]

# print(y.value_counts(normalize=True))




