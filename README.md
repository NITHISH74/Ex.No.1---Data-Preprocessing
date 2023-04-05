# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
i)   Importing the libraries

ii)  Importing the dataset

iii) Taking care of missing data

iv)  Encoding categorical data

v)   Normalizing the data

vi)  Splitting the data into test and train

## PROGRAM:
```python
Developed By: NITHISHWAR S

Register No: 212221230071


import pandas as pd
import numpy as np

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semester 3/19AI411 - Neural Networks/Churn_Modelling.csv")
df

df.isnull().sum()

#Check for Duplicate Values
df.duplicated()

df.describe()

#Detect the Outliers
# Outliers are any abnormal values going beyond
df['Exited'].describe()

""" Normalize the data - There are range of values in different columns of x are different. 

To get a correct ne plot the data of x between 0 and 1 

LabelEncoder can be used to normalize labels.
It can also be used to transform non-numerical labels to numerical labels.
"""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1 = df.copy()

df1["Geography"] = le.fit_transform(df1["Geography"])
df1["Gender"] = le.fit_transform(df1["Gender"])

'''
MinMaxScaler - Transform features by scaling each feature to a given range. 
When we normalize the dataset it brings the value of all the features between 0 and 1 so that all the columns are in the same range, and thus there is no dominant feature.'''

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))

df1

df1.describe()

# Since values like Row Number, Customer Id and surname  doesn't affect the output y(Exited).
#So those are not considered in the x values
X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values
print(X)

y = df1.iloc[:,-1].values
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)
print("Size of X_train: ",len(X_train))

print(X_test)
print("Size of X_test: ",len(X_test))

X_train.shape
```
## OUTPUT:
### Dataset:
![image](https://user-images.githubusercontent.com/94164665/229984090-78371044-71b3-4ab7-a324-d22d90a69322.png)
### Checking For Null Values:
![image](https://user-images.githubusercontent.com/94164665/229984159-1e710a40-fa49-4b15-aa87-71d8950489e8.png)
### Checking For Duplicate Values:
![image](https://user-images.githubusercontent.com/94164665/229984226-95785abb-a86d-47a2-9576-3e0a393cecd7.png)
### Describing Data:

### Checking For Outliers In Exited Column:
![image](https://user-images.githubusercontent.com/94164665/229985393-29725535-4027-4b7f-befb-3bda7c680760.png)

### Normalized Dataset:
![image](https://user-images.githubusercontent.com/94164665/229985103-d48c145e-7cc3-49b0-abc9-36efb8a38d63.png)

### Describing Normalized Dataset:
![image](https://user-images.githubusercontent.com/94164665/229985427-9115dc1d-e599-4579-b04a-ff6d54a816af.png)

## RESULT:
Data preprocessing is performed in a data set downloaded from Kaggle.


