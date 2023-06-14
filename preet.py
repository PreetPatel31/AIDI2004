#!/usr/bin/env python
# coding: utf-8

# In[68]:


pip install GitPython


# In[69]:



import pandas as pd
import joblib
import git
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Read the dataset
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ["ID", "Diagnosis", "radius_mean", "perimeter_mean", ...]  # Replace with actual column names
data = pd.read_csv(dataset_url, names=column_names)


# In[70]:


# Display the first few rows of the dataset
data.head()


# In[71]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[72]:


# Preprocess the data and split it into input features (X) and target variable (y)
X = data.drop(columns=["Diagnosis"])  # Replace with your actual feature columns
y = data["Diagnosis"]


# In[73]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[74]:


# Create an instance of the logistic regression model
model = LogisticRegression()


# In[75]:


# Convert continuous target variable to categorical values
y_train_categorical = pd.cut(y_train, bins=2, labels=[0, 1])


# In[81]:


# Create an instance of the logistic regression model
model = LogisticRegression()

