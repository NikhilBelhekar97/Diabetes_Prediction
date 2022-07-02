#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# # CSV File import
# 

# In[3]:


df= pd.read_csv("E:/data analysis  projects/machine learning project/project1 Diabetes_prediction/diabetes.csv")


# # First five rows in the dataset
# 

# In[5]:


df.head(5)


# # No of rows and columns

# In[6]:


df.shape


# # Statistical measure of the data

# In[7]:


df.describe()


# # No of diabetic and non diabetic labelling

# In[9]:


df['Outcome'].value_counts()


# # Mean values of diabetic and non diabetic label

# In[10]:


df.groupby('Outcome').mean()


# # Seperating data and labels

# In[11]:


X= df.drop(columns='Outcome',axis=1)
Y= df['Outcome']


# In[12]:


X


# In[13]:


Y


# # Fitting and transforming X

# In[14]:


scaler= StandardScaler()


# In[15]:


scaler.fit(X)


# In[17]:


standardized_data= scaler.transform(X)


# In[19]:


print(standardized_data)


# In[20]:


X= standardized_data
Y= df['Outcome']
print(X)
print(Y)


# # Train Test Split

# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[23]:


print(X.shape, X_train.shape, X_test.shape)


# # Training the model

# In[24]:


classifier = svm.SVC(kernel='linear')


# In[49]:


classifier.fit(X_train, Y_train)


# # Making Prediction

# In[50]:


input_data=(0,180,66,39,0,42,1.893,25)


# # Converting input data into numpy array

# In[51]:


input_data_array=np.asarray(input_data)


# In[52]:


#reshaping array


# In[53]:


input_reshaped_array=input_data_array.reshape(1,-1)


# # Standardize the input data

# In[54]:


std_data = scaler.transform(input_reshaped_array)
print(std_data)


# # Prediction

# In[55]:


prediction = classifier.predict(std_data)
print(prediction)
if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

