#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

titanic_data=pd.read_csv('train.csv')
titanic_data.head()


# In[ ]:





# In[5]:


titanic_data.shape


# In[6]:


titanic_data.info()


# In[7]:


titanic_data.isnull().sum()


# In[11]:


titanic_data.drop(columns =['Cabin'], axis=1)
titanic_data


# In[19]:


#titanic_data['Age'].value_counts()

titanic_data['Age'].isnull().sum()

titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)


# In[70]:


titanic_data['Embarked'].mode()


# In[71]:


titanic_data['Embarked'].mode()[0]


# In[54]:


titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)


# In[61]:


titanic_data.isnull().sum()


# In[62]:


titanic_data['Survived'].value_counts()


# In[59]:


sns.set()
sns.countplot('Survived',data=titanic_data)


# In[7]:


sns.countplot('Sex',hue='Survived',data=titanic_data)


# In[8]:


sns.countplot('Pclass', data=titanic_data)


# In[10]:


sns.countplot('Pclass',hue='Survived',data=titanic_data)


# In[11]:


titanic_data['Sex'].value_counts()


# In[12]:


titanic_data['Embarked'].value_counts()


# In[15]:


titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)


# In[16]:


titanic_data.head()


# In[22]:


X=titanic_data.drop(columns=['PassengerId','Survived','Name','Ticket','Cabin','Age'],axis=1)
Y=titanic_data['Survived']


# In[23]:


print(X)


# In[24]:


print(Y)


# In[26]:


#splitting the data into training data and test data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

print(X_train)


# In[21]:


print(X.shape,X_train.shape,X_test.shape)


# In[27]:


# Model Training

#Logistic Regression

model= LogisticRegression()

model.fit(X_train, Y_train)


# In[27]:


model.fit(X_train, Y_train)


# In[28]:


#Model Evaluation

#Accuracy Score

X_train_prediction=model.predict(X_train)

print(X_train_prediction)


# In[29]:


training_data_accuracy=accuracy_score(Y_train,X_train_prediction)


# In[30]:


X_test_prediction=model.predict(X_test)

print(X_test_prediction)


# In[31]:


training_data_accuracy=accuracy_score(Y_train,X_test_prediction)


# In[ ]:




