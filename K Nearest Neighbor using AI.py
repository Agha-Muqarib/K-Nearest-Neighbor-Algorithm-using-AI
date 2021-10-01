#!/usr/bin/env python
# coding: utf-8

# ### Importing the Libraries
# 

# In[1]:


import pandas as pd, scipy, numpy as np
import sklearn.preprocessing


# ### Loading the iris dataset

# In[2]:


headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class'] #Assigning the headers


# In[3]:


ds = pd.read_csv('iris.data', names = headernames)
ds.head()


# In[4]:


print(ds)


# ### Splitting up in feature attributes and class variable

# In[5]:


x = ds.iloc[:, :-1].values


# In[6]:


y=ds.iloc[:, 4].values


# ### Train and Test Split
# Next, we will divide the data into train and test split. Following code will split the dataset into 60% training data and 40% of testing data

# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.40)


# ### Data Scaling using the StandardScaler

# In[8]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ### Training a KNN Classifier

# In[9]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7)
classifier.fit(X_train, y_train)


# ### Making the Predictions

# In[10]:


y_pred = classifier.predict(X_test)


# ### Output

# In[11]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


# In[13]:


Y_pred = classifier.predict(X_test)
print(Y_pred)


# ### K-fold Crossvalidation
# K-Folds cross-validator provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default). Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
# 
# more information on https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

# In[29]:


from sklearn.model_selection import KFold
# prepare cross validation
kfold = KFold(2, True) # value of K and shuffle? 
# enumerate splits
for train, test in kfold.split(X_train):
    print('train: %s, test: %s' % (X_train[train], X_train[test]))
   


# In[30]:


for train, test in kfold.split(X_train):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    result = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(result)
    result1 = classification_report(y_test, y_pred)
    print("Classification Report:",)
    print (result1)
    result2 = accuracy_score(y_test,y_pred)
    print("Accuracy:",result2)

