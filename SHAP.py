#!/usr/bin/env python
# coding: utf-8

# In[1]:



import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import ssl
from pandas import Series
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

ssl._create_default_https_context = ssl._create_unverified_context
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

#coding:utf-8
import json
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

import time, json, requests
# 抓取腾讯疫情实时json数据
url = 'https://raw.githubusercontent.com/nikipi/WHITE_BOX/master/partial_name/FIFA.csv'
data = pd.read_csv(url)
data.head()



# In[2]:


y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary


feature_name=[i for i in data.columns if data[i].dtype in [np.int64,np.int64]]
X=data[feature_name]


# In[3]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)


# In[ ]:


#look at SHAP values for a single row of the dataset 
#(we arbitrarily chose row 5). 
#For context, we'll look at the raw predictions before looking at the SHAP values.


# In[4]:


row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show] 


# In[5]:


data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

my_model.predict_proba(data_for_prediction_array)


# In[ ]:


# The team is 70% likely to have a player win the award.


# In[8]:


import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)


# In[9]:


shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)


# In[10]:


import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(val_X)

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values[1], val_X)


# In[11]:


import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# calculate shap values. This is what we will plot.
shap_values = explainer.shap_values(X)

# make plot.
shap.dependence_plot('Ball Possession %', shap_values[1], X, interaction_index="Goal Scored")


# In[ ]:




