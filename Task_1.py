#!/usr/bin/env python
# coding: utf-8

# <h2 align=center>Tumor Diagnosis (Part 1): Exploratory Data Analysis</h2>
# <img src="https://storage.googleapis.com/kaggle-datasets-images/180/384/3da2510581f9d3b902307ff8d06fe327/dataset-cover.jpg">
# 

# ### About the Dataset:

# The [Breast Cancer Diagnostic data](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) is available on the UCI Machine Learning Repository. This database is also available through the [UW CS ftp server](http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/).
# 
# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

# **Attribute Information**:
# 
# - ID number
# - Diagnosis (M = malignant, B = benign) 3-32)

# Ten real-valued features are computed for each cell nucleus:
# 
# 1. radius (mean of distances from center to points on the perimeter) 
# 2. texture (standard deviation of gray-scale values) 
# 3. perimeter 
# 4. area 
# 5. smoothness (local variation in radius lengths) 
# 6. compactness (perimeter^2 / area - 1.0) 
# 7. concavity (severity of concave portions of the contour) 
# 8. concave points (number of concave portions of the contour)
# 9. symmetry
# 10. fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# 
# All feature values are recoded with four significant digits.
# 
# Missing attribute values: none
# 
# Class distribution: 357 benign, 212 malignant

# ### Task 1: Loading Libraries and Data

# In[74]:


import numpy as np 
import pandas as pd 
import seaborn as sns  
import matplotlib.pyplot as plt
import time


# In[75]:


data = pd.read_csv('data.csv')


# <h2 align=center> Exploratory Data Analysis </h2>
# 
# ---

#  

# ###  2: Separate Target from Features

# In[76]:


# retrieve overall dataset
data.head()


# In[77]:


col  = data.columns
col


# In[78]:


# Create 2 dataframes x as potential feartures and y as target clause
drop_cols = ['id','diagnosis', 'Unnamed: 32']
x = data.drop(drop_cols, axis = 1)
y = data.diagnosis
x.head()


#  

# ### 3: Plot Diagnosis Distributions

# In[79]:


ax = sns.countplot(y, label = 'Count')
B, M = y.value_counts()
print( 'Number of Benign Tumors ', B)
print( 'Number of Malignant Tumors ', M)


# In[80]:


x.describe()


#  

# <h2 align=center> Data Visualization </h2>
# 
# ---

#  

# ### 4: Visualizing Standardized Data with Seaborn

# In[81]:


data1 = x
data1_std = (data1 - data1.mean()) / data1.std()
data1 =pd.concat([y, data1_std.iloc[:, 0: 10]], axis = 1)
data1 = pd.melt( data1, id_vars = 'diagnosis', 
               var_name = 'features', 
               value_name = 'value' )
plt.figure(figsize = [10, 10] )
sns.violinplot( x ='features', y = 'value', hue = 'diagnosis', data = data1, split = True, inner = 'quart' )
plt.xticks( rotation = 45)


#  

# ###  5: Violin Plots and Box Plots

# In[82]:


data2 = x
data2_std = (data2 - data2.mean()) / data2.std()
data2 =pd.concat([y, data2_std.iloc[:, 10: 20]], axis = 1)
data2 = pd.melt( data2, id_vars = 'diagnosis', 
               var_name = 'features', 
               value_name = 'value' )
plt.figure(figsize = [10, 10] )
sns.violinplot( x ='features', y = 'value', hue = 'diagnosis', data = data2, split = True, inner = 'quart' )
plt.xticks( rotation = 45)


# In[83]:


data3 = x
data3_std = (data3 - data3.mean()) / data3.std()
data3 =pd.concat([y, data3_std.iloc[:, 20: 30]], axis = 1)
data3 = pd.melt( data3, id_vars = 'diagnosis', 
               var_name = 'features', 
               value_name = 'value' )
plt.figure(figsize = [10, 10] )
sns.violinplot( x ='features', y = 'value', hue = 'diagnosis', data = data3, split = True, inner = 'quart' )
plt.xticks( rotation = 45)


# In[94]:


plt.figure(figsize = [12, 8] )
sns.boxplot( x = 'features', y = 'value', hue = 'diagnosis', data = data3)
plt.xticks( rotation = 45)


# In[95]:


plt.close('all')


#  

# ###  6: Using Joint Plots for Feature Comparison

# In[96]:


sns.jointplot( x.loc[:, 'concavity_worst'], x.loc[:, 'concave points_worst'], kind = 'regg', color ='orange')


#  

# ###  7: Observing the Distribution of Values and their Variance with Swarm Plots

# In[86]:


plt.figure(figsize = [10, 6] )
sns.set(style = 'whitegrid', palette = 'muted')
sns.swarmplot( x ='features', y = 'value', hue = 'diagnosis', data = data1)
plt.xticks( rotation = 45)


# In[87]:


plt.figure(figsize = [10, 6] )
sns.set(style = 'whitegrid', palette = 'muted')
sns.swarmplot( x ='features', y = 'value', hue = 'diagnosis', data = data2)
plt.xticks( rotation = 45)


# In[88]:


plt.figure(figsize = [10, 6] )
sns.set(style = 'whitegrid', palette = 'muted')
sns.swarmplot( x ='features', y = 'value', hue = 'diagnosis', data = data3)
plt.xticks( rotation = 45)


#  

#  

# ###  8: Observing all Pair-wise Correlations

# In[99]:


f, ax = plt.subplots(figsize = ( 18, 18))
sns.heatmap( x.corr(), annot = True, linewidth = 0.5, fmt = '.1f', ax = ax )


# #### Thank you and Good luck!
