# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 02:12:44 2024

@author: domingosdeeulariadumba
"""

# # The Kids Rights Index Statistical Modeling



# ## Initialization

# In[5]:


# Main libraries

import os
import tabula as tb
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.impute import SimpleImputer


# ## Data Extraction

# In[7]:


# files in the directory

path = (r'C:\Users\domingosdeeularia\...\KidsRightsIndex\Scoretables')

files_list = os.listdir(path)

files_list


# In[8]:


# This step is meant to extract the tables and concatenate them

table_list = []
actual_cols = ['country', 'overall_score', 'life', 'health', 'education', 'protection', 'environment']

for file in files_list:
    table = tb.read_pdf(path + '/' + file, stream = True, pages = 'all')
    df_table = pd.concat([table[i].iloc[4:, 1:8] for i in range(len(table))])
    df_table.columns = actual_cols
    table_list.append(df_table)

df_kri = pd.concat(table_list)


# ## Data Preprocessing

# Replacing every  ',' with '.'

df_kri = df_kri.apply(lambda col: col.str.replace(',', '.'))


# Dropping missing values

df_kri.dropna(inplace = True)
df_kri.head()


# Replacing any non-numeric value by 'x'

df_kri.iloc[:,1:] = df_kri.iloc[:,1:].map(lambda cell: 'x' if cell[0].isnumeric() == False else cell)


# Replacing 'x' with most frequent terms

imp = SimpleImputer(missing_values = 'x', strategy = 'most_frequent')
imp_array = imp.fit_transform(df_kri)


# Passing the imputation array to a dataframe

df_kri = pd.DataFrame(data = imp_array, columns = actual_cols)
df_kri.head()


# Converting numeric values to float

for col in actual_cols[1:]:
    df_kri[col] = df_kri[col].astype(float)


# Re-checking the dtypes

df_kri.dtypes


# ## Statistical Analysis

# Displaying the pairplot

sns.pairplot(df_kri)
plt.show()


# Correlation Heatmap

sns.heatmap(df_kri.iloc[:, 1:].corr(), annot = True, cmap = 'Greys')
plt.show()


# Setting the statistical model

X,y = df_kri.iloc[:, 2:], df_kri.overall_score
X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()


# In[21]:


# Results Summary

results.summary()


# ## Closing Remarks

'''
So, given that the p-value is lower than 0.05 (for every predictor), which is the 
significance level defined has the decision criterion to analyse the 
relationship between the inputs and the output, we can affirm that evidences 
has shown with 95% of confidence  that about 78% (Adj. R-squared) of the 
variation of the overall score in The Kids Rights Index may be expained by the 
combined variation of the predictors. The remained 22% may be explained by 
other external factors.
'''
#                   ________  ________   _______   ______ 
#                  /_  __/ / / / ____/  / ____/ | / / __ \
#                   / / / /_/ / __/    / __/ /  |/ / / / /
#                  / / / __  / /___   / /___/ /|  / /_/ / 
#                 /_/ /_/ /_/_____/  /_____/_/ |_/_____/  
