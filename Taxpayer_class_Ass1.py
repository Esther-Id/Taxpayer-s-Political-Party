#!/usr/bin/env python
# coding: utf-8

# # Taxpayer's Political Party

# __Context:__
# A tax is a compulsory financial charge or some other type of levy imposed on a taxpayer (an individual or legal entity) by a governmental organization in order to fund government spending and various public expenditures.
# 
# 
# __Data Description:__
# The dataset contains information about US taxpayers. There are 10 independent columns and 1 dependent column. This dataset includes attributes like household income, household debt level, if the taxpayer is married or not, how many cars their household has, if they filed their taxes in the last three years or not. Some of the attribute informations are given below:
# 
# - HHI: Household income
# 
# - HHDL: Household debt level
# 
# - Married: There are three categories for a taxpayer 0, 1, 2 with regards to marriage.
# 
# - PoliticalParty: Name of the political party
# 
# - CollegeGrads: Grade in College out of 5
# 
# - AHHAge: Average household age
# 
# - cars: number of cars in house
# 
# - Filed in YYYY: Tax filed in given year YYYY
# 
# 
# __Objective:__
# Build a machine learning model that would predict the political party to which a taxpayer belongs to
# - Evaluation Criteria
# Submissions are evaluated using Accuracy Score.

# ### Import neccessary libraries

# In[204]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR ,SVC
from sklearn import datasets

from sklearn.metrics import roc_curve, confusion_matrix, make_scorer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,  GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


# ### Read in and understand the ataset

# In[205]:


tax_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/tax_payers/train_set_label.csv" )
tax_data.head()

test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/tax_payers/test_set_label.csv')


# In[206]:


tax_data.tail()


# In[207]:


tax_data.shape , test_data.shape


# In[208]:


tax_data.info()


# In[209]:


tax_data.drop("Unnamed: 0",axis = 1,inplace = True)

test_data.drop("Unnamed: 0",axis = 1,inplace = True)


# In[210]:


tax_data.isnull().sum()


# In[211]:


tax_data.duplicated().sum()


# In[212]:


tax_data.nunique()
cat_cols = ["Married","CollegGrads","Cars","Filed in 2017",
"Filed in 2016","Filed in 2015", "PoliticalParty"]
cat_cols


# In[213]:


for col in cat_cols:
    print(tax_data[col].value_counts(normalize = True))
    print("-" * 40)
    


# ### Exploratory Data Analysis

# In[214]:


#summary statistics
tax_data.describe().T


# In[215]:


tax_data.describe(include = "object").T


# In[216]:


#Univariate Analysis
#Dependent variable-Political Party
fig = plt.figure(figsize = (9,5))
sns.countplot(x = tax_data.PoliticalParty);


# In[217]:


#univariate plot of other categorical variables
fig  = plt.figure(figsize = (9,3))
sns.countplot(x = tax_data["Married"])


fig  = plt.figure(figsize = (9,5))
sns.countplot(x = tax_data["Cars"])


fig  = plt.figure(figsize = (9,3))
sns.countplot(x = tax_data["CollegGrads"])

fig  = plt.figure(figsize = (9,3))
sns.countplot(x = tax_data["Filed in 2017"]);


fig  = plt.figure(figsize = (9,3))
sns.countplot(x = tax_data["Filed in 2016"])
        
        
fig  = plt.figure(figsize = (9,3))
sns.countplot(x = tax_data["Filed in 2015"]);        
        


# In[218]:


#univariate plot of Numerical variable

fig  = plt.figure(figsize = (9,3))
sns.distplot(tax_data.HHI, kde = True)

fig  = plt.figure(figsize = (9,3))
sns.distplot(tax_data.HHDL, kde = True)

fig  = plt.figure(figsize = (9,3))
sns.distplot(tax_data["AHHAge"]);


# In[219]:


#bivariate 

# Create a correlation matrix.
corr = tax_data.corr()
fig = plt.figure(figsize = (14,8))
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,annot =True)

plt.title('Heatmap of Correlation Matrix')
corr


# In[220]:


sns.pairplot(tax_data,hue='PoliticalParty');


# ### Modeling

# In[221]:



X = tax_data.drop('PoliticalParty',axis = 1)
y = tax_data.PoliticalParty


X_test = tax_data.drop('PoliticalParty',axis = 1)


# In[222]:


#Theres usuakly no need to scale ensemble tree ML models

X_train,X_val ,y_train,y_val = train_test_split(X,y ,test_size = 0.25,random_state = 1)

X_train.shape,X_val.shape ,y_train.shape,y_val.shape 


# ### DecisionTreeClassifier

# In[223]:


dt=DecisionTreeClassifier(criterion='gini',max_depth=10)

dt.fit(X_train,y_train)


# In[224]:


#evaluate model on train data
dt_pred=dt.predict(X_train)
dt_score = accuracy_score(dt_pred ,y_train)

dt_score


# In[225]:


#evaluate model on validation data
dt_pred = rfc.predict(X_val)

dt_score = accuracy_score(dt_pred ,y_val)
dt_score


# ### RandomForestClassifier

# In[226]:


from sklearn.ensemble import RandomForestClassifier   # import the model
rfc = RandomForestClassifier(n_estimators= 32)


# In[227]:


rfc.fit(X_train,y_train)


# In[228]:


#evaluate model on train data
rfc_pred = rfc.predict(X_train)

rfc_score = accuracy_score(rfc_pred ,y_train)

rfc_score


# In[229]:


#evaluate model on validation data
rfc_pred = rfc.predict(X_val)

rfc_score = accuracy_score(rfc_pred ,y_val)
rfc_score


# __Observation__
# - The tree model are seriously overfitting on the training data

# In[230]:


#evaluate model on validation data

rfc_res = rfc.predict(test_data)

rfc_res = pd.DataFrame(rfc_res) #convert to dataframe
rfc_res.columns = ["prediction"] #rename column as prediction

rfc_res.to_csv("rfc_result" ,index = False)



rfcr = pd.read_csv("rfc_result")
rfcr.head()


# ### GradientBoostingClassifier

# In[231]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc = gbc.fit(X_train,y_train)


# In[232]:


#evaluate model on train data
gbc_pred = gbc.predict(X_train)

gbc_score = accuracy_score(gbc_pred ,y_train)

gbc_score


# In[233]:


#evaluate model on validation data

gbc_pred = gbc.predict(X_val)

gbc_score = accuracy_score(gbc_pred ,y_val)

gbc_score


# In[234]:


res = gbc.predict(test_data)
res = pd.DataFrame(res)

res.index = test_data.index
res.columns = ["prediction"]
res.set_index('prediction' , inplace = True)

res.to_csv('prediction_results.csv') 
#files.download('prediction_results.csv')


# In[235]:


res =pd.read_csv('prediction_results.csv')
res.head()


# In[236]:


#HYPER PARAMETER TUNING To improve models?


# In[237]:


# Grid search method define
def grid_search_wrapper(refit_score='accuracy_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(mdl, param_grid, scoring=scorers, refit=refit_score,
                           cv=None, return_train_score=True, n_jobs=-1)
    grid_search.fit(X,y)

    # make the predictions
    y_pred = grid_search.predict(X)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the Validation data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(target_val, y_pred),
                 columns=['pred_D', 'pred_I', 'pred_R'], index=['D', 'I', 'R']))
    return grid_search

