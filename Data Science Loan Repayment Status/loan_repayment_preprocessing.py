
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score, cohen_kappa_score, matthews_corrcoef
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings("ignore")


# In[2]:


loans = pd.read_csv("loans.csv")
print("DATA SHAPE: ",loans.shape)
print("DATA TYPES:")
print(loans.dtypes)


# In[3]:


print("Checking target variable data frequency")
print(loans["not.fully.paid"].value_counts())
print("a)If information of certain borrowers is missing, we have to treat it for null values, and that can be done in many ways: Mode, median , mean, deletion. We have to treat it according to the variable and it's description.")
print("Checking total null values in dataset")
print(loans.isna().sum())


# In[4]:


print("Column = Public Records\nFilling null values: Checking data frequency")
print("Filling null values with mode:0")
print(loans["pub.rec"].value_counts())
loans["pub.rec"].fillna(0, inplace = True)
print("Null values removed for this column")


# In[5]:


print("Column = Inquiry last 6 months\nFilling null values: Checking data frequency")
print("Filling null values with mode:0")
print(loans["inq.last.6mths"].value_counts().head())
#Filling null values with mode:0
loans["inq.last.6mths"].fillna(0, inplace = True)
print("Null values removed for this column")


# In[6]:


print("Column = Log annual income \nFilling null values: Checking data description")
print("Filling null values with mean : 10.93")
print(loans["log.annual.inc"].describe())
loans["log.annual.inc"].fillna(loans["log.annual.inc"].mean(), inplace = True)
print("Null values removed for this column")


# In[7]:


print("Column = Delinquency 2 years\nFilling null values: Checking data frequency")
print("Filling null values with mode:0")
print(loans["delinq.2yrs"].value_counts())
loans["delinq.2yrs"].fillna(0, inplace = True)
print("Null values removed for this column")


# In[8]:


print("Column = revol.util \nFilling null values: Checking data description")
print(loans["revol.util"].describe())
print("Checking data frequency")
print(loans["revol.util"].value_counts().head())
#Filling null values with mode:0
loans["revol.util"].fillna(int(loans["revol.util"].mode()[0]), inplace = True)
print("Null values removed for this column")


# In[9]:


print(loans["days.with.cr.line"].describe())
print("Null value percent for days.with.cr.line: ", loans["days.with.cr.line"].isna().sum()/len(loans)*100, "%")
#Very less % (0.3%) of the data points are null, and we can see that neither of mean, mode or median replacement is possible in this case, we will drop these rows 
loans.dropna(subset=["days.with.cr.line"], inplace = True)
print("Null values removed for this column")


# In[10]:


print(loans.isna().sum())
print("\n**********Null value treatment complete**********")
loans.to_csv("loans_preprocessed.csv", index = False)

