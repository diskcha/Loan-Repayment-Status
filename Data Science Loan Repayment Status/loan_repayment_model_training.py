
#In [0]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score, cohen_kappa_score, matthews_corrcoef
warnings.filterwarnings("ignore")

# In[1]:
loans = pd.read_csv("loans_preprocessed.csv")
print("Label Encoding Column 'Purpose'")
le =LabelEncoder()
loans.purpose = le.fit_transform(loans.purpose)
x = loans[loans.columns[~loans.columns.isin(['not.fully.paid'])]]
y = loans[loans.columns[loans.columns.isin(['not.fully.paid'])]]
print("Oversampling data to make balanced label dataset using SMOTE oversampling******")
sm = SMOTE()
x, y = sm.fit_resample(x,y)



# In[2]:
#metrics evaluator function
def model_performance(y_test, y_pred):
    print("**********TESTING MODEL USING METRICS************")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Accuracy: ", np.round(accuracy_score(y_test, y_pred),3)*100)
    print("Sensitivity: ", np.round(recall_score(y_test, y_pred),3)*100)
    print("Specificity: ", np.round(tn/(tn+fp),3)*100)
    print("ROC AUC: ", np.round(roc_auc_score(y_test, y_pred),3)*100)
    print("Kappa Score: ", np.round(cohen_kappa_score(y_test, y_pred),3))
    print("Mathews Correlation Coeff", np.round(matthews_corrcoef(y_test, y_pred),3))


# In[3]:

print("Splitting data into training and testing: ratio = 80:20****")
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
#RANDOM FOREST CLASSIFIER with bootstrapping
print("Data trainging in progress: Random Forest Classifier........")
rf = RandomForestClassifier(n_estimators=600,bootstrap=True)
rf.fit(x_train, y_train)
y_pred  = rf.predict(x_test)
print("Training Complete")
#calling our function to check performance
print(model_performance(y_test, y_pred), "\n")
print("d)What is the best suited evaluating metrics for this dataset?")
print("For repayment dectection, we have to have highly sensitive data, since we have to decrease the false negatives more than false postives: i.e.: No repayment calls not detected wrongly are more costly than non correct repayment calls wrongly detected. \nAccuracy is not an accurate predictor, since data is highly imbalanced, even if the model is bad, it will predict 0 more easily and will have high accuracy. \nCohen's kappa and Mathews correlation both are good metrics. Mathew's correlation tells how good a binary classifier is. Our score (0.8) is a good score.")


# In[4]:


print("b)Identify 5 best borrowers that will repay the entire amount. ")
#Highest probability of repaying the amount
x = loans[loans.columns[~loans.columns.isin(['not.fully.paid'])]]
y = loans[loans.columns[loans.columns.isin(['not.fully.paid'])]]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
probs = rf.predict_proba(x_test)
high = []
for i in range(len(probs)):
    high.append(probs[i][0])
index = np.argsort(high)[::-1]
high = sorted(high)[::-1]
print("From our given dataset, we see rows with these 5 indexes with highest probability of repyaing the amount",  index[:5], "with probabilities: ", high[:5])

# In[5]:


print("c)What 5 factors affect loan repayment the most?")
#****FEATURE IMPORTANCE****
#storing indexes of sorted array of feature importance
index = np.argsort(rf.feature_importances_)[::-1]
#sorting array according to increasing feature importance
features = sorted(rf.feature_importances_)[::-1]
cols = list(loans.columns)
cols.remove("not.fully.paid")
col_order = [cols[i] for i in index]
print("5 most important features are: ", ", ".join(col_order[:5]))
plt.figure(figsize=(20,10))
sns.barplot(col_order, features)
plt.title("FEATURE IMPORTANCE")
plt.show()

