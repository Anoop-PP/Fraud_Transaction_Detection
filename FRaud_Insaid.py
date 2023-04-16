#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


df=pd.read_csv("Fraud.csv")


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[3]:


genuine=len(df[df.isFraud==0])
Fraud=len(df[df.isFraud==1])
genuine_percentage=(genuine/(genuine + Fraud)) * 100
Fraud_percentage=(Fraud/(genuine + Fraud)) * 100
print('The number of Genuine transaction is',genuine,'and percentage is {:.4f} %'.format(genuine_percentage))
print('The number of Fraud transaction is',Fraud,'and percentage is {:.4f} %'.format(Fraud_percentage))


# In[4]:


# Merchants
M=df[df['nameDest'].str.contains('M')]
M.head()


# In[25]:



len(M)


# #### Visualization

# In[30]:


plt.figure(figsize=(15,12))
sns.heatmap(df.corr(),annot=True,cmap = 'magma')


# In[17]:


labelz=['Genuine','Fraud']
count_C=df.value_counts(df['isFraud'], sort= True)
count_C.plot(kind = "pie",  autopct='%f%%', startangle=60)
plt.title("Visualization of Labels")
plt.ylabel("Percentage")

plt.show()


# In[19]:


df_1=df.copy()
df_1.head()


# #### Label Encoding

# In[21]:


df_1.info()


# The datatype of following variales should be changed :-  nameOrig ,type ,nameDest .
# Inorder to check the multicolinearity

# In[26]:


objList = df_1.select_dtypes(include = "object").columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for con in objList:
    df_1[con] = le.fit_transform(df_1[con].astype(str))

print (df_1.info())


# In[27]:


df_1.head()


# #### Multicollinearity

# In[29]:


# Variance inflation Factor
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(df):
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    return(vif)

calc_vif(df_1)


# By above it is clear that :
# (oldbalanceOrg  and  newbalanceOrig)
# (oldbalanceDest and  newbalanceDest)
# (nameOrig  and  nameDest )  are correlated to each other
# 
# 

# In[33]:


# And therefore need to combine together
df_1['Actual_amount_orig'] = df_1.apply(lambda x: x['oldbalanceOrg'] - x['newbalanceOrig'],axis=1)
df_1['Actual_amount_dest'] = df_1.apply(lambda x: x['oldbalanceDest'] - x['newbalanceDest'],axis=1)
df_1['TransactionPath'] = df_1.apply(lambda x: x['nameOrig'] + x['nameDest'],axis=1)


df_1 = df_1.drop(['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','step','nameOrig','nameDest'],axis=1)

calc_vif(df_1)


# In[40]:


corr=df_1.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr,annot=True)


# #### Model Building

# In[42]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import itertools
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# In[43]:


# Performinf the Scaling
scaler = StandardScaler()
df_1["NormalizedAmount"] = scaler.fit_transform(df_1["amount"].values.reshape(-1, 1))
df_1.drop(["amount"], inplace= True, axis= 1)

Y = df_1["isFraud"]
X = df_1.drop(["isFraud"], axis= 1)


# #### Train And Test 

# In[44]:


(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size= 0.3, random_state= 42)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)


# #### Model Training
# 

# In[45]:


# DECISION TREE

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

Y_pred_dt = decision_tree.predict(X_test)
decision_tree_score = decision_tree.score(X_test, Y_test) * 100


# In[ ]:


# RANDOM FOREST

random_forest = RandomForestClassifier(n_estimators= 100)
random_forest.fit(X_train, Y_train)

Y_pred_rf = random_forest.predict(X_test)
random_forest_score = random_forest.score(X_test, Y_test) * 100


# ####  Evaluation

# In[48]:




print("Decision Tree Score: ", decision_tree_score)
print("Random Forest Score: ", random_forest_score)


# In[ ]:


# Decision Tree
print("TP,FP,TN,FN - Decision Tree")
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_dt).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')



# Random Forest

print("TP,FP,TN,FN - Random Forest")
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_rf).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')


# For both Decision Tree and Random Forest TP is almost same but FP of Random Forest is less compare Decision Tree and therefore Random Forest seems good

# In[ ]:


# classification report - Decision Tree

classification_report_dt = classification_report(Y_test, Y_pred_dt)
print("Classification Report - Decision Tree")
print(classification_report_dt)



# classification report - Random Forest

classification_report_rf = classification_report(Y_test, Y_pred_rf)
print("Classification Report - Random Forest")
print(classification_report_rf)


# #### Visualization of Confusion Matrix

# In[ ]:


# visualising confusion matrix -Decision Tree


disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_dt)
disp.plot()
plt.title('Confusion Matrix - DT')
plt.show()

# visualising confusion matrix - Random Forest
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf)
disp.plot()
plt.title('Confusion Matrix - RF')
plt.show()


# #### AUC  ROC

# In[ ]:


# AUC ROC - Decision Tree
# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_dt)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC - DECISION TREE')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# AUC ROC - Random Forest
# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_rf)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC - RANDOM FOREST')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# AUC of both DECISION TREE and RANDOM FOREST are same

# RANDOM FOREST and DECISION TREE is used , because the dataset is highly unbalanced (Genuine: Fraud :: 99.87:.13). Another important reason to select these model is that, here we need high precision and recall value than accuracy.
