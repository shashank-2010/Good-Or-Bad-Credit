#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# In[2]:


pip install ucimlrepo


# In[3]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 
  
df = pd.DataFrame(X, columns=statlog_german_credit_data.feature_names)  # Add column names
df['target'] = y


# In[4]:


df_renamed = df.rename(columns={'Attribute1':'checkin_acc','Attribute2':'Duration','Attribute3':'Credit_hist','Attribute4':'Purpose',
           'Attribute5':'Credit_amt','Attribute6':'Savings_acc','Attribute7':'Present_emp_since','Attribute8':'Install_rate',
           'Attribute9':'Personal_status','Attribute10':'guarantors','Attribute11':'residing_since',
           'Attribute12':'Property','Attribute13':'Age','Attribute14':'Other_installment_plans','Attribute15':'Housing',
           'Attribute16':'existin_credits','Attribute17':'Job','Attribute18':'providin_maintenance','Attribute19':'Telephone',
           'Attribute20':'foreign_worker','target':'status'})

#Status -> 1 = Good, 2 = Bad


# In[5]:


df_renamed.drop(['Purpose','guarantors','Property','Housing','providin_maintenance', 'Telephone', 'foreign_worker'],axis=1,inplace=True)


# In[6]:


df_renamed


# In[7]:


df_renamed.columns


# Data Cleaning

# In[8]:


df_renamed.isnull().sum()


# In[9]:


df_renamed.duplicated().sum()


# In[10]:


df_renamed['status'].value_counts()


# Feature Engineering

# In[11]:


X_features = df_renamed.drop('status',axis=1)
X_features.columns


# In[12]:


#one-hot encoding
encoded_credit_df = pd.get_dummies(X_features,drop_first=True)


# In[13]:


encoded_credit_df.columns


# In[14]:


#model building
import statsmodels.api as sm


# In[15]:


df_renamed['status'] = np.where(df_renamed['status'] == 1, 0, 1)


# In[16]:


y = df_renamed.status
X = sm.add_constant(encoded_credit_df)


# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=42)


# In[18]:


logreg = sm.Logit(y_train, X_train).fit()


# In[19]:


logreg.summary2()


# In[20]:


#selecting features with P>|z| lower than 0.05
def get_sig_features(logreg):
    feature_pvals_df = pd.DataFrame(logreg.pvalues)
    feature_pvals_df['vars'] = feature_pvals_df.index
    feature_pvals_df.columns = ['pvals','vars']
    return list(feature_pvals_df[feature_pvals_df.pvals<=0.05]['vars'])


# In[21]:


significant_vars = get_sig_features(logreg)
significant_vars


# In[26]:


significant_vars = ['Duration','Credit_amt','Install_rate','Age','checkin_acc_A13','checkin_acc_A14','Credit_hist_A34','Savings_acc_A65']
X1 = sm.add_constant(encoded_credit_df[significant_vars])


# In[27]:


X1_train,X1_test,y_train,y_test = train_test_split(X1,y,train_size=0.7,random_state=42)


# In[28]:


logreg_1 = sm.Logit(y_train,X1_train).fit()


# In[29]:


logreg_1.summary2()


# Prediction

# In[31]:


y_pred = logreg_1.predict(X1_test)


# In[33]:


pred_df = pd.DataFrame({'Actual_value':y_test, 'Predicted_prob_value':y_pred})


# In[36]:


pred_df.sample(10)


# Optimal Classification Cut-off to understand the classification

# In[40]:


#calculate ROC curve to know true positive rate, false positive rate, threshold
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(pred_df.Actual_value, pred_df.Predicted_prob_value)
roc_auc = auc(fpr, tpr) 


# In[41]:


#plotting Roc curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[44]:


#now calculating optimal classification cut off using yoden's index = max(of the threshold depending diff between sensitivity and specificity)

tpr_fpr = pd.DataFrame({'tpr':tpr,'fpr':fpr, 'Threshold':thresholds})
tpr_fpr['diff'] = tpr_fpr.tpr - tpr_fpr.fpr
tpr_fpr.sort_values('diff', ascending=False).head(3)


# Classification based on the cut off

# In[45]:


pred_df['Predited_status'] = pred_df.Predicted_prob_value.map(lambda x: 1 if x >0.22 else 0)


# In[47]:


pred_df.sample(5)


# In[58]:


#Confusion matrix - to understand the error between the actual_value and Predicted_status
from sklearn.metrics import confusion_matrix, classification_report
con_matrix = confusion_matrix(pred_df.Actual_value,pred_df.Predited_status)
sns.heatmap(con_matrix, annot=True, fmt='.3f',
           xticklabels=['Good Credit','Bad Credit'],
           yticklabels=['Good Credit','Bad Credit'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[60]:


#classification report 
print(classification_report(pred_df.Actual_value,pred_df.Predited_status))


# In[ ]:




