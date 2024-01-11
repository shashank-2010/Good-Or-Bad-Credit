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


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 
  
df = pd.DataFrame(X, columns=statlog_german_credit_data.feature_names)  # Add column names
df['target'] = y


# In[3]:


df_renamed = df.rename(columns={'Attribute1':'checkin_acc','Attribute2':'Duration','Attribute3':'Credit_hist','Attribute4':'Purpose',
           'Attribute5':'Credit_amt','Attribute6':'Savings_acc','Attribute7':'Present_emp_since','Attribute8':'Install_rate',
           'Attribute9':'Personal_status','Attribute10':'guarantors','Attribute11':'residing_since',
           'Attribute12':'Property','Attribute13':'Age','Attribute14':'Other_installment_plans','Attribute15':'Housing',
           'Attribute16':'existin_credits','Attribute17':'Job','Attribute18':'providin_maintenance','Attribute19':'Telephone',
           'Attribute20':'foreign_worker','target':'status'})

#Status -> 1 = Good, 2 = Bad


# In[4]:


df_renamed.drop(['Purpose','guarantors','Property','Housing','providin_maintenance', 'Telephone', 'foreign_worker'],axis=1,inplace=True)


# In[5]:


df_renamed


# In[6]:


df_renamed.columns


# Data Cleaning

# In[7]:


df_renamed.isnull().sum()


# In[8]:


df_renamed.duplicated().sum()


# In[9]:


df_renamed['status'].value_counts()


# Feature Engineering

# In[10]:


X_features = df_renamed.drop('status',axis=1)
X_features.columns


# In[11]:


#one-hot encoding
encoded_credit_df = pd.get_dummies(X_features,drop_first=True)


# In[12]:


encoded_credit_df.columns


# In[13]:


#model building
import statsmodels.api as sm


# In[14]:


df_renamed['status'] = np.where(df_renamed['status'] == 1, 0, 1)


# In[15]:


y = df_renamed.status
X = sm.add_constant(encoded_credit_df)


# In[16]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=42)


# In[17]:


logreg = sm.Logit(y_train, X_train).fit()


# In[18]:


logreg.summary2()


# In[19]:


#selecting features with P>|z| lower than 0.05
def get_sig_features(logreg):
    feature_pvals_df = pd.DataFrame(logreg.pvalues)
    feature_pvals_df['vars'] = feature_pvals_df.index
    feature_pvals_df.columns = ['pvals','vars']
    return list(feature_pvals_df[feature_pvals_df.pvals<=0.05]['vars'])


# In[20]:


significant_vars = get_sig_features(logreg)
significant_vars


# In[21]:


significant_vars = ['Duration','Credit_amt','Install_rate','Age','checkin_acc_A13','checkin_acc_A14','Credit_hist_A34','Savings_acc_A65']
X1 = sm.add_constant(encoded_credit_df[significant_vars])


# In[22]:


X1_train,X1_test,y_train,y_test = train_test_split(X1,y,train_size=0.7,random_state=42)


# In[23]:


logreg_1 = sm.Logit(y_train,X1_train).fit()


# In[24]:


logreg_1.summary2()


# Prediction

# In[25]:


y_pred = logreg_1.predict(X1_test)


# In[26]:


pred_df = pd.DataFrame({'Actual_value':y_test, 'Predicted_prob_value':y_pred})


# In[27]:


pred_df.sample(10)


# Optimal Classification Cut-off to understand the classification

# In[28]:


#calculate ROC curve to know true positive rate, false positive rate, threshold
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(pred_df.Actual_value, pred_df.Predicted_prob_value)
roc_auc = auc(fpr, tpr) 


# In[29]:


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


# In[30]:


#now calculating optimal classification cut off using yoden's index = max(of the threshold depending diff between sensitivity and specificity)

tpr_fpr = pd.DataFrame({'tpr':tpr,'fpr':fpr, 'Threshold':thresholds})
tpr_fpr['diff'] = tpr_fpr.tpr - tpr_fpr.fpr
tpr_fpr.sort_values('diff', ascending=False).head(3)


# Classification based on the cut off

# In[31]:


pred_df['Predited_status'] = pred_df.Predicted_prob_value.map(lambda x: 1 if x >0.22 else 0)


# In[32]:


pred_df.sample(5)


# In[33]:


#Confusion matrix - to understand the error between the actual_value and Predicted_status
from sklearn.metrics import confusion_matrix, classification_report
con_matrix = confusion_matrix(pred_df.Actual_value,pred_df.Predited_status)
sns.heatmap(con_matrix, annot=True, fmt='.3f',
           xticklabels=['Good Credit','Bad Credit'],
           yticklabels=['Good Credit','Bad Credit'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[34]:


#classification report 
print(classification_report(pred_df.Actual_value,pred_df.Predited_status))


# # Using Decision Tree Regression

# In[42]:


y = df_renamed.status
X_new = encoded_credit_df


# In[43]:


X_new_train,X_new_test,y_train,y_test = train_test_split(X_new,y,train_size=0.7, random_state=42)


# In[36]:


from sklearn.tree import DecisionTreeClassifier


# In[37]:


dt_clf = DecisionTreeClassifier()


# In[38]:


#hypertuning the model
from sklearn.model_selection import GridSearchCV


# In[48]:


tuned_para = [{'criterion':['gini','entropy'],
              'max_depth': range(2,10)}]
dt_clf_tuned = GridSearchCV(dt_clf, tuned_para,cv=10,scoring='roc_auc')
dt_clf_tuned.fit(X_new_train,y_train)


# In[50]:


dt_clf_tuned.best_params_


# In[52]:


#tuned model
dt_clf = DecisionTreeClassifier(criterion= 'gini', max_depth = 4)


# In[55]:


#training the tuned model
dt_clf.fit(X_new_train,y_train)
print('Training Score {}'.format(dt_clf.score(X_new_train,y_train)))


# In[57]:


#prediction
y_pred_tree = dt_clf.predict(X_new_test)


# In[59]:


pred_df_tree = pd.DataFrame({'Actual_value':y_test, 'Predicted_prob_value':y_pred_tree})
pred_df_tree.head(5)


# In[87]:


from sklearn.metrics import roc_curve, auc
fpr1, tpr1, thresholds1 = roc_curve(pred_df_tree.Actual_value, pred_df_tree.Predicted_prob_value)
roc_auc = auc(fpr1, tpr1) 


# In[88]:


roc_auc


# In[89]:


plt.figure()
plt.plot(fpr1, tpr1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[66]:


#confusion matrix to visualize the result
from sklearn.metrics import confusion_matrix, classification_report
con_matrix = confusion_matrix(pred_df_tree.Actual_value,pred_df_tree.Predicted_prob_value)
sns.heatmap(con_matrix, annot=True, fmt='.3f',
           xticklabels=['Good Credit','Bad Credit'],
           yticklabels=['Good Credit','Bad Credit'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[68]:


print(classification_report(pred_df_tree.Actual_value,pred_df_tree.Predicted_prob_value))


# # Using KNN

# In[69]:


from sklearn.neighbors import KNeighborsClassifier 


# In[70]:


dt_clf_knn = KNeighborsClassifier()


# In[73]:


#hypertuning
tuned_para_knn = [{'n_neighbors':range(5,10), 'weights': ['uniform', 'distance'],
    'algorithm' :['auto','ball_tree', 'kd_tree', 'brute'],
    'metric':['minkowski','euclidean','mahalanobis']}]
                


# In[74]:


dt_clf_knn_best = GridSearchCV(dt_clf_knn, tuned_para_knn, cv= 10, scoring='roc_auc')


# In[75]:


dt_clf_knn_best.fit(X_new_train,y_train)


# In[76]:


dt_clf_knn_best.best_params_


# In[78]:


dt_clf_knn = KNeighborsClassifier(algorithm='ball_tree',metric='minkowski',n_neighbors= 9,weights='uniform')


# In[79]:


dt_clf_knn.fit(X_new_train,y_train)
print("Training Score {}".format(dt_clf_knn.score(X_new_train,y_train)))


# In[83]:


#prediction
y_pred_knn = dt_clf_knn.predict(X_new_test)


# In[82]:


pred_df_knn = pd.DataFrame({'Actual_value':y_test, 'Predicted_prob_value':y_pred_knn})
pred_df_knn.head(5)


# In[91]:


#roc-curve
fpr2,tpr2,thresholds2 = roc_curve(pred_df_knn.Actual_value,pred_df_knn.Predicted_prob_value)
roc_auc = auc(fpr2,tpr2)
roc_auc


# In[92]:


plt.figure()
plt.plot(fpr2, tpr2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# #  Using RandomForest

# In[93]:


from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()


# In[100]:


#hypertuning
tuned_param_rf = [{'n_estimators':range(100,105), 'criterion':['gini','entropy'],
                  'max_features':['auto','sqrt','log2']}]
clf_rf_best = GridSearchCV(clf_rf, tuned_param_rf, cv=10, scoring='roc_auc')


# In[101]:


clf_rf_best.fit(X_new_train,y_train)


# In[103]:


clf_rf_best.best_params_


# In[104]:


clf_rf = RandomForestClassifier(criterion='entropy', max_features='log2', n_estimators = 100)


# In[106]:


clf_rf.fit(X_new_train,y_train)
print("Training Score {}".format(clf_rf.score(X_new_train,y_train)))


# In[107]:


#Prediction
y_pred_rf = clf_rf.predict(X_new_test)


# In[108]:


pred_df_rf = pd.DataFrame({'Actual_value':y_test, 'Predicted_prob_value':y_pred_rf})
pred_df_rf.head(5)


# In[109]:


#Evaluation
fpr3,tpr3,thresholds3 = roc_curve(pred_df_rf.Actual_value,pred_df_rf.Predicted_prob_value)
roc_auc = auc(fpr3,tpr3)
roc_auc


# In[110]:


plt.figure()
plt.plot(fpr3, tpr3, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[111]:


print(classification_report(pred_df_rf.Actual_value,pred_df_rf.Predicted_prob_value))


# # # Using Gradient bOosting

# In[112]:


from sklearn.ensemble import GradientBoostingClassifier
clf_gb = GradientBoostingClassifier()


# In[115]:


#hypertuning
tuned_param_gb = [{'loss' : ['deviance', 'exponential'], 'n_estimators':range(100,105) }]


# In[116]:


clf_gb_best = GridSearchCV(clf_gb, tuned_param_gb, cv=10, scoring='roc_auc')


# In[117]:


clf_gb_best.fit(X_new_train,y_train)


# In[118]:


clf_gb_best.best_params_


# In[121]:


clf_gb = GradientBoostingClassifier(loss='deviance', n_estimators=101)
clf_gb.fit(X_new_train,y_train)
print("Training Score {}".format(clf_gb.score(X_new_train,y_train)))


# In[122]:


#Prediction
y_pred_gb = clf_gb.predict(X_new_test)


# In[123]:


pred_df_gb = pd.DataFrame({'Actual_value':y_test, 'Predicted_prob_value':y_pred_gb})
pred_df_gb.head(5)


# In[124]:


#Evalution
fpr4,tpr4,thresholds4 = roc_curve(pred_df_gb.Actual_value,pred_df_gb.Predicted_prob_value)
roc_auc = auc(fpr4,tpr4)
roc_auc


# In[125]:


plt.figure()
plt.plot(fpr4, tpr4, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




