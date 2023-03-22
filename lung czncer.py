#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv('C:/Users/user/Downloads/lung_cancer.csv')
data


# In[22]:


data.shape


# In[5]:


data.head()


# In[6]:


data.describe()


# In[9]:


import seaborn as sns
sns.countplot(df['LUNG_CANCER'])
print(data.LUNG_CANCER.value_counts())


# In[10]:


import matplotlib.pyplot as plt
corr_data = df.corr()
sns.clustermap(corr_data,annot= True,fmt = '.2f')
plt.title('Correlation Between Features')
plt.show();


# In[8]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


le = LabelEncoder()
data.GENDER = le.fit_transform(data.GENDER)
ee=data
ee.head()


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


x=ee.drop(["LUNG_CANCER"],axis=1)
y= ee["LUNG_CANCER"]



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
print("No. of training samples : ",len(x_train))

print("No. of test samples : ",len(x_test))
                                                


# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn import datasets
logreg = LogisticRegression()
data = datasets.load_iris()


# In[ ]:


predicted = cross


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




