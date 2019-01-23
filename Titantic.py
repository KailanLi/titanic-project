
# coding: utf-8

# In[25]:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split


# In[2]:

from zipfile import ZipFile

zf = ZipFile('all.zip')
gender = pd.read_csv(zf.open('gender_submission.csv'))
test = pd.read_csv(zf.open('test.csv'))
train = pd.read_csv(zf.open('train.csv'))


# In[3]:

train.info()


# In[4]:

train.isnull().sum()


# In[5]:

sns.set_style("whitegrid")
plt.figure(figsize=[7,5])
sns.heatmap(train.corr())
plt.show()


# In[6]:

graph = sns.FacetGrid(train,col='Pclass',
                      hue='Survived',
                      margin_titles=True)
graph=graph.map(plt.scatter,
                'Fare','Age',
                edgecolor='w',
                alpha=0.8).add_legend()
plt.show()


# In[7]:

ax= sns.boxplot(x="Pclass", y="Age", data=train)
ax= sns.stripplot(x="Pclass", y="Age", data=train, jitter=True, edgecolor="gray")
plt.show()


# Filling the missing value of age

# In[8]:

for i in range(len(train.Age)):
    if pd.isnull(train.iloc[i,5]):
        if train.iloc[i,2]==1:
            train.iloc[i,5]=train['Age'].where(train['Pclass']==1).median()
        if train.iloc[i,2]==2:
            train.iloc[i,5]=train['Age'].where(train['Pclass']==2).median()
        if train.iloc[i,2]==3:
            train.iloc[i,5]=train['Age'].where(train['Pclass']==3).median()
train.isnull().sum()


# In[9]:

for i in range(len(test.Age)):
    if pd.isnull(test.iloc[i,4]):
        if test.iloc[i,1]==1:
            test.iloc[i,4]=test['Age'].where(test['Pclass']==1).median()
        if test.iloc[i,1]==2:
            test.iloc[i,4]=test['Age'].where(test['Pclass']==2).median()
        if test.iloc[i,1]==3:
            test.iloc[i,4]=test['Age'].where(test['Pclass']==3).median()


# In[ ]:




# In[10]:

test.isnull().sum()


# In[ ]:




# In[11]:

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    df.Cabin = df.Cabin.astype('category').cat.codes##category
    return df
def simplify_ages(df):
    df.Age = df.Age.astype('int')
    df.Age = df.Age.astype('float64')
    categories = pd.cut(df.Age,bins=[0, 9, 19, 29, 39, 49, 59, 999],
                    labels=['Child', 'Teenager', 'Young Adult', 
                            'Adult', 'Post Adult', 'Middle Aged', 'Senior'])
    df.Age = categories
    df.Age = df.Age.cat.codes
    return df
def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    df.Fare = df.Fare.astype('category').cat.codes
    return df
def simplify_sex(df):
    df.Sex=df.Sex.apply(lambda x: x[0])
    df.Sex=df.Sex.astype('category').cat.codes
def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.replace(" ", ""))
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    df.Lname=df.Lname.astype('category').cat.codes
    df.NamePrefix=df.NamePrefix.astype('category').cat.codes
    return df 
def drop(df):
    return df.drop(['Ticket','Name','Embarked'],axis=1)
    


# In[12]:

def data_Preprocessing(df):
    df=simplify_cabins(df)
    df=simplify_ages(df)
    df=simplify_fares(df)
    df=simplify_sex(df)
    df=format_name(df)
    df=drop(df)
    return df


# In[13]:

simplify_cabins(train)
simplify_ages(train)
simplify_fares(train)
simplify_sex(train)
format_name(train)
train=drop(train)

simplify_cabins(test)
simplify_ages(test)
simplify_fares(test)
simplify_sex(test)
format_name(test)
test=drop(test)
# In[19]:

train


# Classifiers

# In[26]:

x_all= train.drop(['PassengerId','Survived'],axis=1)
y_all= train['Survived']
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, 
                                                    test_size=0.2,
                                                    random_state=100)


# Random Forest

# In[29]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
RFC=RandomForestClassifier()
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }
acc_score=make_scorer(accuracy_score)

grid_obj=GridSearchCV(RFC,parameters,scoring=acc_score)
grid_obj=grid_obj.fit(X_train,y_train)

RFC=grid_obj.best_estimator_


# In[30]:

RFC.fit(X_train,y_train)


# In[31]:

RFC_prediction=RFC.predict(X_test)
RFC_score = accuracy_score(y_test,RFC_prediction)
print(RFC_score)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



