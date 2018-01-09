# Titanic
2nd task for the HAIT intern matching event

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
In [3]:

data_raw=pd.read_csv('titanic_train.csv')
In [4]:

data_val=pd.read_csv('titanic_test.csv')
In [5]:

data1=data_raw.copy(deep=True)
In [6]:

data_cleaner=[data1,data_val]
In [7]:

data_raw.sample(10)
Out[7]:
PassengerId    Survived    Pclass    Name    Sex    Age    SibSp    Parch    Ticket    Fare    Cabin    Embarked
870    871    0    3    Balkic, Mr. Cerin    male    26.0    0    0    349248    7.8958    NaN    S
306    307    1    1    Fleming, Miss. Margaret    female    NaN    0    0    17421    110.8833    NaN    C
877    878    0    3    Petroff, Mr. Nedelio    male    19.0    0    0    349212    7.8958    NaN    S
325    326    1    1    Young, Miss. Marie Grice    female    36.0    0    0    PC 17760    135.6333    C32    C
491    492    0    3    Windelov, Mr. Einar    male    21.0    0    0    SOTON/OQ 3101317    7.2500    NaN    S
732    733    0    2    Knight, Mr. Robert J    male    NaN    0    0    239855    0.0000    NaN    S
723    724    0    2    Hodges, Mr. Henry Price    male    50.0    0    0    250643    13.0000    NaN    S
79    80    1    3    Dowdell, Miss. Elizabeth    female    30.0    0    0    364516    12.4750    NaN    S
482    483    0    3    Rouse, Mr. Richard Henry    male    50.0    0    0    A/5 3594    8.0500    NaN    S
599    600    1    1    Duff Gordon, Sir. Cosmo Edmund ("Mr Morgan")    male    49.0    1    0    PC 17485    56.9292    A20    C
In [8]:

data1.isnull().sum()
Out[8]:
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
In [9]:

data_val.isnull().sum()
Out[9]:
PassengerId      0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
In [10]:

for dataset in data_cleaner:
dataset['Age'].fillna(dataset['Age'].median(),inplace=True)
dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)
dataset['Fare'].fillna(dataset['Fare'].median(),inplace=True)
​
​drop_columns=['PassengerId','Ticket','Cabin']
​data1.drop(drop_columns,axis=1,inplace=True)
​In [11]:
​
​data1.isnull().sum()
​Out[11]:
​Survived    0
​Pclass      0
​Name        0
​Sex         0
​Age         0
​SibSp       0
​Parch       0
​Fare        0
​Embarked    0
​dtype: int64
​In [12]:
​
​data_val.isnull().sum()
​Out[12]:
​PassengerId      0
​Pclass           0
​Name             0
​Sex              0
​Age              0
​SibSp            0
​Parch            0
​Ticket           0
​Fare             0
​Cabin          327
​Embarked         0
​dtype: int64
​In [13]:
​
​for dataset in data_cleaner:
​dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1
​dataset['IsAlone']=1
​dataset['IsAlone'].loc[dataset['FamilySize']>1]=0
​dataset['Title']=dataset['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]
​dataset['FareBin']=pd.qcut(dataset['Fare'],4)
​dataset['AgeBin']=pd.cut(dataset['Age'].astype(int),5)
​/Users/yoshizawarikuto/.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/pandas/core/indexing.py:179: SettingWithCopyWarning:
​A value is trying to be set on a copy of a slice from a DataFrame
​
​See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
​self._setitem_with_indexer(indexer, value)
​In [14]:
​
​stat_min=10
​title_names=(data1['Title'].value_counts()<stat_min)
​data1['Title']=data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x]==True else x)
​data1['Title'].value_counts()
​Out[14]:
​Mr        517
​Miss      182
​Mrs       125
​Master     40
​Misc        27
​Name: Title, dtype: int64
​In [15]:
​
​data1_x = ['Sex','Pclass', 'Embarked', 'Title', 'AgeBin', 'FareBin', 'FamilySize', 'IsAlone']
​In [16]:
​
​data1_x_dummy=pd.get_dummies(data1[data1_x])
​In [17]:
​
​data1_x_dummy.head()
​Out[17]:
​Pclass    FamilySize    IsAlone    Sex_female    Sex_male    Embarked_C    Embarked_Q    Embarked_S    Title_ Master    Title_ Miss    ...    Title_Misc    AgeBin_(-0.08, 16.0]    AgeBin_(16.0, 32.0]    AgeBin_(32.0, 48.0]    AgeBin_(48.0, 64.0]    AgeBin_(64.0, 80.0]    FareBin_(-0.001, 7.91]    FareBin_(7.91, 14.454]    FareBin_(14.454, 31.0]    FareBin_(31.0, 512.329]
​0    3    2    0    0    1    0    0    1    0    0    ...    0    0    1    0    0    0    1    0    0    0
​1    1    2    0    1    0    1    0    0    0    0    ...    0    0    0    1    0    0    0    0    0    1
​2    3    1    1    1    0    0    0    1    0    1    ...    0    0    1    0    0    0    0    1    0    0
​3    1    2    0    1    0    0    0    1    0    0    ...    0    0    0    1    0    0    0    0    0    1
​4    3    1    1    0    1    0    0    1    0    0    ...    0    0    0    1    0    0    0    1    0    0
​5 rows × 22 columns
​
​In [49]:
​
​from sklearn.model_selection import StratifiedShuffleSplit
​cv_split=StratifiedShuffleSplit(n_splits=25,test_size=.3,random_state=0)
​from sklearn.ensemble import RandomForestClassifier
​rfc_1=RandomForestClassifier(random_state=0,n_estimators=100)
​from sklearn.model_selection import cross_val_score
​cross_val_score(rfc_1,data1_x_dummy,data1['Survived'],cv=cv_split).mean()
​Out[49]:
​0.82208955223880598
​In [50]:
​
​stat_min2=10
​title_names2=(data_val['Title'].value_counts()<stat_min2)
​data_val['Title']=data_val['Title'].apply(lambda x: 'Misc' if title_names2.loc[x]==True else x)
​data_val['Title'].value_counts()
​Out[50]:
​Mr        240
​Miss       78
​Mrs        72
​Master     21
​Misc         7
​Name: Title, dtype: int64
​In [51]:
​
​data_val_x = ['Sex','Pclass', 'Embarked', 'Title', 'AgeBin', 'FareBin', 'FamilySize', 'IsAlone']
​data_val_x_dummy=pd.get_dummies(data_val[data_val_x])
​In [52]:
​
​predict=
​rfc_1.fit(data1_x_dummy,data1['Survived'])
​predict=rfc_1.predict(data_val_x_dummy)
​In [53]:
​
​df_predict=pd.DataFrame(predict,columns=['Survived'])
​In [54]:
​
​df_predict.head()
​Out[54]:
​Survived
​0    0
​1    1
​2    0
​3    0
​4    0
​In [ ]:
​
​
