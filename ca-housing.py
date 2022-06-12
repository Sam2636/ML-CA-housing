#from statistics import linear_regression
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson

#preprocessing

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer

#read data
df=pd.read_csv('housing.csv')
print(df.head())
print(df.shape)
print(df.sample(5))

print(df.dtypes)

#null values

print(df.isnull().sum())

#basic stats
print(df.describe())
print(df.hist(figsize=(16,9)))
fig,axis=plt.subplots(1,7,figsize=(16,6))

sns.boxplot(data=df[['housing_median_age']],ax=axis[0]);
sns.boxplot(data=df[['total_rooms']],ax=axis[1]);
sns.boxplot(data=df[['total_bedrooms']],ax=axis[2]);
sns.boxplot(data=df[['population']],ax=axis[3]);
sns.boxplot(data=df[['households']],ax=axis[4]);
sns.boxplot(data=df[['median_income']],ax=axis[5]);
sns.boxplot(data=df[['median_house_value']],ax=axis[6]);
#plt.show()#-------->to show the plot



#handling null values
print(df.isnull().sum())

#to identify the columns with null values
#plt.figure(figsize=(16,5))

#plt.scatter(df['latitude'],df['longitude'],c='green',s=6)
data_null=df[df['total_bedrooms'].isnull()]
data_null.shape

plt.figure(figsize=(16,9))
plt.scatter(df['latitude'],df['longitude'],c='green',s=6)
plt.scatter(data_null['latitude'],data_null['longitude'],c='red',s=6)

#observation to drop the rows where the total bedroom is null
#take the mean/median value of the total bedroom

df['total_bedrooms'].fillna(df['total_bedrooms'].median(),inplace=True)
print(df.corr())

plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn')


#handling categorical data
df.ocean_proximity.value_counts()

#visulaization(occurence of ocean proximity)
plt.figure(figsize=(12,6))
sns.boxplot(data=df,x='ocean_proximity',y='median_house_value');
#plt.show()

#encode categorical data

df['ocean_proximity'].replace({"INLAND":0,
                                "<1H OCEAN":1,
                               "NEAR OCEAN":2,
                               "NEAR BAY":3,        
                               "ISLAND":4},inplace=True)

#####data split
#train test split

x=df.drop(['median_house_value'],axis=1).values
y=df['median_house_value'].values

from sklearn.model_selection import train_test_split

#split the data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

###model building
#**KNN REGRESSOR**

from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor

knn=KNeighborsRegressor(n_neighbors=11,metric='minkowski',p=2,weights='uniform')

#fit the model to the training data
knn.fit(x_train,y_train)

#making prediction on the testing set
y_pred=knn.predict(x_test)

print("---------------------knn---------------------")

print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('R2 Score:',metrics.r2_score(y_test,y_pred))

#print(df.describe().T)

#------------------------------------------------knn-scaling--------------------------------------------------
#lets scale this
scaler=RobustScaler()

x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

knn=KNeighborsRegressor(n_neighbors=11,metric='minkowski',p=2,weights='uniform')

#fit the model to the training data
knn.fit(x_train_scaled,y_train)

#making prediction on the testing set
y_pred=knn.predict(x_test_scaled)

print("---------------------knn-scaling---------------------")

print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('R2 Score:',metrics.r2_score(y_test,y_pred))


#------------------linear regression--------------------
from sklearn.linear_model import LinearRegression

lR=LinearRegression()

#fit the model to the training data
lR.fit(x_train,y_train)

#making prediction on the testing set
y_pred=lR.predict(x_test)
print("--------------linear regression--------------------")

print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("MAPE:",np.mean(abs((y_test-y_pred)/y_test))*100)
print("MPE:",np.mean(y_test-y_pred/y_test)*100)
print('R2 Score:',metrics.r2_score(y_test,y_pred))

#----------------scaling linear regression--------------------

from sklearn.linear_model import LinearRegression

lR=LinearRegression()

#fit the model to the training data
lR.fit(x_train_scaled,y_train)

#making prediction on the testing set
y_pred=lR.predict(x_test_scaled)

print("-------------scaling-linear-regression----------------")

print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("MAPE:",np.mean(abs((y_test-y_pred)/y_test))*100)
print("MPE:",np.mean(y_test-y_pred/y_test)*100)
print('R2 Score:',metrics.r2_score(y_test,y_pred))
