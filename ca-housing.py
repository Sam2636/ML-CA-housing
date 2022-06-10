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
plt.show()

