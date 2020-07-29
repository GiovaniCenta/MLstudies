
#================== Pre-Processing with census data ==================

import pandas as pd
base = pd.read_csv("census.csv")
#database census doesn't have log errors or outline values, so the pre-processing is just separate predictors and class
predictors = base.iloc[:,0:14].values
classAt = base.iloc[:,14].values

#=========Replacing non numerical values with numerical
#For algorithms to function correctly, we need to encode non numerical labels
from sklearn.preprocessing import LabelEncoder

newWorkClassLabel=LabelEncoder().fit_transform(predictors[:,1]) #this line will change work-class, one string column, to a numerical column
predictors[:,1] = newWorkClassLabel
#now for every other non numerical column
predictors[:,3] = LabelEncoder().fit_transform(predictors[:,3])
predictors[:,5] = LabelEncoder().fit_transform(predictors[:,5])
predictors[:,6] = LabelEncoder().fit_transform(predictors[:,6])
predictors[:,7] = LabelEncoder().fit_transform(predictors[:,7])
predictors[:,8] = LabelEncoder().fit_transform(predictors[:,8])
predictors[:,9] = LabelEncoder().fit_transform(predictors[:,9])
predictors[:,13] = LabelEncoder().fit_transform(predictors[:,13])




#=====IMPORTANT: not all columns will have a specified order. Like race,sex,etc. For our ML algorithm understands that we will need to create "Dummy variables"
#To create dummy variables we use OneHotEncoder
#To make binaries values of those variables we use ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1,3,5,6,7,8,9,13])])
predictors = ct.fit_transform(predictors).toarray()        #"dummy" columns

classAt = LabelEncoder().fit_transform(classAt) #label encode class atribute as well

#======Finally: staggering dataframe 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)
 

