# -*- coding: utf-8 -*-

#========================================Pre-Processing========================================#
#to process data of a database via some ML algorithms, you need to convert categorical variables(ex strings) into numeric variables
import pandas as pd
base = pd.read_csv("credit_data.csv") #database into base dataframe



#=====Some dataframe explanantions
#25%,50%,75% -> quartiles
#client_id is a not mesurable variable(nominable variable)
#default is the class atribute(if id is going to afford to pay or not the loan), it's a nominal variable



#======Working with non comparable data

#===log errors(ex: age<0), deleted errors, etc
#base.loc[base["age"]<0] #filtering column age with logs that have age<0
#base.drop('age',1,inplace=True) # Delete whole age column, use it in last cases(to much errors, irreparable)
#base.drop(base[base["age"]<0].index,inplace=True) #will delete only logs with age<0, still not the best
#best method is to replace invalid logs with the mean if it is possible
#ageMean=base["age"][base["age"]>0].mean()  #will return the mean value of all ages>0
#base.loc[base.age<0,'age'] = ageMean #will replace <0 values with the mean  


#====Missing data
#pd.isnull(base.age) #will return true or false if it is NaN value in whole df
#nullValues=base.loc[pd.isnull(base.age)] #will store in nullValues all logs with age=NaN


#====Predictors and class atribute
#to work with some ml algorithms with need to separate predictors and the class atribute
predictors=base.iloc[:,1:4].values #(all rows,(income,age,loan columns (predictors))
classVar=base.iloc[:,4].values #(all rows, (default columns))
import numpy as np
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(predictors[:, 0:3])
predictors[:, 0:3] = imputer.transform(predictors[:,0:3])


#=====Staggering variables
#For ml knn algorithms not consider one variable most important than the other variable
#Also will execute algorithms much faster
#Two ways of doing it -> X is one variable, for example,age
###### Standardisation: 
#x = (x - mean(x))/standard deviation(x)
#can have values<0 
#Stronger when data has to much outliners(higher SD)
######Normalization: 
# x = (x -min(x))/(max(x) - min(x))
# values between 0 and 1

#Doing a test with Standardisation
from sklearn.preprocessing import StandardScaler
predictors=StandardScaler().fit_transform(predictors) #transforms and fit predictors returning updated df







 


