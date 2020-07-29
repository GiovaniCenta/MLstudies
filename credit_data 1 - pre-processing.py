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
ageMean=base["age"][base["age"]>0].mean()  #will return the mean value of all ages>0
base.loc[base.age<0,'age'] = ageMean #will replace <0 values with the mean  


#====Missing data
pd.isnull(base.age) #will return true or false if it is NaN value in whole df
nullValues=base.loc[pd.isnull(base.age)] #will store in nullValues all logs with age=NaN


#====Predictors and class atribute
#to work with some ml algorithms with need to separate predictors and the class atribute
predictors=base.iloc[:,1:4] #(all rows,(income,age,loan columns (predictors))
classVar=base.iloc[:,4] #(all rows, (default columns))
from sklearn.impute import SimpleImputer
import numpy as np
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #will imput all missing values with mean
imputer = imputer.fit(predictors[:, 0:3]) #process to fit the imputer on predictors.
predictors[:, 0:3] = imputer.transform(predictors[:,0:3]) #finally predictors will be updated w/o NaN via transform

#=====Staggering variables



 


