# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:13:18 2020

@author: Cliente
"""

import pandas as pd
base = pd.read_csv("C:/Users/Cliente/.spyder-py3/ml1/Pre-Processing/credit_data.csv")




#Invalid age replace
base.loc[base.age < 0, 'age'] = base.age.mean()
x=base.age.mean()




#Separating class label and predictors
predictors = base.iloc[:,1:4].values
classLabel = base.iloc[:,4].values

#Input missing data
import numpy as np

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer = imputer.fit(predictors[:,1:4])
predictors[:,1:4] = imputer.transform(predictors[:,1:4])

#Scaling data
from sklearn.preprocessing import StandardScaler
predictors=StandardScaler().fit_transform(predictors) #transforms and fit predictors returning updated df

from sklearn.model_selection import train_test_split
predictors_training, predictors_test, classLabel_training, classLabel_test = train_test_split(predictors, classLabel, test_size=0.25, random_state=0)




###################################### Algorithm ######################################
from sklearn.tree import DecisionTreeClassifier, export
classificator = DecisionTreeClassifier(criterion='entropy')
classificator.fit(predictors, classLabel)
print(classificator.feature_importances_)

export.export_graphviz(classificator,      #generate a tree to visualize further
                       out_file = 'creditdataTree.dot',
                       feature_names = ['income', 'age', 'loan'],
                       class_names = 'classLabel',
                       filled = True,
                       leaves_parallel= True)





#                             Income,   Age,    Loan
pred1=classificator.predict([[50000.00,60.00,10000.00]]) #returns class=0



#Now for whole predictors_test
predictions = classificator.predict(predictors_test) 


from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(classLabel_test, predictions) #will compare prredictions with the original classLabel
matrix = confusion_matrix(classLabel_test, predictions) #confusion matrix ->Primary diagonal: correct classification // Other indexes->Incorrect classification, index [0,1] -> correct answer is 0 but classified as 1,etc



#Precision tests:     
#decision trees doesnt work without SimpleImputer bcause of missing values
#With all pre-processing: 100%
#Without standard scaler 100% (decision trees doenst need scaler)


