"""
=====================================================================
Template with all pre-processing, for more info, go to Pre-Processing files
=====================================================================
"""

import pandas as pd

base = pd.read_csv('census.csv')

predictors = base.iloc[:, 0:14].values
classAt = base.iloc[:, 14].values
                
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],remainder='passthrough')
predictors = column_tranformer.fit_transform(predictors).toarray()

labelencoder_classAt = LabelEncoder()
classAt = labelencoder_classAt.fit_transform(classAt)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)

#Here we split 
from sklearn.model_selection import train_test_split
predictors_training, predictors_test, classAt_training, classAt_test = train_test_split(predictors, classAt, test_size=0.25, random_state=0)
#test_size= % of the base that will be used, test size=25% training size=75%
