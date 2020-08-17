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


##################################################################Algorithm


from sklearn.neighbors import KNeighborsClassifier
classificator = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)    #metric = minkowski and p=2 = euclidian distance
classificator.fit(predictors_training, classLabel_training)
predictions = classificator.predict(predictors_test)

from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(classLabel_test, predictions)
matrix = confusion_matrix(classLabel_test, predictions)

import collections
collections.Counter(classLabel_test)


#Predictions neighbors = 5
#With all pre processing = 98,6 %  [BEST RESULT]
#Without scaler = 83,6%

#Predictions neighbors = 10
#With all pre processing = 97,4 %
#Without scaler = 84,6%

#Predictions neighbors = 2
#With all pre processing = 97,2 %
#Without scaler = 85,4%