import pandas as pd
base = pd.read_csv("C:/Users/Cliente/.spyder-py3/ml1/Pre-Processing/census.csv")

################################################################# Pre-Processing ##################################################################################################################################
#First Step: Separating class label and predictors
predictors = base.iloc[:,0:14].values
classLabel= base.iloc[:,14].values

#Second Step: Enconding non numerical labels
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


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

onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
predictors = onehotencorder.fit_transform(predictors).toarray()

labelencoder_classLabel = LabelEncoder()
classLabel = labelencoder_classLabel.fit_transform(classLabel)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)


#Fourth Step: Here we split in training and test bases 
from sklearn.model_selection import train_test_split
predictors_training, predictors_test, classLabel_training, classLabel_test = train_test_split(predictors, classLabel, test_size=0.15, random_state=0)

###################################################################################################################################################################################################
import time
beginning = time.time()


from sklearn.neighbors import KNeighborsClassifier
classificator = KNeighborsClassifier(n_neighbors=8, metric='minkowski', p = 2)    #metric = minkowski and p=2 = euclidian distance
classificator.fit(predictors_training, classLabel_training)
predictions = classificator.predict(predictors_test)

from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(classLabel_test, predictions)
matrix = confusion_matrix(classLabel_test, predictions)

import collections
collections.Counter(classLabel_test)




end = time.time()
runningTime = end - beginning


#Predictions Results neighbours = 5
#With all pre processing: 82,2%     //k = 10: 82,9%   //k = 30: 82,38% //k = 15:82,84% 
#Without OHE: 82,19%
#Without scaler:77,60%
#Label encoder only:77,46%

