
import pandas as pd
base = pd.read_csv("C:/Users/Cliente/.spyder-py3/ml1/Pre-Processing/wine.data")
base.columns = ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', 'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols', 'Proanthocyanins',  'Color intensity', 'Hue',  'OD280/ OD315 of diluted wines',  'Proline']
print(len(base.columns))

predictors=base.iloc[:,1:].values
classLabel=base.iloc[:,0].values

#Third Step: Here we will scale everything
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)

#Fourth Step: Here we split in training and test bases 
from sklearn.model_selection import train_test_split
predictors_training, predictors_test, classLabel_training, classLabel_test = train_test_split(predictors, classLabel, test_size=0.25, random_state=0)


###################################################################################################################################################

from sklearn.naive_bayes import GaussianNB
classificator = GaussianNB()
classificator.fit(predictors_training, classLabel_training) #this fit method will generate the probability table
predictions = classificator.predict(predictors_test) 

from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(classLabel_test, predictions) #will compare prredictions with the original classLabel
matrix = confusion_matrix(classLabel_test, predictions) #confusion matrix ->Primary diagonal: correct classification // Other indexes->Incorrect classification, index [0,1] -> correct answer is 0 but classified as 1,etc


#PRECISION:
#With scaler:97,7%
#Without scaler:97,7%