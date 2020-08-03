#==========================================From the book Python Machine Learning, by Sebastian Raschka
#CHAPTER 4
import pandas as pd
import numpy as np
df = pd.read_csv("credit_data.csv")

########################Dropping missing values########################
#one of the simplest ways to deal with missing data is to simply remove columns or rows

#number of missing values
mv=df.isnull().sum()

#Drop row with mv:
df=df.dropna(axis=0)

#Drop column with mv:
df=df.dropna(axis=1)

#only drop rows where all columns are NaN
df=df.dropna(how='all')

#drop rows that have less than 4 real values
df=df.dropna(thresh=4)

#drop rows, in this case, where column = 'age' is NaN
df=df.dropna(subset=['age'])

########################################################################


########################Imputing missing values########################

#Often, the removal of samples or dropping of entire feature columns is simply not feasible, because we might lose too much valuable data. In this case, we can use different interpolation techniques to estimate the missing values from the other training samples in our dataset.

"""
#->>>>>>>>>This was the way to replace missing values with mean,doesnt work anymore
from sklearn.impute import Imputer
imr = Imputer( missing_values =' NaN', strategy =' mean', axis = 0)
mr = imr.fit( df.values)
imputed_data = imr.transform(df.values)
"""
#updated
from sklearn.impute import SimpleImputer
imr = SimpleImputer(missing_values=np.nan, strategy ='mean')
mr = imr.fit( df.values)
df = imr.transform(df.values)
#We can use other strategies, like most_frequent for categorial variables


#####Understandig scikit learn estimator
#Transformer classes in scikit-learn are used for data transformation, the two essential methods are fit and transform
#Fit method learn parameters from the training data and the transofrm method uses those parameters to transform the data


########################Handling categorical data########################
#Difference between nominal and ordinal features:
#One categorial data can be ordinal, like t-shirt size: XL>L>M
#Nominal has no particular order
#Creating an example dataset
import pandas as pd
df2 = pd.DataFrame([[' green', 'M', 10.1, 'class1'],[' red', 'L', 13.5, 'class2'],[' blue', 'XL', 15.3, 'class1']])
df2.columns = ['color', 'size', 'price', 'classlabel']
#color:nominal
#size:ordinal
#price:numerial
#and the class label

#######Handling ordinal features
#to interpret correctly the ordinal freatures, we need to convert the categorical string values into integer
#we dont have one method to automatically order sizes,so, we have to define them manually
size_mapping={'XL':3,'L':2,'M':1}
df2['size'] = df2['size'].map(size_mapping) #This will make the categorical feature('size') turn into an orderned numerical feature

#If we want to transform the integer values back to the original we just make an inverted size mapping
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df2['size'].map(inv_size_mapping)


########Enconding class labels
#Many class labels are encoded as integer values
#Although most estimators for classification in scikit-learn convert class labels to integers internally, it is considered good practice to provide class labels as integer arrays to avoid technical glitches.
#We can also do a maping like before
#But sklearn already has one method that handle this
from sklearn.preprocessing import LabelEncoder
classLEncoder=LabelEncoder()
Y=classLEncoder.fit_transform(df2['classlabel'].values)
#Doing the same with color label
X=df2[['color','size','price']].values
colorLe=LabelEncoder()
X[:,0]=colorLe.fit_transform(X[:,0])

#There is a problem now, we dont want that one color have a big valuer than the other color,bcause they are categorial variables
#To solve this we use one hot encoder
#Binary values will indicate that they are "numerical equal"
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features=[0]) #0 is the column we want to transform
ohe.fit_transform(X).toarray()

#One easier way of doing that is to create dummy features via get_dummiees method
#The get dummiers method will only convert string columns and leave all other
pd.get_dummies(df2[['price','color','size']])
#To reduce the correlation among variables, we can simply remove one feature column from the one-hot encoded array. Note that we do not lose any important information by removing a feature column, though; for example, if we remove the column color_blue, the feature information is still preserved since if we observe color_green = 0 and color_red = 0, it implies that the observation must be blue. If we use the get_dummies function, we can drop the first column by passing a True
pd.get_dummies(df2[['price', 'color', 'size']], drop_first = True)
#The OneHotEncoder does not have a parameter for column removal, but we can simply slice the one-hot encoded NumPy array as shown in the following code
#ohe = OneHotEncoder(categorical_features=[0])ohe.fit_transform(X).toarray()[:, 1:]


##########Partitioning a dataset into separate training and test sets
#Comparing predictions to true label in the test set can be understood as an perfomance evaluation of our model.
#Example with one wine data
import pandas as pd 
df_wine = pd.read_csv('C:/Users/Cliente/.spyder-py3/ml1/Pre-Processing/wine.data',header=None)
df_wine.columns = ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', 'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols', 'Proanthocyanins',  'Color intensity', 'Hue',  'OD280/ OD315 of diluted wines',  'Proline']
#The class label are for different types of grape grown 
#Separating class and redicors labels
predictors,classLabel=df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
#Spliting in train and test 
from sklearn.model_selection import train_test_split
predictors_train,predictors_test,classLabel_train,classLabel_test = train_test_split(predictors, classLabel, test_size=0.3, random_state=0,stratify=classLabel)
#doing this 70% of the samples go to train df's and 30% to test's dataframes
#smaller the test set more inaccurate the esmitamtin of generalization error
#Diving a dataset into training and test sets is all about balancing this trade-off
#most common useds are 60:40,70:30,80:20, depending on the size of the initial dataset
#for larger datasets 90:10 and 99:1 are also common

###########Bringing features to the same scale
#Feature scaling is a crucial step, only in few algorithms can be forgotten, decision trees and random forest are two.
#For a most ML algorithms to work properly we need to scale thigs,we cant have one column with values between 100 and 1000 and other with values between 0 and 1 (of course, if they are both with same importance)
#There are two different common approaches to bring different features into scale
#Normalization and Standardization

######Normalization:Refers to rescalling of the features to a range of [0,1],special case of min max scalling
#to normalize our data we can simply apply the min max scaling to each feature column
#the new value of a sample can be calculated as follows
# x = (x -min(x))/(max(x) - min(x))
from sklearn.preprocessing import MinMaxScaler 
mms = MinMaxScaler()
predictors_train_norm = mms.fit_transform(predictors_train)
predictors_test_norm = mms.transform(predictors_test)
#although normalization via min max scaling is commonly used technique that is useful when we need values in a bounded interval, stardardization can be more practical for manyt ML algorithms, especially for optimization


######Standardization: Using standardization, we center the feature columns at mean 0 with standard deviation 1 so that the feature columns takes the form of a normal distribution, which makes it easier to learn the weights. Furthermore, standardization maintains useful information about outliers and makes the algorithm less sensitive to them in contrast to min-max scaling, which scales the data to a limited range of values.
#allow negative values
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
predictors_train_std = stdsc.fit_transform(predictors_train)
predictors_test_std = stdsc.transform(predictors_test)

######Selecting meaningful features
#if we notice that a model performs much better on a training dataset than on the test dataset, the observation is a strong indicator of overfitting
#Overfitting means the model fits the parameters too closely with regard to the particual observations in the training dataset,but doest not generalize well to new data
#the reason is that our model is too complex for the given training daata
#general solutions: Collect more training data, introduce a penalty for complexity regularization, choose a simpler model with fewer parameters, reduce the dimensionality of the data
#Collecting more training data is often non applicable, further we will learn if more training data is helpful
# L1 l2 processing






#x = (x - mean(x))/standard deviation(x)  




#########Obs: Weka is an interesting software to visualize probabilities tables and such






