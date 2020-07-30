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

