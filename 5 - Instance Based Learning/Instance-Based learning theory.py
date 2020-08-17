#SOURCES
#Charu Aggarwal. Data Classification Algorithms and Applications,Chapter 6
"""
->Most classification methods are based on building a model in the training phase, and then using this model for specific test instances, 
during the actual classification phase.

->In instance-based learning, this clean separation between the training and testing phase is usually not present. 
The specific instance, which needs to be classified, is used to create a model that is local to a specific test instance.

->Instance-based learning is also sometimes referred to as lazy learning, since most of the computational work is not done upfront,
and one waits to obtain the test instance, before creating a model for it

->instance-based learning has a different set of tradeoffs, in that it requires very little or no processing for creating a global abstraction of the 
training data, but can sometimes be expensive at classification time. 
This is because instance-based learning typically has to determine the relevant local instances, 
and create a local model from these instances at classification time.

->OBS:While the obvious way to create a local model is to use a k-nearest neighbor classifier, numerous other kinds of lazy solutions are possible, 
which combine the power of lazy learning with other models such as locally-weighted regression, decision trees, rule-based methods, and SVM classifiers
It is possible to use the traditional “eager” learning methods such as Bayes methods, SVM methods , decision trees, 
or neural networks in order to improve the effectiveness of local learning algorithms, 
by applying them only on the local neighborhood of the test instance at classification time.

->The advantage of instance-based learning methods is that they can be used in order to create models that are optimized to specific test instances. 
On the other hand, this can come at a cost, since the computational load of performing the classification can be high. 
As a result, it may often not be possible to create complex models because of the computational requirements.
depends highly upon the data domain, size of the data, data noisiness


"""

#The Nearest Neighbor Classifier
"""
->Most commonly used instance based classifier 

->In this method, the nearest k instances to the test instance are determined. 
Then, a simple model is constructed on this set of k nearest neighbors in order to determine the class label.

->One of the nice characteristics of the nearest neighbor classification approach is that it can be used for practically any data type,
as long as a distance function is available to quantify the distances between objects.
Distance functions are often designed with a specific focus on the classification task. 
Distance function design is a widely studied topic in many domains

->A key issue with the use of nearest neighbor classifiers is the efficiency of the approach in the classification process. 
This is because the retrieval of the k nearest neighbors may require a running time that is linear in the size of the data set.

->With the increase in typical data sizes over the last few years, this continues to be a significant problem . 
Therefore, it is useful to create indexes, which can efficiently retrieve the k nearest neighbors of the underlying data. 
This is generally possible for many data domains, but may not be true of all data domains in general.

#Distance is calculated via euclidian distance
#K is the number of houw much near registers
#knn is very dependent of standarizantions
#Its a powerful and simple algorithm
#Use when the relantioship between caracteristhics are complex
#Small value of K: Outiliers can be a problem
#Big value of K: Can overfit
#Slow to make predictions




"""
