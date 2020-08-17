
#Sources
# Hastie, Trevor; Tibshirani, Robert; Friedman, Jerome (2008). The Elements of Statistical Learning : Data Mining, Inference, and Prediction
# https://scikit-learn.org/stable/modules/svm.html
# https://en.wikipedia.org/wiki/Support_vector_machine


#########################Support Vector Machines
"""
->One of the most robust prediction methods
->an SVM training algorithm builds a model that assigns new examples to one category or the other,
 making it a non-probabilistic binary linear classifier  
->SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible.
more wide that gap, better the results
->a support-vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection
->a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class (so-called functional margin), since in general the larger the margin, the lower the generalization error of the classifier 

->To work with non linear data we use Kernel Trick: Change the data in one way that a non linear data became linear
->Applications
SVMs are helpful in text and hypertext categorization, classification of images,satelite data
Hand-written characters can be recognized using SVM

->advantages:
Garbage data doesn't influence much
Useful in classification and regression
Easy to use in comparison with neural networks

->disavantages
Need to test with many parameters 
Slow sometimes
Black box method -> cant visualize easily whats happening


    
"""
