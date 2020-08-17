#SOURCES
#https://christophm.github.io/interpretable-ml-book/rules.html

"""
"A decision rule is a simple IF-THEN statement consisting of a condition (also called antecedent) and a prediction. For example:
IF it rains today AND if it is April (condition), THEN it will rain tomorrow (prediction). A single decision rule or a combination of several rules can be used to make predictions.
Decision rules follow a general structure: IF the conditions are met THEN make a certain prediction. Decision rules are probably the most interpretable prediction model"
"""


############################################################ OneR algorithm (Learn Rules from a Single Feature) ############################################################

"""

-> Commonly used to test simple things
-> "One feature does the work"

-> Characterized by its simplicity, interpretability and its use as a benchmark.
-> Selects the one that carries the most information about the outcome of interest and creates decision rules from this feature.
-> OneR model is a decision tree with only one split.
-> How the best feature is chosen by OneR?
OneR creates the cross tables between each feature and the outcome
For each feature, we go through the table row by row: Each feature value is the IF-part of a rule; the most common class for instances with this feature value is the prediction, the THEN-part of the rule.
Doing that we will make an error because not every feature have that value
For each feature we calculate the total error rate of the generated rules, which is the sum of the errors
We will use the feature with less errors
-> OneR prefers features with many possible levels, because those features can overfit the target more easily. 
Imagine a dataset that contains only noise and no signal, which means that all features take on random values and have no predictive value for the target. 
Some features have more levels than others. The features with more levels can now more easily overfit. 
A feature that has a separate level for each instance from the data would perfectly predict the entire training dataset. 
A solution would be to split the data into training and validation sets, learn the rules on the training data and evaluate the total error for choosing the feature on the validation set.
-> Ties are another issue, i.e. when two features result in the same total error. OneR solves ties by either taking the first feature with the lowest error or the one with the lowest p-value of a chi-squared test.
->OneR does not support regression tasks. But we can turn a regression task into a classification task by cutting the continuous outcome into intervals.

"""


############################################################ Sequential Covering ############################################################
"""

"""





##################### Trees vs rules
"""
->Generally rules dont gives better results than decision trees
->Rules are slower than decision trees
->Both algorithm occupies low storage space
""" 