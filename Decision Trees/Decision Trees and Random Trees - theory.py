#Sources
#Lior Rokach,Oded Maimon. DATA MINING WITH DECISION TREES: THEORY AND APPLICATIONS (2ND EDITION)
#Hartshorn, Scott. Machine Learning With Random Forests And Decision Trees: A Visual Guide For Beginners (p. 10). Edição do Kindle. 
#Theobald, Oliver. Machine Learning For Absolute Beginners: A Plain English Introduction (Second Edition) (Machine Learning From Scratch Book 1) (p. 96). Scatterplot Press. Edição do Kindle. 

"""
Usage:
->Decision trees are used primarily for solving classification problems but can also be used as a regression model to predict numeric outcomes. Classification trees predict categorical outcomes using numeric and categorical variables as input, whereas regression trees predict numeric outcomes using numeric and categorical variables as input. Decision trees can be applied to a wide range of use cases including picking a scholarship recipient, predicting e-commerce sales, and selecting the right job applicant.

One reason to use:
->Part of the appeal of decision trees is they can be displayed graphically and they are easy to explain to non-experts. 
When a customer queries why they weren’t selected for a home loan, for example, you can share the decision tree to show the decision-making process, which is not possible when using a black-box technique.

Building:
->Decision trees start with a root node that acts as a starting point and is followed by splits that produce branches, also known as edges. The branches then link to leaves, also known as nodes, 
which form decision points. This process is repeated using the data points collected in each new leaf.
A final categorization is produced when a leaf no longer generates any new branches and results in what’s called a terminal node. 
Beginning first at the root node, decision trees analyze data by splitting data into subsets, with a node for each value of the variable (i.e. sunny, overcast, rainy). 
The aim is to keep the tree as small as possible and this is achieved by selecting a variable that optimally splits the data into homogenous groups, such that it minimizes the level of data entropy at thenext branch.
->building a decision tree starts with setting a variable as the root node, with each outcome for that variable assigned a branch to a new decision node,“Yes” and “No.” 
A second variable is then chosen to split the variables further to create new branches and decision nodes. As we want the nodes to collect as many instances of the same class as possible, 
we need to select each variable strategically based on entropy, also called information value. Measured in units called bits (using a base 2 logarithm expression), entropy is calculated based on the composition 
of instances found in each node.

->>>>To calculate entropy -> E(S) = ∑(-p*log2(p))
class label example: high ->6/14 medium ->4/14 low->4/14 
Entropy(S) = -6/14 * log2(6/14) - 4/14* log2(4/14) - 4/14* log2(4/14)
->>>>Info gain
higher the gain more important is the atribute
we calculate each atribute gain
Gain(S,A) = Entropy(S) - ∑ ((Sv/S) * Entropy(Sv)) 
->Explanantion: Entropy(S) : ClassLabel entropy
                Sv/S: column value probability
                Entropy(Sv): column value entropy
->Example: ClassLabel risk: High,Medium,Low
            Atribute Column income:High,Medium,Low

            






#Attributes in vertices
#Each branch -> values
#Class labels -> leafs
#the higher in the tree the greater the importance
#Training in a decision tree is to find the order of importance
"""

############################# RANDOM FORESTS #############################
"""



"The Random Forest is a bit of a Swiss Army Knife of machine learning algorithms.   It can be applied to a wide range of problems, and be fairly good at all of them.  However, it might not be as good as a specialized algorithm for any given specific problem."



"""