################################################### NAIVE BAYES  ###################################################
#Naive bayes is common used in spam filtering, emotion minering, doc separation
#Does well with data in which the inputs are independent from one another
#Prefers oribkens where the probability of any attribute is greater than zero
#To calculate a probability of an error,or a spam in the case above, we need  to use something called conditional probabilities
#is defined as follows: P A B= P(A ∩ B) ,the "and" function
#Bayes formula
#P(H|E) = P(H) * P(E|H)/ P(E)
#P(H) -> probability a hypothesis is true(before any evidence)
#P(E|H) -> probability of seeing the evidence if the hypothesis is true
#P(E) -> probability of seeing the evidence
#P(H|E) -> Probability a hypothesis is true given some evidence
# | represents given
#We assume that no pair of features are dependent 
#Each feature is given the same weight
#Bayes’ Theorem finds the probability of an event occurring given the probability of another event that has already occurred
#Algorithm will do a comparing classLabel with the atributes and will make a table with every probability
#Will multiple every probability from every row and returns a numbers, that will be the probability of that class to happen
#Will sum everything and returns the probability 
#Laplace correction: When we have 0's we add one extra variable, will change whole table , x/n is replaced with x/(n+1)
#Some bayes implementations have the parameter radius: as the name says, will pick one radius of elements and compare themsemvels  

###Some advantages:
#Fast,simple interpretation,high dimensions(high number of atributes/columns)
#Nice predictions in small bases
###Some disvantages
#Considers some atributes independents(sometimes this isnt ideal)

 