#SOURCES
#
#https://en.wikipedia.org/wiki/Gradient_descent
#Applied logistic regression,David W. Hosmer

"""
-> Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable
-> Is one classification algorithm, doesn't has anything to do with regression algorithms
-> The goal is to "drawn" a function to describe most correctly possible the data ->drawns a sigmoid function


This function is based on the linear regression function 
->first thing to do is to use the line equation 
y = b0 + b1*x   
->after that we use sigmoid function
p=1/(1+e^(-y))   \\values between 0 and 1
->after we use logit transformation
log(p/(1-p)) = b0 + b1*x

->We will find b0 and after that b1's for each atribute
mostly of the times with the gradient descent algorithm, one algorithm that uses the mimimal error with calculus concepts

y axis -> probability of yes
x axis -> atribute


 










"""