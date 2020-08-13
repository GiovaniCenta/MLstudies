import Orange
#For this algorithm we will use Orange

base = Orange.data.Table('C:/Users/Cliente/.spyder-py3/ml1/Pre-Processing/risco_credito.csv')
base.domain

cn2_learner = Orange.classification.rules.CN2Learner()    #will generate rules
classifier = cn2_learner(base)                             #usign the rules

for rules in classifier.rule_list:
    print(rules)
    
result = classifier([['boa','alta','nenhuma','acima_35'],['ruim','alta','adequada','0_15']])

for i in result:
    print(base.domain.class_var.values[i])    #will print for each result the variable value
    