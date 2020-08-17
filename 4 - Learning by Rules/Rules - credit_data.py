import Orange

base = Orange.data.Table('credit_data.csv')
base.domain

base_divided = Orange.evaluation.testing.sample(base, n=0.25)
base_training = base_divided[1]
base_test = base_divided[0]
len(base_training)
len(base_test)

cn2_learner = Orange.classification.rules.CN2Learner()
classificator = cn2_learner(base_training)

for regras in classificator.rule_list:
    print(regras)
    
result = Orange.evaluation.testing.TestOnTestData(base_training, base_test, [classificator])
print(Orange.evaluation.CA(result))

#RESULT:0.488