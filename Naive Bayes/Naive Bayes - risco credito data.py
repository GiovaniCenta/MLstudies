import pandas as pd

base = pd.read_csv('C:/Users/Cliente/.spyder-py3/ml1/Pre-Processing/risco_credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
                  
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
                 
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)
# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]]) #retorna baixo e moderado 
print(classificador.classes_) #aqui retorna o nome das variaveis das classes
print(classificador.class_count_) #aqui o numero vezes que cada um apararece
print(classificador.class_prior_) #as probabilidades de cada classe