# -*- coding: utf-8 -*-

import pandas as pd

base=pd.read_csv('census.csv') #saving .csv data do a dataframe("base")

#print(base.describe()) #describe some caracteristics from base, like min,max, mean,etc

#========================================Last,first lines========================================
first8 = base.head(8) #8 first lines
last4=base.tail(4) #4 last lines


#========================================Some selects========================================
selectFirstRow=base[0:1]
selectRows=base[1:4]

selectFirstColumn=base.iloc[:,0:1] #select first column
selectColumns=base.iloc[:,5:8]

selectFirstElement=base.iloc[0]
selectSomeElements=base.iloc[10:50]

selFirstRowColumns = base.iloc[0:1,0:1]
selRowColumns = base.iloc[5:10,3:5]

for col in base.columns:
    print(col) #show columns names



#========================================Some filters========================================
ageLessThanX = base.loc[base['age']<50] #returns database logs that has age<X

maxAge=base.loc[base['age'].max()] #base log with max age
#obs: you can use base.loc[base.age.max()] instead

maxAgeCG=maxAge.loc['capital-gain'] #capital gains of max age

maxAgeEducation=maxAge.loc['education'] #educational level of the older interviewee

maxHWEd = base.loc[base['hour-per-week'].max()].loc['education']
#print(f'Max hours per week education: {maxHWEd}')


minHWEd = base.loc[base['hour-per-week'].min()].loc['education']
#print(f'Min hours per week education: {minHWEd}')

sex_income=base[['sex','income']]  #returns a dataframe with both sex and income only


#filter latinos
latinos=base.loc[base["native-country"].str.contains("Cuba") | base["native-country"].str.contains("Mexico") | base["native-country"].str.contains("Puerto-Rico") | base["native-country"].str.contains("Hounduras")] 
print(f' Latinos avg:\n Hour per week: {latinos["hour-per-week"].mean():.2f} \n Age: {latinos["age"].mean():.2f} \n')

#latinos with hour-per-week<50 and income>50k
latinosIncomeHours=latinos.loc[latinos["hour-per-week"]<50].loc[latinos["income"].str.contains(">50K")]
print(f'Number of latinos with hour-per-week<50 and income>50k: {latinosIncomeHours["age"].count()}')

#filter null age values
nNullAge=base.loc[pd.isnull(base['age'])]



