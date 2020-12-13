
import pandas as pd
import numpy as np
pd.pandas.set_option('display.max_columns',None)
#data loading
data=pd.read_csv('admission_predict.csv.txt')
print(data.head())

#Check any missing values
print(data.isnull().sum())

#Drop the unnesscary column 'Serila No.'
data=data.drop(labels=['Serial No.'],axis=1)

#Rename the columns
data.rename(columns={'GRE Score':'GRE_Score','TOEFL Score':'TOEFL_Score','University Rating':'University_Rating',
                     'Chance of Admit ':'Chance_of_Admit'},inplace=True)
print(data.columns)

#split the data in to 'x' and 'y'
x=data.drop('Chance_of_Admit',axis=1)
y=data['Chance_of_Admit']

#splitting in to train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=42)

#scale the model
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#x_train=sc.fit_transform(x_train)
#x_test=sc.transform((x_test))

#Model building
from sklearn.linear_model import  LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

#predict the test model
model_pred=model.predict(x_test)
print(model_pred[:9])

#get the score on the test data
from sklearn.metrics import r2_score,classification_report
print(r2_score(y_test,model_pred))

import pickle

filename='Linear_model.pickle'
pickle.dump(model,open(filename, 'wb'))

