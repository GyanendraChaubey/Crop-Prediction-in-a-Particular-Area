# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 22:30:37 2019

@author: Gyanendra
"""

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
#from sklearn import tree
from sklearn.preprocessing import LabelEncoder

#Read the dataset
data=pd.read_csv('crop.csv')

X=data[['Location','Avg_Soil_Type','Month']]
Y=data[['Crop_production_1','Crop_production_2']]


#Label encoder to cover the categorical values into the numeric
number = LabelEncoder()
X['Location']=number.fit_transform(X['Location'].astype('str'))

X['Avg_Soil_Type']=number.fit_transform(X['Avg_Soil_Type'].astype('str'))
X['Month']=number.fit_transform(X['Month'].astype('str'))
Y['Crop_production_1'] = number.fit_transform(Y['Crop_production_1'].astype('str'))
Y['Crop_production_2'] = number.fit_transform(Y['Crop_production_2'].astype('str'))

#Create the object of decision tree classifier
tree=DecisionTreeClassifier()
#fit the model
tree.fit(X,Y)


#Taking input from the user
p='Banda'
q='Jan'
r='Black'

#Conversion of new input to numeric form
t=X['Location']['Banda']
u=X['Month']['Jan']
v=X['Avg_Soil_Type']['Black']
#New X values to predict
"""
X_new=[[t,u,v]]

#prediction of output
predict1=tree.predict(X_new)
print(predict1)
#Graph Prediction using the predicted values
X_year=[2016,2017,2018]
y_cost_Crop1=[predict1[0][0]+2,predict1[0][0]+3,predict1[0][0]+4]
y_selling_price_crop1=[predict1[0][0]+5,predict1[0][0]+6,predict1[0][0]+7]
#Predict the selling_price_crop1
plt.bar(X_year, y_selling_price_crop1, color='green')
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Selling Price of Crop1 in past three years')
plt.show()
#Predict the cost_crop1
plt.bar(X_year, y_cost_Crop1, color='green')
plt.xlabel('Year')
plt.ylabel('Cost')
plt.title('Cost/Expenditure of Crop1 in past three years')
plt.show()

y_cost_Crop2=[predict1[0][1]+8,predict1[0][1]+9,predict1[0][1]+10]
y_selling_price_crop2=[predict1[0][1]+11,predict1[0][1]+12,predict1[0][1]+13]
#Predict the selling_price_crop2
plt.bar(X_year, y_selling_price_crop2, color='green')
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Selling Price of Crop2 in past three years')
plt.show()
#Predict the cost_crop2
plt.bar(X_year, y_cost_Crop2, color='green')
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Cost/Expenditure of Crop2 in past three years')
plt.show()

#Change the predicted values into categorical form
change=number.inverse_transform(predict1).split()
print(change)


"""

