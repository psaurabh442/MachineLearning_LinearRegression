### Building Supervised Machine Learning model using Linear Regression

import os

print(os.getcwd())  #prints the current working directory
print(os.listdir()) #prints every file and folder sitting inside that current working directory

import numpy as np      #library for numerical computing
import pandas as pd    # Data Manipulation
import matplotlib.pyplot as plt     #use to actually draw charts and graphs.
import seaborn as sns    # Data Visualization

## load and read the data set
data = pd.read_excel("MachineLearning_RelationalRegression_insurance_data_1500.xlsx")
print(data.info())  #info of columns and rows, data types and memory usage.
print(data.head())  #Check data and display first few
print(data.isna().sum())  #will display all columns which doesn't have any null values.

## Exploratory Data Analysis (EDA)
print("EDA :",data.describe()) ##pandas method that generates summary statistics for every Numeric column in DataFrame

plt.scatter(data['Age'], data['Expenses']) ##draws one dot per row — x-position from data['Age'], y-position from data['Expenses']
plt.xlabel('Age')  # These are for the chart generated
plt.ylabel('Expenses')
plt.title('Age vs Expenses')
plt.show()

data.groupby('Smoker')['Expenses'].mean().plot(kind='bar')  ##data.groupby('Smoker')-splits your 1500 rows into two groups: all rows where Smoker == 'Yes', and all rows where Smoker == 'No'.
#['Expenses'].mean() — within each group, takes just the Expenses column and averages it. Result: a tiny 2-row table (one average for smokers, one for non-smokers).
plt.xlabel('Smoker')  # These are for the chart generated
plt.ylabel('Mean Expenses')
plt.title('Average Expenses by Smoker Status')
plt.show()

#Now Convert all categorical values in columns into numeric value using Label Encoding

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['Sex'] = le.fit_transform(data['Sex'])
data['Region'] = le.fit_transform(data['Region'])
data['Smoker'] = le.fit_transform(data['Smoker'])

print(data.head())


## Dividing data into Dependent and Independent Columns

x = data[['Age','Sex','BMI','Children','Smoker','Region']]
y = data[['Expenses']]

## Spilliting data for Training and Testing Purpose.

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.80) # variables assigned and 0.80 is 805 data for training purpose and 20% data for testing purpose

## Creating a SML model using Linear Regression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train) #Training of the SML takes place on top of 80% of the data

prediction = regressor.predict(x_test) #predicted premium gets generated as o/p as array.
print("Training data prediction is:",prediction)

## Checking for the accuracy of the SML model

from sklearn.metrics import r2_score
score = r2_score(y_test,prediction)
print("Testing Predicted score is:",score)