

import numpy as np      #library for numerical computing
import pandas as pd    # Data Manipulation
import matplotlib.pyplot as plt     #use to actually draw charts and graphs.
import seaborn as sns    # Data Visualization

## load and read the data set
data = pd.read_excel("MachineLearning_Classification_diabetes_data_500.xlsx")
print(data.info())  #info of columns and rows, data types and memory usage.
print(data.head())  #Check data and display first few
print(data.isna().sum())  #will display all columns which doesn't have any null values.

## Dividing data into Dependent and Independent Columns

x = data[['Pregnancies','Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]

y = data['Outcome'] # the double brackets make y a DataFrame (2D, shape like (500, 1)) instead of a plain 1D Series. sklearn's LogisticRegression expects a 1D target

## Spilliting data for Training and Testing Purpose.

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.85) # variables assigned and 0.80 is 85% data for training purpose and 15% data for testing purpose

## Scaling the features so they're all on a similar scale

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

## Using Algorithm for Binary Classification

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000) #By default Logistic regression does only 100 iteration after that it gives error or warning so better give max iteration number in it.

model.fit(x_train,y_train) #Training of the model takes place
ml_prediction = model.predict(x_test)
print("Outcome prediction :",ml_prediction)

## Checking accuracy

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,ml_prediction)
print("Data Accuracy Score is :",ac)