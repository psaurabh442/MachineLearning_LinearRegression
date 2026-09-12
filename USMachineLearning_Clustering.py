## HealthCare - Diabetes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel("MachineLearning_Classification_diabetes_data_500.xlsx")
print(data.info)
print(data.head())

## Using Scipy_Cluster_Hierarchy for clustering the data with Dendrogram as graphical representation

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
print(dendrogram)
plt.title("Dendrogram for Diabetes Clustering")
plt.show()



#### Clustering for Countries

new_data = pd.read_csv("C:\\Users\\Saurrabh Pandey\\PycharmProjects\\MachineLearning\\USML_countries_Clustering.csv")
print(new_data.head())

#Clustering it with Longitude and Latitude as it is a better representation for grouping on the earth and disacrding language as it is not an explicit way as other languages are also spoken in the countries.

#using slicing so that only 2 columns are used for better view -

my_data = new_data.iloc[:,1:3]
print(my_data.head())

dendrogram = sch.dendrogram(sch.linkage(my_data, method='ward'))
plt.title("Dendrogram for Countries Clustering")
plt.show()