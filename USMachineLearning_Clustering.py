## HealthCare - Diabetes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster

data = pd.read_excel("MachineLearning_Classification_diabetes_data_500.xlsx")
print(data.info)
print(data.head())

## Using Scipy_Cluster_Hierarchy for clustering the data with Dendrogram as graphical representation

# Keep only numeric columns for clustering (drop any ID/label columns)
data_numeric = data.select_dtypes(include=[np.number])

Z = sch.linkage(data_numeric, method='ward')

plt.figure(figsize=(10, 6))
sch.dendrogram(Z)
plt.axhline(y=150, color='r', linestyle='--')  # adjust y to match where your dendrogram splits into 2
plt.title("Dendrogram for Diabetes Clustering (2-cluster cut)")
plt.show()

# Assign each point to one of 2 clusters based on that cut
clusters = fcluster(Z, t=2, criterion='maxclust')
print(pd.Series(clusters).value_counts())


#### Clustering for Countries

new_data = pd.read_csv(r"C:\Users\Saurrabh Pandey\PycharmProjects\MachineLearning\USML_countries_Clustering.csv")
print(new_data.head())

my_data = new_data.iloc[:, 1:3]
print(my_data.head())

plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(my_data, method='ward'))
plt.title("Dendrogram for Countries Clustering")
plt.show()