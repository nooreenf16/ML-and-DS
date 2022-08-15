# Topic: AGE RANGE OF CUSTOMERS
Problem Definition:
 A dataset is taken from Kaggle which contains mall customer information. The main focus is on the Annual Income k($) and Spending Score (1 - 100) columns. Given these data on these two features, the Age range of the customers is to be produced as output.
The main idea of this software is to find out the age range so that age-specific marketing strategies can be adopted, taking the pricing and other monetary values as parameters.

Work Done:
An unsupervised machine learning algorithm - K Means is used to perform the clustering.
Analysis of the dataset is done to figure out the best features for the clustering. 
Elbow Method is used to find out the optimal number of clusters.
K Means algorithm is run on the dataset based on the above result. The result of this is added to the main data frame.
An application of the above work is coded, that is, to find the age range of the customers to which the user input data belongs to.


# 2. Dataset:
The dataset consists of mall customer information.
The dataset has been taken from Kaggle. It consists of information of customers who spend in malls. It mainly focuses on the monetary information of the customers and aims at evolving the marketing strategy and identify the target audience characteristics like age group, gender etc.
The link is: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python?resource=download
Name of the dataset in the folder – Mall_Customer.csv

Descriptives:
•	The dataset  contains 200 rows and 6 columns.
•	The columns present are 'CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'
•	All columns except ‘Gender’ are of numeric data type.
•	Only 'Annual Income (k$)' and  'Spending Score (1-100)' columns are used in the KMeans algorithm.
•	There are no null valued columns or rows in the entire dataset.
•	The scatter plot of the data distinctly shows 5 clusters, but this is a human intuition.
•	Further descriptives:

	CustomerID	Age	Annual Income (k$)	Spending Score (1-100)
count	200.000000	200.000000	200.000000	200.000000
mean	100.500000	38.850000	60.560000	50.200000
std	57.879185	13.969007	26.264721	25.823522
min	1.000000	18.000000	15.000000	1.000000
25%	50.750000	28.750000	41.500000	34.750000
50%	100.500000	36.000000	61.500000	50.000000
75%	150.250000	49.000000	78.000000	73.000000
max	200.000000	70.000000	137.000000	99.000000

# Description of the system:
The project consists of K Means clustering algorithm. It is an unsupervised learning algorithm which is used to find groups in the data (which implies that data in each of the respective groups have similar properties or behaviour). 
Libraries used:
•	Pandas
•	Numpy
•	Sklearn
•	Matplotlib
The main idea of this project is to identify clusters in the data using the K Means clustering algorithm and use these clusters to perform various other analytics and tasks. The clusters can be visually seen by the scatterplot but this human intuition can be verified by the algorithm. The number of clusters in the dataset is found out by Elbow Method. The clusters thus formed give rise to labels which are then assigned to the main data frame. The application takes a 2D array with data for Annual Income and Spending Score to determine the Age range of the cluster to which that  particular data belongs to.

# Procedure:
•	The dataset is loaded and analysed by using various functions like describe(), info etc. 
•	A raw scatterplot of the dataset is plotted.
•	A new data variable, a 2D array,  ‘X’ is initialized with the columns 'Annual Income (k$)' and  'Spending Score (1-100)' .
•	Then, to find the optimal number of clusters, Elbow method is used. A graph is plotted with the kmeans inertia and the number of clusters. The number of clusters corresponding to a kink in the graph is the optimal number. This value is then used for the clustering purpose.
•	Cluster centers are computed and cluster index is predicted for each sample. Since K=5 we will get the cluster index from 0 to 4 for every data point in our dataset.
•	The kmeans labels are then assigned to the main dataframe.
•	A labelled and coloured scatterplot of the grouped data and their centroid is plotted to visually identify the clusters.
•	Application of the above clustering is coded. It outputs the highest and lowest Age of the cluster to which the input data belongs to.

# Conclusion: 
Customers (data) are grouped based on their properties which can help better identify and predict their behaviours. The dataset does not have any labels associated with it, but still it can be split and grouped based on underlying patterns. An application of the clustering done is made for it to work for user input.
