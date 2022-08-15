# Topic: SELLER DATA CLUSTERING
Problem Definition:
The dataset has been taken from UCI Machine Learning Library. It contains Facebook Live Seller Data. Based on this data, clustering is to be performed by selecting the basis target variable which will be the underlying rule behind the centroid formation. The data is to be altered to be fit for this type of clustering.
The main idea behind this software is to perform K Means clustering on multi variate data and analyzing the result as well the rand score of the fit K Means model.

# Work Done:
An unsupervised machine learning algorithm - K Means is used to perform the clustering.
Analysis of the dataset is done and  non-relevant features are dropped.
Feature Encoding (Label Encoding) is done on the categorical variables.
Scaling (MinMax Scaling) is done on the numerical variables.
Elbow Method is used to find out the optimal number of clusters.
K Means clustering algorithm is run on the dataset based on the above result. The result of this is added to the main data frame.






# 2. Data set:
The dataset contains information of Facebook Live Sellers in Thailand. 
The data set has been taken from UCI Machine Learning Repository.  The variability of consumer engagement is analysed through a Principal Component Analysis, highlighting the changes induced by the use of Facebook Live. The seasonal component is analysed through a study of the averages of the different engagement metrics for different time-frames (hourly, daily and monthly). Finally, we identify statistical outlier posts, that are qualitatively analyzed further, in terms of their selling approach and activities.
The link is - https://archive.ics.uci.edu/ml/datasets/Facebook+Live+Sellers+in+Thailand
Name of the dataset in the folder – sellers.csv

Descriptives:
•	The dataset consists of 7050 rows and 16 columns of int64, float64 and object type.
•	The columns in the dataset are 'status_type', 'num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys'.
•	The columns ‘status_id’ and 'status_published' are dropped as they have no role.
•	The dataset contains 4 redundant columns (that means it has 4 columns with all null values). These columns are dropped
•	The categorical variable ‘status_type’ is the target variable, based on which the clustering takes place and is used to verify the results.
•	Further descriptives:
	status_id	num_reactions	num_comments	num_shares	num_likes	num_loves	num_wows	num_hahas	num_sads	num_angrys	Column1	Column2	Column3	Column4
count	7050.000000	7050.000000	7050.000000	7050.000000	7050.000000	7050.000000	7050.000000	7050.000000	7050.000000	7050.000000	0.0	0.0	0.0	0.0
mean	3525.500000	230.117163	224.356028	40.022553	215.043121	12.728652	1.289362	0.696454	0.243688	0.113191	NaN	NaN	NaN	NaN
std	2035.304031	462.625309	889.636820	131.599965	449.472357	39.972930	8.719650	3.957183	1.597156	0.726812	NaN	NaN	NaN	NaN
min	1.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	NaN	NaN	NaN	NaN
25%	1763.250000	17.000000	0.000000	0.000000	17.000000	0.000000	0.000000	0.000000	0.000000	0.000000	NaN	NaN	NaN	NaN
50%	3525.500000	59.500000	4.000000	0.000000	58.000000	0.000000	0.000000	0.000000	0.000000	0.000000	NaN	NaN	NaN	NaN
75%	5287.750000	219.000000	23.000000	4.000000	184.750000	3.000000	0.000000	0.000000	0.000000	0.000000	NaN	NaN	NaN	NaN
max	7050.000000	4710.000000	20990.000000	3424.000000	4710.000000	657.000000	278.000000	157.000000	51.000000	31.000000	NaN	Nan	Nan	Nan


# Description of the system:
The project consists of K Means clustering algorithm. It is an unsupervised learning algorithm which is used to find groups in the data (which implies that data in each of the respective groups have similar properties or behaviour. 
Libraries used:
•	Pandas
•	Numpy
•	Sklearn
•	Matplotlib
The main idea of this project is to identify clusters in the data using the K Means clustering algorithm and test its performance by calculating the rand score of the labels and the target variable. As a result, the number of clusters from the target variable is supposed to be to which corresponds to the k value found out as 4 by Elbow method.
# Procedure:
•	The dataset is loaded and analysed by using various functions like describe(), info etc. 
•	The irrelevant columns are dropped. Target Variable is set.
•	As the target variable is categorical, label encoding is done on it. The resultant encoded categorical variables are {0: 'link', 1: 'photo', 2: 'status', 3: 'video'}
•	Feature Scaling (MinMax Scaling) is performed to scale the data  between 0 and 1.
•	Then, to find the optimal number of clusters, Elbow method is used. A graph is plotted with the kmeans score and the number of clusters. The number of clusters corresponding to a kink in the graph is the optimal number. This value is then used for the clustering purpose.
•	Accuracy  is calculated from the original target variable and  the labels formed by the kmeans algorithm.
•	These labels are then assigned to the main dataframe.

# Conclusion:
The rand score of this model is 91%. This can be improved further by applying PCA (principle component analysis) on the dataset and more feature analysis and transformations.

