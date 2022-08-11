import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


def elbow(df):
    cs = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='random', random_state=42)
        kmeans.fit(df)
        cs.append(kmeans.inertia_)

    plt.plot(range(1, 11), cs)
    plt.scatter(3, cs[2], s=200, c='red', marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('CS')
    plt.show()


def cluster_and_visualise(datafilename, K, featurenames):
    # importing data
    labels = np.genfromtxt(datafilename, delimiter=',', usecols=0, dtype=str)
    raw_data = np.genfromtxt(datafilename, delimiter=',')[:, 1:]
    df = pd.DataFrame({label: row for label, row in zip(labels, raw_data)})
    indices = []
    for i in range(len(featurenames)):
        indices.append(np.where(featurenames[i] == labels)[0][0])
    X = df.iloc[:, indices].values

    # getting number of clusters 'K' using elbow method
    # n = elbow(df)

    # running KMeans algorithm with the K passed in argument
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    pred = kmeans.predict(X)

    # Visualizing the clusters
    colours = ['green', 'blue', 'orange']
    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(K):
        ax.scatter(X[pred == i, 0], X[pred == i, 1],
                   c=colours[i], label='Cluster {}'.format(i))
    # Cluster centroids
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
               :, 1],  s=900, c='red', label='Centroid', marker='.')

    ax.set_xlabel(featurenames[0])
    ax.set_ylabel(featurenames[1])
    ax.legend()
    ax.set_title('Cluster Plot')

    # saving the plot as jpg
    plt.savefig('myVisualisation.jpg')

    return fig, ax


# runner
featurenames = ['Year', 'Total']
fig, ax = cluster_and_visualise('data.txt', 3, featurenames)
