import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

# Import necessary libraries
import matplotlib.pyplot as plt

# Generate synthetic data for clustering
# Here we create a dataset with 3 clusters for demonstration purposes
n_samples = 300
n_features = 2
n_clusters = 3
random_state = 42
data, labels_true = make_blobs(n_samples=n_samples, 
                               n_features=n_features, 
                               centers=n_clusters, 
                               cluster_std=1.0, 
                               random_state=random_state)

# Visualize the generated data
plt.scatter(data[:, 0], data[:, 1], s=30, c='gray', marker='o', label='Data points')
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Apply K-Means clustering
# Initialize the KMeans model with the desired number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

# Fit the model to the data and predict cluster labels
predicted_labels = kmeans.fit_predict(data)

# Visualize the clustering results
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis', s=30, marker='o', label='Clustered points')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title("Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Print the cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Evaluate the clustering performance (optional)
# Note: This requires ground truth labels, which we have in this synthetic example
ari_score = adjusted_rand_score(labels_true, predicted_labels)
print(f"Adjusted Rand Index (ARI): {ari_score:.2f}")