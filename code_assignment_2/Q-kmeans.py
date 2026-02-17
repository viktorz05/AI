import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def plot_clusters(x_feat, y_feat, labels, clusters, ith_fig, jth_iter):
    plt.scatter(x_feat, y_feat, c=labels, cmap='viridis', label='Data points')

    plt.scatter(clusters[:, 0], clusters[:, 1], s =200, color='red', marker='X', label='Centers')
    plt.title('Clustered data using wo selected features (petal length and width')
    plt.xlabel(iris.feature_names[FEATURE_X_INDEX])
    plt.ylabel(iris.feature_names[FEATURE_Y_INDEX])
    plt.legend()
    plt.show()
    plt.savefig(f'kmeans-{ith_fig}-clusters-{jth_iter}-iters.png')

# 0. Adjust parameters
NUM_CLUSTERS = [2, 3, 4]    # Number of clusters for K-Means (Experiment with 2, 3, 4)
MAX_ITER = [5, 10, 20]         # Maximum number of iterations for the algorithm (Experiment with 5, 10, 20)
FEATURE_X_INDEX = 2    # Index of the feature for the x-axis (0 to 3 for Iris)
FEATURE_Y_INDEX = 3    # Index of the feature for the y-axis (0 to 3 for Iris)

# 1. Import any other required libraries (e.g., numpy, scikit-learn)

# 2. Load the Iris dataset using scikit-learn's load_iris() function
iris = datasets.load_iris()
X = iris.data[:, [FEATURE_X_INDEX, FEATURE_Y_INDEX]]
_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, FEATURE_X_INDEX], iris.data[:, FEATURE_Y_INDEX], c =iris.target) # petal length and width
ax.set(xlabel = iris.feature_names[FEATURE_X_INDEX], ylabel=iris.feature_names[FEATURE_Y_INDEX])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
plt.show()
# 3. Implement K-Means Clustering
    # 3.1. Import KMeans from scikit-learn
    # 3.2. Create an instance of KMeans with the specified number of clusters and max_iter
for cluster in range(3):
    for iter in range(3):
        kmeans = KMeans(n_clusters=NUM_CLUSTERS[cluster], max_iter=MAX_ITER[iter])
            # 3.3. Fit the KMeans model to the data X
        kmeans.fit(X)
            # 3.4. Obtain the cluster labels
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        # 4. Visualize the Results
            # 4.1. Extract the features for visualization
        x_features = iris.data[:, FEATURE_X_INDEX]
        y_features = iris.data[:, FEATURE_Y_INDEX]
            # 4.2. Create a scatter plot of x_feature vs y_feature, colored by the cluster labels
            # 4.3. Use different colors to represent different clusters
        print(f"plotting {cluster} clusters using {iter} iterations...")
        plot_clusters(x_features, y_features, labels, centers, cluster, iter)
    


