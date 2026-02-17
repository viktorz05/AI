import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def plot_clusters(labels, num_clusters, max_iter):

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(x_features, y_features, c=labels, cmap='viridis', s=30)
    ax.scatter(centers[:, 0], centers[:, 1], s=200, color='red', marker='X', label='Centers')
    ax.set_title(f'KMeans: {num_clusters} clusters, {max_iter} max_iter')
    ax.set_xlabel(iris.feature_names[FEATURE_X_INDEX])
    ax.set_ylabel(iris.feature_names[FEATURE_Y_INDEX])
    _ = ax.legend(
    sc.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
    out_fname = f'kmeans-{num_clusters}-clusters-{max_iter}-iters.png'
    plt.savefig(out_fname, bbox_inches='tight')
    print(f"Saved plot: {out_fname}")
    plt.show()
    plt.close(fig)

# 0. Adjust parameters
NUM_CLUSTERS = [2, 3, 4]    # Number of clusters for K-Means (Experiment with 2, 3, 4)
MAX_ITER = [5, 10, 20]         # Maximum number of iterations for the algorithm (Experiment with 5, 10, 20)
FEATURE_X_INDEX = 2    # Index of the feature for the x-axis (0 to 3 for Iris)
FEATURE_Y_INDEX = 3    # Index of the feature for the y-axis (0 to 3 for Iris)

# 1. Import any other required libraries (e.g., numpy, scikit-learn)

# 2. Load the Iris dataset using scikit-learn's load_iris() function
iris = datasets.load_iris()
X = iris.data[:, [FEATURE_X_INDEX, FEATURE_Y_INDEX]]
x_features = iris.data[:, FEATURE_X_INDEX]
y_features = iris.data[:, FEATURE_Y_INDEX]

fig_og, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, FEATURE_X_INDEX], iris.data[:, FEATURE_Y_INDEX], c =iris.target) # petal length and width
ax.set(xlabel = iris.feature_names[FEATURE_X_INDEX], ylabel=iris.feature_names[FEATURE_Y_INDEX])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
ax.set_title('Iris dataset')
plt.savefig('Original Iris dataset', bbox_inches='tight')
plt.show()
plt.close(fig_og)
# 3. Implement K-Means Clustering
    # 3.1. Import KMeans from scikit-learn
    # 3.2. Create an instance of KMeans with the specified number of clusters and max_iter
for cluster in NUM_CLUSTERS:
    for iter in MAX_ITER:
        kmeans = KMeans(n_clusters=cluster, max_iter=iter)
            # 3.3. Fit the KMeans model to the data X
        kmeans.fit(X)
            # 3.4. Obtain the cluster labels
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        # 4. Visualize the Results
            # 4.1. Extract the features for visualization

            # 4.2. Create a scatter plot of x_feature vs y_feature, colored by the cluster labels
            # 4.3. Use different colors to represent different clusters
        plot_clusters(labels, cluster, iter)
    


