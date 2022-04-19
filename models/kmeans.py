from sklearn.cluster import KMeans


class KMeans_():
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

        self.kmeans = KMeans(n_clusters=self.n_clusters)

    def fit(self, X, y):
        """
        """
        self.kmeans.fit(X, y)

    def predict(self, X):
        """
        """
        return self.kmeans.predict(X)