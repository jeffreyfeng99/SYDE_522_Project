from sklearn.cluster import KMeans


class KMeans_():
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def forward(self, data):
        """
        """
        kmeans = KMeans(n_clusters=self.n_clusters).fit(data)
        return kmeans