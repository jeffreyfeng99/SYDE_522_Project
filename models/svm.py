from sklearn import svm

class SVM_():
    def __init__(self, kernel='rbf', gamma=0.5, c=0.1):
        self.kernel = kernel
        self.gamma = gamma
        self.c = c

        self.svm_ = svm.SVC(kernel=self.kernel, gamma=self.gamma, C=self.c)

    def fit(self, X, y):
        """
        """

        self.svm_.fit(X, y)

    def predict(self, X):
        """
        """
        return self.svm_.predict(X)