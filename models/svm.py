from sklearn import svm

# TODO grid search?
class SVM_():
    def __init__(self, kernel='rbf', gamma=0.5, c=0.1):
        self.kernel = kernel
        self.gamma = gamma
        self.c = c

    def forward(self, data):
        """
        """
        # TODO: unpack data? in init?

        svm_ = svm.SVC(kernel=self.kernel, gamma=self.gamma, C=self.c).fit(data)
        return svm_