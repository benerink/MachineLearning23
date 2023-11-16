from sklearn.tree import DecisionTreeClassifier


class DecisionTree:

    def __init__(self, min_samples_split=10, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.model = DecisionTreeClassifier(
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            max_features=n_features
        )

    def fit(self, X, y):
        print("Decision Tree: Starting fit.")
        self.model.fit(X, y)
        print("Decision Tree: Fit completed.")

    def predict(self, X):
        return self.model.predict(X)
