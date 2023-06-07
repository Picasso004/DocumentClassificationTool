from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class NaiveBayes:
    def __init__(self):
        self.model = MultinomialNB()

    def train(self, data, target):
        self.model.fit(data, target)

    def predict(self, data):
        return self.model.predict(data)

    def evaluate(self, target,  predictions):
        accuracy = accuracy_score(target, predictions)
        return accuracy


class MinDistance:
    def __init__(self):
        self.model = None
        self.target_labels = None

    def train(self, data, target):
        self.model = NearestNeighbors(n_neighbors=1)
        self.model.fit(data)
        self.target_labels = target

    def predict(self, data):
        if self.model is None or self.target_labels is None:
            raise ValueError("Model has not been trained yet. Please call train() method first.")

        distances, indices = self.model.kneighbors(data)
        predictions = self.target_labels[indices.flatten()]
        return predictions

    def evaluate(self, true_labels, predictions):
        return accuracy_score(true_labels, predictions)


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, data, target):
        """Trains the Random Forest classifier on the training data."""
        self.model.fit(data, target)

    def predict(self, data):
        """Predicts the class labels for the test data."""
        return self.model.predict(data)

    def evaluate(self, target, predictions):
        """Evaluates the model's performance on the test data."""
        accuracy = accuracy_score(target, predictions)
        return accuracy
