# Custom model implementations used in Assignment 2
# These classes mirror the from-scratch implementations in the notebook.

import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            p = self.sigmoid(X @ self.weights + self.bias)
            p_clip = np.clip(p, 1e-7, 1 - 1e-7)

            loss = -np.mean(
                y * np.log(p_clip) +
                (1 - y) * np.log(1 - p_clip)
            )

            error = p - y
            dw = (X.T @ error) / n_samples
            db = np.mean(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            self.loss_history.append(float(loss))

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

logistic = LogisticRegressionScratch(
    learning_rate=0.01,
    n_iterations=1000
)
logistic.fit(X_train_processed, y_train_np)

y_prob_logistic = logistic.predict_proba(X_test_processed)
y_pred_logistic = logistic.predict(X_test_processed)

print("Logistic Regression complete.")
print("Positive predictions:", int(y_pred_logistic.sum()))

plt.figure(figsize=(7, 4))
plt.plot(logistic.loss_history)
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("Logistic Regression Training Loss")
plt.grid(True)
plt.show()


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, probability=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.probability = probability


class DecisionTreeScratch:
    def __init__(self, max_depth=6, min_samples_split=50, max_thresholds=16):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.root = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        p = np.mean(y)
        return 2.0 * p * (1.0 - p)

    def _best_split(self, X, y):
        n, p = X.shape
        parent = self._gini(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for j in range(p):
            values = X[:, j]
            if np.all(values == values[0]):
                continue

            qs = np.linspace(0.05, 0.95, self.max_thresholds)
            thresholds = np.unique(np.quantile(values, qs))

            for threshold in thresholds:
                left = values <= threshold
                nl = int(left.sum())
                nr = n - nl
                if nl == 0 or nr == 0:
                    continue

                yl = y[left]
                yr = y[~left]
                weighted = (nl / n) * self._gini(yl) + (nr / n) * self._gini(yr)
                gain = parent - weighted

                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_threshold = float(threshold)

        return best_feature, best_threshold

    def _leaf(self, y):
        prob = float(np.mean(y))
        value = int(prob >= 0.5)
        return TreeNode(value=value, probability=prob)

    def _build(self, X, y, depth):
        if len(y) == 0:
            return self._leaf(np.array([0]))

        if len(np.unique(y)) == 1:
            return self._leaf(y)

        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return self._leaf(y)

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return self._leaf(y)

        mask = X[:, feature] <= threshold

        return TreeNode(
            feature=feature,
            threshold=threshold,
            left=self._build(X[mask], y[mask], depth + 1),
            right=self._build(X[~mask], y[~mask], depth + 1)
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        self.root = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        while node.value is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def _prob_one(self, x, node):
        while node.value is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.probability

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.array([self._predict_one(x, self.root) for x in X], dtype=int)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.array([self._prob_one(x, self.root) for x in X], dtype=float)

tree = DecisionTreeScratch(
    max_depth=6,
    min_samples_split=50,
    max_thresholds=16
)
tree.fit(X_train_processed, y_train_np)

y_prob_tree = tree.predict_proba(X_test_processed)
y_pred_tree = tree.predict(X_test_processed)

print("Decision Tree complete.")
print("Positive predictions:", int(y_pred_tree.sum()))


class KNNScratch:
    def __init__(self, k=5, batch_size=256):
        self.k = k
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self.train_sq = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y, dtype=np.int8)
        self.train_sq = np.sum(self.X_train ** 2, axis=1)
        return self

    def _predict_probability_batch(self, X_batch):
        X_batch = np.asarray(X_batch, dtype=np.float32)

        # Squared Euclidean distance:
        # ||x||^2 + ||z||^2 - 2*x.z
        x_sq = np.sum(X_batch ** 2, axis=1, keepdims=True)
        distances = (
            x_sq +
            self.train_sq[None, :] -
            2.0 * (X_batch @ self.X_train.T)
        )
        distances = np.maximum(distances, 0.0)

        nearest = np.argpartition(
            distances,
            self.k - 1,
            axis=1
        )[:, :self.k]

        return np.mean(self.y_train[nearest], axis=1)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float32)

        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            out[start:end] = self._predict_probability_batch(X[start:end])

        return out

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

knn = KNNScratch(k=5, batch_size=256)
knn.fit(X_train_processed, y_train_np)

y_prob_knn = knn.predict_proba(X_test_processed)
y_pred_knn = (y_prob_knn >= 0.5).astype(int)

print("kNN complete.")
print("Positive predictions:", int(y_pred_knn.sum()))


class GaussianNaiveBayesScratch:
    def __init__(self):
        self.classes = None
        self.means = None
        self.variances = None
        self.priors = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.means = np.zeros((n_classes, n_features), dtype=np.float32)
        self.variances = np.zeros((n_classes, n_features), dtype=np.float32)
        self.priors = np.zeros(n_classes, dtype=np.float32)

        for i, c in enumerate(self.classes):
            Xc = X[y == c]
            self.priors[i] = len(Xc) / len(X)
            self.means[i] = np.mean(Xc, axis=0)
            self.variances[i] = np.var(Xc, axis=0) + 1e-6

        return self

    def _log_scores(self, X):
        X = np.asarray(X, dtype=np.float32)
        scores = []

        for i in range(len(self.classes)):
            mean = self.means[i]
            var = self.variances[i]

            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * var) +
                ((X - mean) ** 2) / var,
                axis=1
            )

            scores.append(
                np.log(self.priors[i]) + log_likelihood
            )

        return np.vstack(scores).T

    def predict_proba(self, X):
        scores = self._log_scores(X)
        scores -= np.max(scores, axis=1, keepdims=True)
        probs = np.exp(scores)
        return probs / np.sum(probs, axis=1, keepdims=True)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

nb = GaussianNaiveBayesScratch()
nb.fit(X_train_processed, y_train_np)

nb_proba = nb.predict_proba(X_test_processed)
y_prob_nb = nb_proba[:, 1]
y_pred_nb = nb.predict(X_test_processed)

print("Naive Bayes complete.")
print("Positive predictions:", int(y_pred_nb.sum()))


class RandomForestScratch:
    def __init__(
        self,
        n_trees=10,
        max_depth=6,
        min_samples_split=50,
        max_features="sqrt",
        max_thresholds=12,
        random_state=42
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.random_state = random_state
        self.trees = []
        self.feature_subsets = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape
        if self.max_features == "sqrt":
            n_selected = max(1, int(np.sqrt(n_features)))
        else:
            n_selected = min(int(self.max_features), n_features)

        class0 = np.where(y == 0)[0]
        class1 = np.where(y == 1)[0]
        n_each = min(len(class0), len(class1))

        self.trees = []
        self.feature_subsets = []

        for t in range(self.n_trees):
            # Balanced bootstrap
            idx0 = rng.choice(class0, size=n_each, replace=True)
            idx1 = rng.choice(class1, size=n_each, replace=True)
            indices = np.concatenate([idx0, idx1])
            rng.shuffle(indices)

            features = np.sort(
                rng.choice(
                    n_features,
                    size=n_selected,
                    replace=False
                )
            )

            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_thresholds=self.max_thresholds
            )

            tree.fit(X[indices][:, features], y[indices])

            self.trees.append(tree)
            self.feature_subsets.append(features)

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        all_probs = []

        for tree, features in zip(self.trees, self.feature_subsets):
            all_probs.append(
                tree.predict_proba(X[:, features])
            )

        return np.mean(np.vstack(all_probs), axis=0)

    def predict(self, X, threshold=0.5):
        return (
            self.predict_proba(X) >= threshold
        ).astype(int)

rf = RandomForestScratch(
    n_trees=10,
    max_depth=6,
    min_samples_split=50,
    max_features="sqrt",
    max_thresholds=12,
    random_state=42
)

rf.fit(X_train_processed, y_train_np)

y_prob_rf = rf.predict_proba(X_test_processed)
y_pred_rf = (y_prob_rf >= 0.5).astype(int)

print("Random Forest complete.")
print("Positive predictions:", int(y_pred_rf.sum()))
