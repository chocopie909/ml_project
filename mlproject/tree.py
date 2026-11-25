"""Numeric binary CART classifier. Exhaustive Gini splits with leaf constraints."""
import numpy as np

def best_split(x, y, min_samples_leaf=1):
    order = np.argsort(x, kind='stable'); x = np.asarray(x)[order]; y = np.asarray(y)[order]
    n = len(y)
    if n < 2*min_samples_leaf:
        return None
    k = np.arange(1, n)
    valid = (x[:-1] < x[1:]) & (k >= min_samples_leaf) & (n-k >= min_samples_leaf)
    if not valid.any():
        return None
    left = np.cumsum(y)[:-1]; right = y.sum()-left
    gini = (k*(2*(left/k)*(1-left/k)) + (n-k)*(2*(right/(n-k))*(1-right/(n-k))))/n
    idx = np.flatnonzero(valid)[np.argmin(gini[valid])]
    return float(x[idx]/2+x[idx+1]/2), float(gini[idx])

class BinaryTree:
    def __init__(self, max_depth=4, min_samples_leaf=5):
        self.max_depth, self.min_samples_leaf = max_depth, min_samples_leaf

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if set(np.unique(y)) - {0, 1}:
            raise ValueError('BinaryTree supports labels 0 and 1')
        if not np.isfinite(X).all() or len(y) == 0:
            raise ValueError('Expected nonempty finite data')
        self.n_features_in_ = X.shape[1]
        self.root_ = self._grow(X, y, 0)
        return self

    def _grow(self, X, y, depth):
        node = {'probability': float(y.mean()), 'size': len(y)}
        parent = 2*y.mean()*(1-y.mean())
        if depth >= self.max_depth or parent == 0:
            return node
        best = None
        for f in range(X.shape[1]):
            candidate = best_split(X[:, f], y, self.min_samples_leaf)
            if candidate and (best is None or candidate[1] < best[2]):
                best = (f, candidate[0], candidate[1])
        if best is None or best[2] >= parent-1e-12:
            return node
        f, threshold, _ = best; mask = X[:, f] <= threshold
        node.update(feature=f, threshold=threshold, left=self._grow(X[mask], y[mask], depth+1),
                    right=self._grow(X[~mask], y[~mask], depth+1))
        return node

    def predict_proba(self, X):
        if not hasattr(self, 'root_'):
            raise ValueError('Fit model first')
        def one(row):
            node = self.root_
            while 'feature' in node:
                node = node['left'] if row[node['feature']] <= node['threshold'] else node['right']
            return node['probability']
        positive = np.array([one(row) for row in np.asarray(X)])
        return np.column_stack([1-positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)
