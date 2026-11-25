"""Binary logistic gradient boosting using sklearn regression trees as weak learners.
Labels remain 0/1. Shrinkage is applied consistently during fit and inference.
"""
import numpy as np
from scipy.special import expit, logit
from sklearn.tree import DecisionTreeRegressor

def logistic_loss(y, scores):
    return float(np.mean(np.logaddexp(0, scores)-y*scores))

class LogisticBoosting:
    def __init__(self, n_estimators=150, learning_rate=.1, max_depth=2,
                 min_samples_leaf=5, patience=15, random_state=42):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth
        self.min_samples_leaf, self.patience, self.random_state = min_samples_leaf, patience, random_state

    def fit(self, X, y, eval_set=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError('Both classes 0 and 1 are required')
        if self.learning_rate <= 0 or self.n_estimators < 1:
            raise ValueError('Positive learning rate and estimator count required')
        self.init_score_ = float(logit(np.clip(y.mean(), 1e-8, 1-1e-8)))
        self.models_ = []; self.history_ = []
        scores = np.full(len(y), self.init_score_)
        if eval_set is not None:
            Xv, yv = np.asarray(eval_set[0]), np.asarray(eval_set[1])
            val_scores = np.full(len(yv), self.init_score_)
            best_loss = logistic_loss(yv, val_scores); best_count = 0; stalled = 0
        self.initial_loss_ = logistic_loss(y, scores)
        for i in range(self.n_estimators):
            residual = y-expit(scores)
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf, random_state=self.random_state+i)
            tree.fit(X, residual)
            self.models_.append(tree)
            scores += self.learning_rate*tree.predict(X)
            item = {'iteration': i+1, 'train_loss': logistic_loss(y, scores)}
            if eval_set is not None:
                val_scores += self.learning_rate*tree.predict(Xv)
                item['val_loss'] = logistic_loss(yv, val_scores)
                if item['val_loss'] < best_loss-1e-8:
                    best_loss, best_count, stalled = item['val_loss'], len(self.models_), 0
                else:
                    stalled += 1
            self.history_.append(item)
            if eval_set is not None and self.patience and stalled >= self.patience:
                break
        if eval_set is not None:
            self.models_ = self.models_[:best_count]
        self.best_iteration_ = len(self.models_)
        return self

    def decision_function(self, X):
        if not hasattr(self, 'models_'):
            raise ValueError('Fit model first')
        X = np.asarray(X, dtype=float)
        return np.full(len(X), self.init_score_) + sum(
            (self.learning_rate*m.predict(X) for m in self.models_), start=np.zeros(len(X)))

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1-p, p])

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)
