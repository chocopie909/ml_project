"""Linear regression with explicit GD, minibatch SGD, momentum and Adam."""
import numpy as np

def loss_gradient(X, y, w, loss='mse', l2=0.0, delta=1.0):
    residual = X @ w - y
    if loss == 'mse':
        value = np.mean(residual**2)
        derivative = 2 * residual
    elif loss == 'mae':
        value = np.mean(np.abs(residual))
        derivative = np.sign(residual)
    elif loss == 'huber':
        small = np.abs(residual) <= delta
        value = np.mean(np.where(small, .5*residual**2, delta*(np.abs(residual)-.5*delta)))
        derivative = np.clip(residual, -delta, delta)
    elif loss == 'logcosh':
        value = np.mean(np.logaddexp(residual, -residual) - np.log(2))
        derivative = np.tanh(residual)
    else:
        raise ValueError('Unknown loss')
    penalty = w.copy()
    penalty[-1] = 0  # intercept is last column and is not penalized
    return float(value + .5*l2*(penalty@penalty)), X.T@derivative/len(y) + l2*penalty

class GradientRegressor:
    def __init__(self, method='adam', learning_rate=.03, epochs=600, batch_size=32,
                 loss='mse', l2=0.0, random_state=42):
        self.method, self.learning_rate, self.epochs = method, learning_rate, epochs
        self.batch_size, self.loss, self.l2 = batch_size, loss, l2
        self.random_state = random_state

    def fit(self, X, y):
        if self.method not in ['gd', 'sgd', 'momentum', 'adam']:
            raise ValueError('Unknown optimizer')
        X = np.column_stack([np.asarray(X, dtype=float), np.ones(len(X))])
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.weights_ = np.zeros(X.shape[1])
        first = np.zeros_like(self.weights_); second = first.copy()
        self.loss_history_ = [loss_gradient(X, y, self.weights_, self.loss, self.l2)[0]]
        for step in range(1, self.epochs+1):
            ids = rng.choice(len(y), size=min(self.batch_size, len(y)), replace=False) if self.method == 'sgd' else np.arange(len(y))
            _, grad = loss_gradient(X[ids], y[ids], self.weights_, self.loss, self.l2)
            rate = self.learning_rate
            if self.method == 'momentum':
                first = .9*first + grad
                update = rate*first
            elif self.method == 'adam':
                first = .9*first + .1*grad; second = .999*second + .001*grad**2
                update = rate*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
            elif self.method == 'sgd':
                update = rate/np.sqrt(1+step/100)*grad
            else:
                update = rate*grad
            self.weights_ -= update
            value = loss_gradient(X, y, self.weights_, self.loss, self.l2)[0]
            if not np.isfinite(value):
                raise FloatingPointError('Non-finite training loss')
            self.loss_history_.append(value)
        return self

    def predict(self, X):
        if not hasattr(self, 'weights_'):
            raise ValueError('Fit model first')
        X = np.asarray(X, dtype=float)
        return X@self.weights_[:-1] + self.weights_[-1]
