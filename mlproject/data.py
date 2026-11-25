"""Real bundled datasets. No HTTP requests or external CSV files."""
import hashlib
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split

def dataset(task):
    if task == 'regression':
        bunch = load_diabetes(as_frame=True, scaled=False)
    elif task == 'classification':
        bunch = load_breast_cancer(as_frame=True)
    else:
        raise ValueError('task must be regression or classification')
    X = bunch.data.copy()
    # Positive class explicitly means malignant, unlike the original sklearn labels.
    y = bunch.target.copy() if task == 'regression' else (bunch.target == 0).astype(int)
    return X, y

def split(task, seed=42):
    X, y = dataset(task)
    ids = np.arange(len(y))
    strat = y if task == 'classification' else None
    trainval, test = train_test_split(ids, test_size=.2, random_state=seed, stratify=strat)
    train, val = train_test_split(trainval, test_size=.25, random_state=seed,
                                stratify=y.iloc[trainval] if strat is not None else None)
    return {name: (X.iloc[idx].copy(), y.iloc[idx].copy()) for name, idx in
            [('train', train), ('val', val), ('test', test)]}

def fingerprint(task):
    X, y = dataset(task)
    return hashlib.sha256(np.column_stack([X.values, y.values]).astype('<f8').tobytes()).hexdigest()
