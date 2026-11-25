import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from mlproject.optimizers import loss_gradient, GradientRegressor
from mlproject.tree import best_split, BinaryTree
from mlproject.boosting import LogisticBoosting, logistic_loss
from mlproject.data import split

@pytest.mark.parametrize('loss',['mse','mae','huber','logcosh'])
@pytest.mark.parametrize('l2',[0,.3])
def test_gradient_matches_finite_difference(loss,l2):
    rng=np.random.default_rng(7);X=np.column_stack([rng.normal(size=(30,4)),np.ones(30)])
    y=rng.normal(size=30);w=rng.normal(size=5);original=w.copy()
    _,grad=loss_gradient(X,y,w,loss,l2)
    h=1e-6;numeric=np.zeros(5)
    for i in range(5):
        step=np.zeros(5);step[i]=h
        numeric[i]=(loss_gradient(X,y,w+step,loss,l2)[0]-loss_gradient(X,y,w-step,loss,l2)[0])/(2*h)
    np.testing.assert_allclose(grad,numeric,atol=1e-6)
    np.testing.assert_array_equal(w,original)

@pytest.mark.parametrize('method',['gd','sgd','momentum','adam'])
def test_optimizer_converges_and_refit_resets(method):
    rng=np.random.default_rng(2);X=rng.normal(size=(120,4));y=X@np.array([.2,-.4,.7,1.])+ .3
    m=GradientRegressor(method=method,learning_rate=.03,epochs=600).fit(X,y)
    assert np.mean((m.predict(X)-y)**2)<.01
    first=m.predict(X).copy();m.fit(X,y)
    np.testing.assert_allclose(m.predict(X),first)

def test_regularization_does_not_penalize_intercept():
    X=np.ones((10,2));y=np.ones(10);w=np.array([0.,3.])
    a,g=loss_gradient(X,y,w,l2=0);b,h=loss_gradient(X,y,w,l2=100)
    assert a==b;np.testing.assert_array_equal(g,h)

def test_large_residual_logcosh_is_finite():
    value,grad=loss_gradient(np.ones((3,2)),np.zeros(3),np.array([1000.,1000.]),'logcosh')
    assert np.isfinite(value) and np.isfinite(grad).all()

def test_split_matches_brute_force_with_leaf_constraint():
    rng=np.random.default_rng(4);x=rng.integers(0,10,size=40);y=rng.integers(0,2,size=40)
    candidates=[]
    unique=np.unique(x)
    for t in (unique[:-1]+unique[1:])/2:
        mask=x<=t
        if min(mask.sum(),(~mask).sum())<6:continue
        score=sum(len(z)*2*z.mean()*(1-z.mean()) for z in [y[mask],y[~mask]])/len(y)
        candidates.append((score,t))
    score,t=min(candidates)
    threshold,gini=best_split(x,y,6)
    assert threshold==t;assert gini==pytest.approx(score)

def test_constant_feature_and_pure_class():
    assert best_split(np.ones(10),np.arange(10)%2) is None
    m=BinaryTree().fit(np.ones((10,2)),np.ones(10))
    np.testing.assert_array_equal(m.predict(np.ones((3,2))),np.ones(3))

def test_tree_matches_sklearn_simple_case_and_leaf_sizes():
    X=np.arange(30).reshape(-1,1);y=(X[:,0]>12).astype(int)
    m=BinaryTree(max_depth=3,min_samples_leaf=4).fit(X,y)
    ref=DecisionTreeClassifier(max_depth=3,min_samples_leaf=4).fit(X,y)
    np.testing.assert_array_equal(m.predict(X),ref.predict(X))
    def walk(n):
        if 'feature' not in n:assert n['size']>=4
        else:walk(n['left']);walk(n['right'])
    walk(m.root_)

def test_boosting_probability_loss_and_repeatability():
    X,y=make_classification(n_samples=180,n_features=6,n_informative=4,random_state=4)
    m=LogisticBoosting(n_estimators=60,learning_rate=.2).fit(X,y)
    p=m.predict_proba(X);assert np.isfinite(p).all();np.testing.assert_allclose(p.sum(axis=1),1)
    assert m.history_[-1]['train_loss']<m.initial_loss_
    expected=np.full(len(y),m.init_score_)
    for tree in m.models_:expected += m.learning_rate*tree.predict(X)
    np.testing.assert_allclose(m.decision_function(X),expected)
    m.fit(X,y);np.testing.assert_allclose(m.predict_proba(X),p)

def test_boosting_early_stopping_restores_best_model():
    X,y=make_classification(n_samples=160,n_features=6,random_state=6)
    m=LogisticBoosting(n_estimators=100,patience=4,learning_rate=.5).fit(X[:100],y[:100],(X[100:],1-y[100:]))
    losses=[logistic_loss(1-y[100:],np.full(60,m.init_score_))]+[r['val_loss'] for r in m.history_]
    assert m.best_iteration_==int(np.argmin(losses))
    assert len(m.history_)<100
    assert logistic_loss(1-y[100:],m.decision_function(X[100:]))==pytest.approx(min(losses))

@pytest.mark.parametrize('task',['regression','classification'])
def test_data_splits_disjoint_and_stable(task):
    s=split(task);again=split(task)
    ids=[set(s[k][0].index) for k in ['train','val','test']]
    assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
    for k in s:np.testing.assert_array_equal(s[k][0].index,again[k][0].index)
    if task=='classification':
        for k in s:assert set(s[k][1])=={0,1}
