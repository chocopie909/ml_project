"""Five connected experiments. Selection uses validation, never the test split."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    brier_score_loss, log_loss, confusion_matrix)
from threadpoolctl import threadpool_limits
from .data import split, dataset, fingerprint
from .optimizers import GradientRegressor
from .tree import BinaryTree
from .boosting import LogisticBoosting

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports'
REPORTS.mkdir(exist_ok=True)

def regression_metrics(y, pred):
    return {'rmse': float(np.sqrt(mean_squared_error(y,pred))), 'mae': mean_absolute_error(y,pred), 'r2': r2_score(y,pred)}

def classification_metrics(y, p):
    pred = p >= .5
    return {'accuracy': accuracy_score(y,pred), 'precision': precision_score(y,pred,zero_division=0),
        'recall': recall_score(y,pred,zero_division=0), 'f1': f1_score(y,pred,zero_division=0),
        'roc_auc': roc_auc_score(y,p), 'brier': brier_score_loss(y,p), 'log_loss': log_loss(y,p)}

def save(name, rows):
    df = pd.DataFrame(rows); df.to_csv(REPORTS/f'{name}.csv',index=False)
    return df

def figure(name):
    plt.tight_layout(); plt.savefig(REPORTS/f'{name}.png',dpi=150); plt.close()

def regression():
    s=split('regression'); X,y=s['train']; Xv,yv=s['val']; Xt,yt=s['test']
    rows=[]; tuning=[]; models={}
    baseline=DummyRegressor().fit(X,y)
    rows.append({'model':'Dummy mean',**regression_metrics(yt,baseline.predict(Xt))})
    for label,cls in [('Ridge',Ridge),('Lasso',Lasso)]:
        candidates=[]
        for a in [.01,.1,1,10,100]:
            model=make_pipeline(StandardScaler(),cls(alpha=a,max_iter=10000))
            model.fit(X,y); val=regression_metrics(yv,model.predict(Xv))['rmse']
            candidates.append((val,a,model));tuning.append({'model':label,'alpha':a,'val_rmse':val})
        val,a,model=min(candidates,key=lambda x:x[0]);models[label]=model
        rows.append({'model':label,'alpha':a,'val_rmse':val,**regression_metrics(yt,model.predict(Xt))})
    save('01_regression_tuning',tuning)
    plt.figure(figsize=(6,4));plt.scatter(yt,models['Ridge'].predict(Xt),alpha=.7)
    plt.plot([yt.min(),yt.max()],[yt.min(),yt.max()],'k--');plt.xlabel('Observed target');plt.ylabel('Ridge prediction')
    plt.title('Diabetes: untouched test split');figure('01_regression')
    return save('01_regression',rows)

def optimization():
    s=split('regression'); X,y=s['train']; Xv,yv=s['val']; Xt,yt=s['test']
    scaler=StandardScaler().fit(X);X=scaler.transform(X);Xv=scaler.transform(Xv);Xt=scaler.transform(Xt)
    mean,std=float(y.mean()),float(y.std(ddof=0));z=(y.values-mean)/std
    rows=[];tuning=[];plt.figure(figsize=(7,4))
    for method in ['gd','sgd','momentum','adam']:
        candidates=[]
        for rate in [.001,.01,.03]:
            m=GradientRegressor(method=method,learning_rate=rate,epochs=700).fit(X,z)
            val=regression_metrics(yv,m.predict(Xv)*std+mean)['rmse']
            candidates.append((val,rate,m));tuning.append({'method':method,'learning_rate':rate,'val_rmse':val})
        val,rate,m=min(candidates,key=lambda x:x[0])
        rows.append({'model':method,'learning_rate':rate,'val_rmse':val,**regression_metrics(yt,m.predict(Xt)*std+mean)})
        plt.plot(m.loss_history_,label=method)
    plt.xlabel('Updates (SGD uses one minibatch per update)');plt.ylabel('Training MSE, standardized target');plt.legend();figure('02_optimization')
    save('02_optimization_tuning',tuning)
    # Robust losses and L2: validation experiment only, no repeated test selection.
    robust=[]
    for loss in ['mse','mae','huber','logcosh']:
        for l2 in [0,.1]:
            m=GradientRegressor(loss=loss,l2=l2,epochs=700).fit(X,z)
            robust.append({'loss':loss,'l2':l2,'val_rmse':regression_metrics(yv,m.predict(Xv)*std+mean)['rmse']})
    save('02_robust_losses',robust)
    return save('02_optimization',rows)

def classification():
    s=split('classification');X,y=s['train'];Xv,yv=s['val'];Xt,yt=s['test']
    rows=[];tuning=[];models={}
    dummy=DummyClassifier().fit(X,y)
    rows.append({'model':'Dummy prior',**classification_metrics(yt,dummy.predict_proba(Xt)[:,1])})
    for label in ['Logistic regression','Calibrated SVM']:
        candidates=[]
        for C in [.1,1,10]:
            if label=='Logistic regression':
                m=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000,random_state=42))
            else:
                # The scaler is inside the estimator, so calibration folds cannot leak scaling.
                m=CalibratedClassifierCV(make_pipeline(StandardScaler(),SVC(C=C)),method='sigmoid',cv=3)
            m.fit(X,y);auc=roc_auc_score(yv,m.predict_proba(Xv)[:,1]);candidates.append((auc,C,m))
            tuning.append({'model':label,'C':C,'val_roc_auc':auc})
        auc,C,m=max(candidates,key=lambda x:x[0]);models[label]=m
        p=m.predict_proba(Xt)[:,1]
        rows.append({'model':label,'C':C,'val_roc_auc':auc,**classification_metrics(yt,p)})
        np.savetxt(REPORTS/f'03_confusion_{"lr" if label.startswith("Logistic") else "svm"}.csv',confusion_matrix(yt,p>=.5),delimiter=',',fmt='%d')
    plt.figure(figsize=(6,4));plt.plot([0,1],[0,1],'k--')
    for label,m in models.items():
        observed,predicted=calibration_curve(yt,m.predict_proba(Xt)[:,1],n_bins=6,strategy='quantile')
        plt.plot(predicted,observed,'o-',label=label)
    plt.xlabel('Predicted probability');plt.ylabel('Observed positive fraction');plt.legend();figure('03_calibration')
    save('03_classification_tuning',tuning)
    return save('03_classification',rows)

def trees():
    s=split('classification');X,y=s['train'];Xv,yv=s['val'];Xt,yt=s['test']
    rows=[];tuning=[]
    for label in ['Custom CART','sklearn CART']:
        candidates=[]
        for depth in [2,3,4,6]:
            for leaf in [3,8]:
                m=BinaryTree(depth,leaf) if label.startswith('Custom') else DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,random_state=42)
                m.fit(X,y);auc=roc_auc_score(yv,m.predict_proba(Xv)[:,1]);candidates.append((auc,depth,leaf,m))
                tuning.append({'model':label,'depth':depth,'min_leaf':leaf,'val_roc_auc':auc})
        auc,depth,leaf,m=max(candidates,key=lambda x:x[0])
        rows.append({'model':label,'depth':depth,'min_leaf':leaf,'val_roc_auc':auc,**classification_metrics(yt,m.predict_proba(Xt)[:,1])})
    t=save('04_tree_tuning',tuning)
    plt.figure(figsize=(6,4))
    for label in t.model.unique():
        curve=t[t.model==label].groupby('depth').val_roc_auc.max();plt.plot(curve.index,curve.values,'o-',label=label)
    plt.xlabel('Max depth');plt.ylabel('Best validation ROC-AUC across min leaf');plt.legend();figure('04_trees')
    return save('04_trees',rows)

def boosting():
    s=split('classification');X,y=s['train'];Xv,yv=s['val'];Xt,yt=s['test']
    rows=[];tuning=[];chosen=None
    for label in ['Custom logistic boosting','sklearn gradient boosting']:
        candidates=[]
        for rate in [.05,.1,.3]:
            for depth in [1,2]:
                if label.startswith('Custom'):
                    m=LogisticBoosting(n_estimators=200,learning_rate=rate,max_depth=depth).fit(X,y,eval_set=(Xv,yv))
                else:
                    m=GradientBoostingClassifier(n_estimators=200,learning_rate=rate,max_depth=depth,min_samples_leaf=5,random_state=42).fit(X,y)
                auc=roc_auc_score(yv,m.predict_proba(Xv)[:,1]);candidates.append((auc,rate,depth,m))
                tuning.append({'model':label,'rate':rate,'depth':depth,'val_roc_auc':auc})
        auc,rate,depth,m=max(candidates,key=lambda x:x[0]);p=m.predict_proba(Xt)[:,1]
        rows.append({'model':label,'rate':rate,'depth':depth,'trees':len(m.models_) if label.startswith('Custom') else 200,'val_roc_auc':auc,**classification_metrics(yt,p)})
        if label.startswith('Custom'):chosen=m
    h=pd.DataFrame(chosen.history_);h.to_csv(REPORTS/'05_boosting_history.csv',index=False)
    plt.figure(figsize=(6,4));plt.plot(h.iteration,h.train_loss,label='train');plt.plot(h.iteration,h.val_loss,label='validation')
    plt.axvline(chosen.best_iteration_,linestyle='--',color='grey',label='retained iteration')
    plt.xlabel('Trees');plt.ylabel('Log loss');plt.legend();figure('05_boosting')
    save('05_boosting_tuning',tuning)
    return save('05_boosting',rows)

FUNCTIONS={'01':regression,'02':optimization,'03':classification,'04':trees,'05':boosting}

def run(name):
    with threadpool_limits(limits=2):
        return FUNCTIONS[name]()

def run_all():
    import platform, sklearn
    result={}
    for name in FUNCTIONS:
        print('Running',name,flush=True);result[name]=run(name)
    metadata={'seed':42,'split':'60/20/20; validation selects parameters; test is untouched until evaluation',
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
        'positive_class':'malignant = 1','datasets':{task:{'shape':list(dataset(task)[0].shape),'sha256':fingerprint(task),
        'split_sizes':{k:len(v[1]) for k,v in split(task).items()}} for task in ['regression','classification']}}
    (REPORTS/'run.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    return result
