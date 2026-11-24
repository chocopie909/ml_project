"""New synthetic-data demonstration; not results from the upstream notebooks."""
from pathlib import Path
import json,platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_regression,make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.dummy import DummyRegressor,DummyClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score,roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
def main():
    rows=[]
    with threadpool_limits(limits=2):
        X,y=make_regression(n_samples=2000,n_features=12,n_informative=8,noise=20,random_state=42)
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42)
        for name,model in [('DummyMean',DummyRegressor()),('Ridge',make_pipeline(StandardScaler(),Ridge(alpha=1.0)))]:
            model.fit(Xtr,ytr);pred=model.predict(Xte)
            rows.append(dict(task='regression',model=name,rmse=float(np.sqrt(mean_squared_error(yte,pred))),r2=r2_score(yte,pred),accuracy=None,roc_auc=None))
        X,y=make_classification(n_samples=2000,n_features=12,n_informative=8,n_redundant=2,flip_y=.03,class_sep=1,random_state=42)
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
        models=[('DummyPrior',DummyClassifier()),('LogisticRegression',make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42))),('SVM',make_pipeline(StandardScaler(),SVC(C=1))),('DecisionTree',DecisionTreeClassifier(max_depth=6,random_state=42)),('GradientBoosting',GradientBoostingClassifier(n_estimators=100,learning_rate=.1,max_depth=3,random_state=42))]
        for name,model in models:
            model.fit(Xtr,ytr);pred=model.predict(Xte)
            scores=model.decision_function(Xte) if hasattr(model,'decision_function') else model.predict_proba(Xte)[:,1]
            rows.append(dict(task='classification',model=name,rmse=None,r2=None,accuracy=accuracy_score(yte,pred),roc_auc=roc_auc_score(yte,scores)))
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(reports/'benchmark.csv',index=False)
    (reports/'run.json').write_text(json.dumps({'seed':42,'rows_per_task':2000,'train_rows':1500,'test_rows':500,'data':'synthetic sklearn generators; no external data','selection':'fixed parameters, no holdout tuning','python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},indent=2),encoding='utf8')
    print(df.to_string(index=False))
    return df
if __name__=='__main__':main()
