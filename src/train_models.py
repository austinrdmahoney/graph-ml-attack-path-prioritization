"""Train baseline classifiers and a two-layer graph convolution network."""
import random
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES=['severity','exploitability','external_exposure','asset_criticality','privilege_indicator']

class GCN(nn.Module):
    def __init__(self,n_features,hidden=32):
        super().__init__()
        self.l1=nn.Linear(n_features,hidden)
        self.l2=nn.Linear(hidden,1)
    def forward(self,x,adj):
        h=torch.relu(self.l1(torch.sparse.mm(adj,x)))
        return self.l2(torch.sparse.mm(adj,h)).squeeze(-1)

def fit_models(vulns,adj,seed=42,epochs=140):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    X=vulns[FEATURES].to_numpy(dtype=np.float32)
    y=vulns.synthetic_exploitation_label.to_numpy(dtype=int)
    indices=np.arange(len(y))
    train, test=train_test_split(indices,test_size=0.25,random_state=seed,stratify=y)
    train,valid=train_test_split(train,test_size=0.2,random_state=seed,stratify=y[train])
    scaler=StandardScaler().fit(X[train]);Xs=scaler.transform(X).astype(np.float32)
    rf=RandomForestClassifier(n_estimators=200,min_samples_leaf=3,random_state=seed,n_jobs=1)
    xgb=XGBClassifier(n_estimators=160,max_depth=3,learning_rate=0.04,subsample=0.9,
                      colsample_bytree=0.9,eval_metric='logloss',random_state=seed,n_jobs=1)
    rf.fit(Xs[train],y[train]);xgb.fit(Xs[train],y[train])
    pred={'Random Forest':rf.predict_proba(Xs)[:,1], 'XGBoost':xgb.predict_proba(Xs)[:,1]}
    coo=adj.tocoo();inds=torch.tensor(np.vstack([coo.row,coo.col]),dtype=torch.long)
    values=torch.tensor(coo.data,dtype=torch.float32)
    tensor_adj=torch.sparse_coo_tensor(inds,values,coo.shape).coalesce()
    tx=torch.tensor(Xs);ty=torch.tensor(y,dtype=torch.float32)
    model=GCN(Xs.shape[1]);optimizer=torch.optim.Adam(model.parameters(),lr=0.012,weight_decay=0.002)
    loss_fn=nn.BCEWithLogitsLoss()
    best_loss=float('inf');best_state=None;patience=0
    for _ in range(epochs):
        model.train();optimizer.zero_grad()
        logits=model(tx,tensor_adj);loss=loss_fn(logits[train],ty[train]);loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss=loss_fn(model(tx,tensor_adj)[valid],ty[valid]).item()
        if val_loss<best_loss-1e-5:
            best_loss=val_loss;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};patience=0
        else:
            patience+=1
            if patience>=25:break
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():pred['GNN']=torch.sigmoid(model(tx,tensor_adj)).numpy()
    rows=[]
    for name,probs in pred.items():
        classification=(probs[test]>=0.5).astype(int)
        rows.append({'model':name,'test_n':len(test),'precision':precision_score(y[test],classification,zero_division=0),
            'recall':recall_score(y[test],classification,zero_division=0),
            'f1':f1_score(y[test],classification,zero_division=0),
            'roc_auc':roc_auc_score(y[test],probs[test])})
    # Held-out test only; full-graph structure is transductively visible to GNN.
    return pred,rows,{'train':train,'validation':valid,'test':test}
