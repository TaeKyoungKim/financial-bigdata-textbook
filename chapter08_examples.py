# BLOCK setup START
from pathlib import Path
import copy, json, random, platform, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, average_precision_score, roc_auc_score, confusion_matrix

OUT=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
(OUT/'assets').mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
DEVICE=torch.device('cpu')
def seed_all(seed=8001):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
seed_all()
results={}
def records(frame):return json.loads(frame.to_json(orient='records',double_precision=12))
def savefig(name):
    plt.tight_layout();plt.savefig(OUT/'assets'/name,dpi=160);plt.close()
# BLOCK setup END

# BLOCK xor START
bits=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
xor=np.array([0,1,1,0])
linear=LogisticRegression(C=100).fit(bits,xor)
hidden1=np.maximum(bits[:,0]-bits[:,1],0)
hidden2=np.maximum(bits[:,1]-bits[:,0],0)
score=hidden1+hidden2
assert np.array_equal((score>.5).astype(int),xor)
results['xor']=records(pd.DataFrame({'price_up':bits[:,0],'macro_up':bits[:,1],
    'divergence_breakout_label':xor,'linear_probability':linear.predict_proba(bits)[:,1],
    'hidden_up_divergence':hidden1,'hidden_down_divergence':hidden2,'nonlinear_output':score}))
# BLOCK xor END

# BLOCK backprop START
x=torch.tensor([.4,-.7],dtype=torch.float64)
w=torch.tensor([.3,-.2],dtype=torch.float64,requires_grad=True)
b=torch.tensor(.1,dtype=torch.float64,requires_grad=True)
v=torch.tensor(.8,dtype=torch.float64,requires_grad=True)
c=torch.tensor(-.1,dtype=torch.float64,requires_grad=True)
y=torch.tensor(.2,dtype=torch.float64)
z=w@x+b;h=torch.tanh(z);prediction=v*h+c
loss=.5*(prediction-y)**2;loss.backward()
manual=(prediction.detach()-y)*v.detach()*(1-h.detach()**2)*x
assert torch.allclose(w.grad,manual)
def scalar_loss(weights):
    return .5*(.8*np.tanh(weights@x.numpy()+.1)-.1-.2)**2
eps=1e-6;finite=[]
for j in range(2):
    plus=w.detach().numpy().copy();minus=plus.copy();plus[j]+=eps;minus[j]-=eps
    finite.append((scalar_loss(plus)-scalar_loss(minus))/(2*eps))
assert np.allclose(finite,manual.numpy(),atol=1e-9)
results['backprop']=records(pd.DataFrame({'weight':['w1','w2'],'manual':manual.numpy(),'autograd':w.grad.numpy(),'finite_difference':finite}))
activations={'Sigmoid':nn.Sigmoid(),'Tanh':nn.Tanh(),'LeakyReLU':nn.LeakyReLU(.05),'GELU':nn.GELU()}
grid=torch.linspace(-6,6,500,requires_grad=True)
fig,axes=plt.subplots(1,2,figsize=(12,4))
results['activation_gradients']=[]
for name,fn in activations.items():
    output=fn(grid);derivative=torch.autograd.grad(output.sum(),grid)[0]
    axes[0].plot(grid.detach(),output.detach(),label=name);axes[1].plot(grid.detach(),derivative,label=name)
    for value in [-5.,0.,5.]:
        point=torch.tensor(value,requires_grad=True);out=fn(point)
        grad=torch.autograd.grad(out,point)[0]
        results['activation_gradients'].append(dict(activation=name,input=value,derivative=float(grad)))
for ax in axes:ax.set_xlabel('Preactivation z');ax.legend()
axes[0].set_ylabel('Activation');axes[1].set_ylabel('Derivative');savefig('fig01_activations.png')
# BLOCK backprop END

# BLOCK losses START
class DirectionalMSE(nn.Module):
    def __init__(self,alpha=2.):
        super().__init__()
        if alpha<0:raise ValueError('alpha must be nonnegative')
        self.alpha=alpha
    def forward(self,pred,target):
        if pred.shape!=target.shape:raise ValueError('Shape mismatch')
        return ((pred-target).square()+self.alpha*torch.relu(-target*pred)).mean()
class AsymmetricMSE(nn.Module):
    def __init__(self,under_weight=3.,over_weight=1.):
        super().__init__();self.under_weight=under_weight;self.over_weight=over_weight
    def forward(self,pred,target):
        weights=torch.where(pred<target,self.under_weight,self.over_weight)
        return (weights*(pred-target).square()).mean()
targets=torch.tensor([.02,.02,-.02,-.02],dtype=torch.float64)
preds=torch.tensor([.01,-.01,.01,-.01],dtype=torch.float64,requires_grad=True)
objective=DirectionalMSE(2.)(preds,targets);objective.backward()
expected=(2*(preds.detach()-targets)-2*targets*(targets*preds.detach()<0))/len(targets)
assert torch.allclose(preds.grad,expected)
assert torch.autograd.gradcheck(DirectionalMSE(2.),(preds.detach().requires_grad_(),targets))
results['directional_examples']=records(pd.DataFrame({'true_return':targets.numpy(),'prediction':preds.detach().numpy(),
    'MSE':(preds.detach()-targets).square().numpy(),'direction_penalty_alpha2':(2*torch.relu(-targets*preds.detach())).numpy(),
    'mean_loss_gradient':preds.grad.numpy()}))
curve=np.linspace(-.04,.05,500);actual=.02
plt.figure(figsize=(9,4))
for alpha in [0,2,5]:plt.plot(curve,(curve-actual)**2+alpha*np.maximum(0,-actual*curve),label=f'alpha={alpha}')
plt.axvline(0,color='gray',ls='--');plt.axvline(actual,color='black',ls=':',label='True return')
plt.xlabel('Predicted simple return');plt.ylabel('Per-observation loss');plt.legend();savefig('fig02_directional_loss.png')
# BLOCK losses END

# BLOCK credit_data START
rng=np.random.default_rng(8002)
FEATURES=['Debt_Ratio','Current_Ratio','Operating_Margin','Interest_Coverage','ROA','Cashflow_to_Debt']
company_quality=rng.normal(size=200)
rows=[]
for year in range(2010,2025):
    quality=.65*company_quality+.75*rng.normal(size=200)
    debt=np.exp(5.1-.45*quality+.25*rng.normal(size=200))
    current=np.exp(5+.28*quality+.2*rng.normal(size=200))
    margin=6+4*quality+3*rng.normal(size=200)
    coverage=3+1.4*quality+1.3*rng.normal(size=200)
    roa=3+1.7*quality+1.3*rng.normal(size=200)
    cash=9+5*quality+4*rng.normal(size=200)
    logits=-2.8+.004*(debt-150)-.006*(current-150)-.10*(margin-5)-.18*(coverage-3)-.12*(roa-3)-.04*(cash-8)
    logits+=1.1*((debt>220)&(coverage<1.5))+.5*((margin<0)&(cash<0))
    probability=1/(1+np.exp(-logits));label=rng.binomial(1,probability)
    frame=pd.DataFrame(np.column_stack([debt,current,margin,coverage,roa,cash]),columns=FEATURES)
    frame['company_id']=[f'COMPANY_{i+1:03d}' for i in range(200)]
    frame['asof_date']=pd.Timestamp(year,6,30);frame['label_end']=pd.Timestamp(year+1,6,30)
    frame['default_next_year']=label;frame['oracle_probability']=probability
    rows.append(frame)
credit=pd.concat(rows,ignore_index=True)
missing=rng.random((len(credit),6))<.02
credit.loc[:,FEATURES]=credit[FEATURES].mask(missing)
train=credit[credit.asof_date.dt.year<=2017].copy()
valid=credit[credit.asof_date.dt.year.between(2019,2020)].copy()
test=credit[credit.asof_date.dt.year>=2022].copy()
assert train.label_end.max()<valid.asof_date.min()
assert valid.label_end.max()<test.asof_date.min()
imputer=SimpleImputer(strategy='median').fit(train[FEATURES])
scaler=RobustScaler().fit(imputer.transform(train[FEATURES]))
def transform(frame):return scaler.transform(imputer.transform(frame[FEATURES])).astype(np.float32)
Xtrain,Xvalid,Xtest=map(transform,[train,valid,test])
ytrain=train.default_next_year.to_numpy(dtype=np.float32)
yvalid=valid.default_next_year.to_numpy(dtype=np.float32)
ytest=test.default_next_year.to_numpy(dtype=np.float32)
results['splits']=[dict(split=name,rows=len(d),companies=int(d.company_id.nunique()),
    first_date=str(d.asof_date.min().date()),last_date=str(d.asof_date.max().date()),
    defaults=int(d.default_next_year.sum()),default_rate=float(d.default_next_year.mean())) for name,d in [('train',train),('valid',valid),('test',test)]]
credit.to_csv(OUT/'synthetic_credit_panel.csv',index=False)
# BLOCK credit_data END

# BLOCK pipeline START
class FinancialDataset(Dataset):
    def __init__(self,features,labels):
        self.X=torch.as_tensor(np.asarray(features),dtype=torch.float32)
        self.y=torch.as_tensor(np.asarray(labels),dtype=torch.float32)
        if self.X.ndim!=2 or self.y.shape!=(len(self.X),):raise ValueError('Invalid dataset shape')
        if not torch.isfinite(self.X).all():raise ValueError('Nonfinite features')
    def __len__(self):return len(self.y)
    def __getitem__(self,index):return self.X[index],self.y[index]
def make_loader(features,labels,shuffle,seed=42):
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(FinancialDataset(features,labels),batch_size=64,shuffle=shuffle,
        generator=generator,num_workers=0,drop_last=False)
class FinancialMLP(nn.Module):
    def __init__(self,input_dim=6,activation='leaky',dropout=.1):
        super().__init__()
        def act():return nn.LeakyReLU(.05) if activation=='leaky' else nn.GELU()
        self.network=nn.Sequential(nn.Linear(input_dim,16),act(),nn.Dropout(dropout),
            nn.Linear(16,8),act(),nn.Linear(8,1))
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
    def forward(self,x):return self.network(x).squeeze(-1)
def fit_model(model,train_loader,valid_loader,criterion,max_epochs=200,patience=20,lr=.002):
    model.to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    best=float('inf');best_state=None;best_epoch=0;stale=0;history=[]
    for epoch in range(1,max_epochs+1):
        model.train();total=0.;count=0
        for batch_x,batch_y in train_loader:
            batch_x,batch_y=batch_x.to(DEVICE),batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            prediction=model(batch_x);loss=criterion(prediction,batch_y)
            loss.backward();optimizer.step()
            total+=loss.item()*len(batch_y);count+=len(batch_y)
        model.eval();vtotal=0.;vcount=0
        with torch.no_grad():
            for batch_x,batch_y in valid_loader:
                batch_x,batch_y=batch_x.to(DEVICE),batch_y.to(DEVICE)
                vtotal+=criterion(model(batch_x),batch_y).item()*len(batch_y);vcount+=len(batch_y)
        val_loss=vtotal/vcount
        history.append(dict(epoch=epoch,train_loss=total/count,valid_loss=val_loss))
        if val_loss<best-1e-5:
            best=val_loss;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        if stale>=patience:break
    if best_state is None:raise RuntimeError('No valid checkpoint')
    model.load_state_dict(best_state);model.eval()
    return history,best_epoch,best
def infer(model,features,probability=False):
    model.eval()
    with torch.no_grad():
        out=model(torch.as_tensor(features,dtype=torch.float32,device=DEVICE))
        if probability:out=torch.sigmoid(out)
    return out.cpu().numpy()
# BLOCK pipeline END

# BLOCK train_credit START
credit_models={};histories={};results['activation_selection']=[]
for activation in ['leaky','gelu']:
    seed_all(8003)
    model=FinancialMLP(6,activation)
    history,epoch,best=fit_model(model,make_loader(Xtrain,ytrain,True),make_loader(Xvalid,yvalid,False),nn.BCEWithLogitsLoss())
    credit_models[activation]=model;histories[activation]=history
    results['activation_selection'].append(dict(activation=activation,best_epoch=epoch,epochs_run=len(history),valid_BCE=best))
chosen=min(results['activation_selection'],key=lambda r:r['valid_BCE'])['activation']
model=credit_models[chosen]
baseline=LogisticRegression(C=1,max_iter=2000).fit(Xtrain,ytrain)
def credit_metrics(name,target,p):
    cm=confusion_matrix(target,p>=.5,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    return dict(model=name,logloss=float(log_loss(target,p,labels=[0,1])),Brier=float(brier_score_loss(target,p)),
        AP=float(average_precision_score(target,p)),ROC_AUC=float(roc_auc_score(target,p)),
        TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))
prob=infer(model,Xtest,True)
results['test_credit']=[credit_metrics('MLP selected on validation',ytest,prob),
    credit_metrics('Logistic baseline',ytest,baseline.predict_proba(Xtest)[:,1])]
results['selected_model']={'activation':chosen,'parameters':sum(p.numel() for p in model.parameters()),'test_base_rate':float(ytest.mean())}
torch.save(model.state_dict(),OUT/'credit_mlp_state.pt')
np.savez(OUT/'preprocessing.npz',median=imputer.statistics_,center=scaler.center_,scale=scaler.scale_)
metadata={'features':FEATURES,'activation':chosen,'input_dim':6,'dropout':.1,'threshold_example':.5}
(OUT/'model_config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for activation,h in histories.items():
    frame=pd.DataFrame(h);axes[0].plot(frame.epoch,frame.valid_loss,label=activation)
    if activation==chosen:axes[0].plot(frame.epoch,frame.train_loss,ls='--',label=activation+' train (dropout active)')
axes[0].set(xlabel='Epoch',ylabel='BCE');axes[0].legend()
bins=np.linspace(0,1,6);bin_ids=np.clip(np.digitize(prob,bins)-1,0,4)
results['calibration']=[]
for k in range(5):
    mask=bin_ids==k
    if mask.any():results['calibration'].append(dict(bin=k,count=int(mask.sum()),mean_probability=float(prob[mask].mean()),observed_rate=float(ytest[mask].mean())))
cal=pd.DataFrame(results['calibration'])
axes[1].plot([0,1],[0,1],'k--');axes[1].plot(cal.mean_probability,cal.observed_rate,'o-')
axes[1].set(xlabel='Mean predicted PD',ylabel='Observed default fraction');savefig('fig03_credit_training.png')
# BLOCK train_credit END

# BLOCK regression START
rng=np.random.default_rng(8004)
macro_scores=rng.normal(size=(1700,4))
mu=.004*macro_scores[:,0]*macro_scores[:,1]-.002*macro_scores[:,2]+.001*np.maximum(macro_scores[:,3],0)
next_return=mu+rng.normal(0,.01,1700)
rs=StandardScaler().fit(macro_scores[:1000])
RX=rs.transform(macro_scores).astype(np.float32)
RY=(100*next_return).astype(np.float32)
results['regression']=[]
for alpha in [0.,2.]:
    seed_all(8005)
    reg=FinancialMLP(4,'leaky',dropout=0.)
    history,epoch,best=fit_model(reg,make_loader(RX[:1000],RY[:1000],True),
        make_loader(RX[1000:1250],RY[1000:1250],False),DirectionalMSE(alpha),max_epochs=180,patience=20)
    pred=infer(reg,RX[1250:]);truth=RY[1250:];wrong=pred*truth<0
    results['regression'].append(dict(alpha=alpha,best_epoch=epoch,
        test_RMSE_bp=float(np.sqrt(np.mean((pred-truth)**2))*100),
        test_direction_accuracy=float(np.mean(np.sign(pred)==np.sign(truth))),
        test_mean_direction_penalty_pct2=float(np.maximum(0,-truth*pred).mean()),
        wrong_sign_mean_absolute_actual_bp=float(np.abs(truth[wrong]).mean()*100),
        prediction_SD_pct=float(pred.std())))
pd.DataFrame({'MacroScore_1':macro_scores[:,0],'MacroScore_2':macro_scores[:,1],
    'MacroScore_3':macro_scores[:,2],'MacroScore_4':macro_scores[:,3],
    'realized_next_return':next_return}).to_csv(OUT/'synthetic_directional_returns.csv',index=False)
# BLOCK regression END

# BLOCK inference START
config=json.loads((OUT/'model_config.json').read_text(encoding='utf-8'))
reloaded=FinancialMLP(config['input_dim'],config['activation'],config['dropout'])
reloaded.load_state_dict(torch.load(OUT/'credit_mlp_state.pt',map_location='cpu',weights_only=True))
prep=np.load(OUT/'preprocessing.npz')
borrower=pd.DataFrame([[280.,85.,-2.,.7,-1.,-3.]],columns=FEATURES)
raw=borrower[config['features']].to_numpy(dtype=float)
filled=np.where(np.isnan(raw),prep['median'],raw)
normalized=((filled-prep['center'])/prep['scale']).astype(np.float32)
assert np.allclose(normalized,transform(borrower))
pd_new=infer(reloaded,normalized,True)
assert np.allclose(infer(reloaded,Xtest,True),prob)
results['borrower']={'input':borrower.iloc[0].to_dict(),'predicted_PD':float(pd_new[0]),
    'illustrative_threshold':.5,'flag':bool(pd_new[0]>=.5)}
results['python']=platform.python_version()
results['versions']={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','matplotlib','torch']}
results['checks']=['manual_backprop_autograd_finite_difference','directional_gradient_check',
    'label_maturity_before_validation_and_test','checkpoint_restores_best_validation',
    'preprocessing_saved_and_reloaded','model_reloaded_predictions_match']
(OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
requirements=[f'{p}=={v}' for p,v in results['versions'].items() if p!='torch']
requirements.append('torch=='+results['versions']['torch'].split('+')[0])
(OUT/'requirements.txt').write_text('\n'.join(requirements)+'\n',encoding='utf-8')
for key in ['splits','activation_selection','test_credit','regression']:
    print('\n'+key+'\n'+pd.DataFrame(results[key]).to_string(index=False))
print('\nBorrower PD:',results['borrower']['predicted_PD'])
print('All numerical checks passed.')
# BLOCK inference END
