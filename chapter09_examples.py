# BLOCK setup START
from pathlib import Path
import copy,json,random,time,platform,importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler

OUT=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
(OUT/'assets').mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
DEVICE=torch.device('cpu')
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
results={}
def records(frame):return json.loads(frame.to_json(orient='records',double_precision=12))
def savefig(name):
    plt.tight_layout();plt.savefig(OUT/'assets'/name,dpi=160);plt.close()
# BLOCK setup END

# BLOCK gradients START
results['gradient_demo']=[]
for recurrent_weight in [.5,1.,1.2]:
    for length in [10,30,60]:
        initial=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        hidden=initial
        for _ in range(length):hidden=torch.tanh(recurrent_weight*hidden)
        gradient=torch.autograd.grad(hidden,initial)[0].item()
        assert np.isclose(gradient,recurrent_weight**length)
        results['gradient_demo'].append(dict(weight=recurrent_weight,steps=length,
            dh_final_dh_initial=gradient,analytic=recurrent_weight**length))
fig,axes=plt.subplots(1,2,figsize=(12,4));steps=np.arange(1,101)
for w in [.5,1.,1.2]:axes[0].semilogy(steps,w**steps,label=f'RNN w={w}, h=0')
for f in [.9,.99,.999]:axes[1].semilogy(steps,f**steps,label=f'Fixed forget gate={f}')
axes[0].set(xlabel='Time steps',ylabel='RNN derivative magnitude');axes[1].set(xlabel='Time steps',ylabel='Direct cell-path retention')
for ax in axes:ax.legend()
savefig('fig01_memory_gradients.png')
# BLOCK gradients END

# BLOCK gates START
seed_all(9001)
cell=nn.LSTMCell(6,4).double()
x=torch.linspace(-.3,.3,6,dtype=torch.float64).reshape(1,6)
h0=torch.tensor([[.1,-.1,.2,-.2]],dtype=torch.float64)
c0=torch.tensor([[.2,.3,-.1,-.4]],dtype=torch.float64)
pre=x@cell.weight_ih.T+cell.bias_ih+h0@cell.weight_hh.T+cell.bias_hh
i,f,g,o=pre.chunk(4,dim=1)
i,f,g,o=torch.sigmoid(i),torch.sigmoid(f),torch.tanh(g),torch.sigmoid(o)
c=f*c0+i*g;h=o*torch.tanh(c)
reference_h,reference_c=cell(x,(h0,c0))
assert torch.allclose(h,reference_h) and torch.allclose(c,reference_c)
results['lstm_gates']=records(pd.DataFrame({'unit':np.arange(4),'forget':f.detach().numpy()[0],
    'input':i.detach().numpy()[0],'candidate':g.detach().numpy()[0],'old_cell':c0.numpy()[0],
    'retained':(f*c0).detach().numpy()[0],'written':(i*g).detach().numpy()[0],
    'new_cell':c.detach().numpy()[0],'output':o.detach().numpy()[0],'hidden':h.detach().numpy()[0]}))
gru=nn.GRUCell(6,4).double()
input_part=x@gru.weight_ih.T+gru.bias_ih
hidden_part=h0@gru.weight_hh.T+gru.bias_hh
ir,iz,inn=input_part.chunk(3,dim=1);hr,hz,hn=hidden_part.chunk(3,dim=1)
reset=torch.sigmoid(ir+hr);update=torch.sigmoid(iz+hz)
candidate=torch.tanh(inn+reset*hn)
new_h=(1-update)*candidate+update*h0
assert torch.allclose(new_h,gru(x,h0))
results['memory_constants']=[dict(forget=f,retention_60=f**60,
    half_life_steps=float(np.log(.5)/np.log(f))) for f in [.9,.99,.999]]
# BLOCK gates END

# BLOCK data START
rng=np.random.default_rng(9002)
length=1650;dates=pd.bdate_range('2015-01-02',periods=length)
state=np.zeros(length)
for t in range(1,length):state[t]=.97*state[t-1]+.18*rng.normal()
logreturn=np.zeros(length)
for t in range(1,length):
    sigma=.007+.003/(1+np.exp(-state[t-1]))
    logreturn[t]=.00015+.0005*np.tanh(state[t-1])+.08*logreturn[t-1]+sigma*rng.normal()
close=100*np.exp(np.cumsum(logreturn))
previous=np.r_[close[0],close[:-1]]
open_price=previous*np.exp(rng.normal(0,.002,length))
high=np.maximum(open_price,close)*np.exp(np.abs(rng.normal(0,.003,length)))
low=np.minimum(open_price,close)*np.exp(-np.abs(rng.normal(0,.003,length)))
volume=1e6*np.exp(.2*np.abs(logreturn)/.01+.15*state+rng.normal(0,.25,length))
frame=pd.DataFrame({'Open':open_price,'High':high,'Low':low,'Close':close,'Volume':volume},index=dates)
frame['Volatility']=pd.Series(np.expm1(logreturn),index=dates).rolling(20).std(ddof=1)*np.sqrt(252)
HORIZON=5;LOOKBACK=60;FEATURES=['Open','High','Low','Close','Volume','Volatility']
frame['target_return']=frame.Close.shift(-HORIZON)/frame.Close-1
frame['target_price']=frame.Close.shift(-HORIZON)
assert (frame.High>=frame[['Open','Close']].max(axis=1)).all()
assert (frame.Low<=frame[['Open','Close']].min(axis=1)).all()
assert np.isfinite(frame[FEATURES].iloc[19:].to_numpy()).all()
first_origin=19+LOOKBACK-1
assert np.isclose(frame.target_return.iloc[100],frame.Close.iloc[100+HORIZON]/frame.Close.iloc[100]-1)
frame.to_csv(OUT/'synthetic_ohlcv.csv',index_label='date')
fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
axes[0].plot(frame.index,frame.Close);axes[0].set_ylabel('Synthetic close')
axes[1].plot(frame.index,frame.Volatility*100);axes[1].set_ylabel('Trailing annual volatility (%)');savefig('fig02_market.png')
# BLOCK data END

# BLOCK dataset START
class WindowDataset(Dataset):
    def __init__(self,scaled_features,targets,origins,lookback=60):
        self.features=torch.as_tensor(scaled_features,dtype=torch.float32)
        self.targets=torch.as_tensor(targets,dtype=torch.float32)
        self.origins=np.asarray(origins,dtype=int);self.lookback=lookback
    def __len__(self):return len(self.origins)
    def __getitem__(self,index):
        t=int(self.origins[index])
        x=self.features[t-self.lookback+1:t+1]
        y=self.targets[t]
        if x.shape!=(self.lookback,6):raise ValueError('Invalid window shape')
        if not torch.isfinite(x).all() or not torch.isfinite(y):raise ValueError('Nonfinite example')
        return x,y
def loader(features,targets,origins,shuffle):
    return DataLoader(WindowDataset(features,targets,origins,LOOKBACK),batch_size=64,
        shuffle=shuffle,num_workers=0,generator=torch.Generator().manual_seed(42))
def walk_forward_splits():
    for fold,test_start in enumerate([1000,1200,1400],start=1):
        valid_start=test_start-205;valid_end=test_start-HORIZON-1
        train_end=valid_start-HORIZON-1
        train_ids=np.arange(first_origin,train_end+1)
        valid_ids=np.arange(valid_start,valid_end+1)
        test_ids=np.arange(test_start,test_start+200)
        assert train_ids[-1]+HORIZON<valid_ids[0]
        assert valid_ids[-1]+HORIZON<test_ids[0]
        assert test_ids[-1]+HORIZON<len(frame)
        yield fold,train_ids,valid_ids,test_ids
# BLOCK dataset END

# BLOCK models START
class SequenceRegressor(nn.Module):
    def __init__(self,kind='LSTM',hidden_size=16):
        super().__init__()
        cls={'RNN':nn.RNN,'LSTM':nn.LSTM,'GRU':nn.GRU}[kind]
        self.recurrent=cls(input_size=6,hidden_size=hidden_size,num_layers=1,batch_first=True,bidirectional=False)
        self.head=nn.Linear(hidden_size,1)
    def forward(self,x):
        # No state argument: each independent 60-day window starts at zero.
        sequence,_=self.recurrent(x)
        return self.head(sequence[:,-1,:]).squeeze(-1)
def fit_model(model,training,validation):
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    criterion=nn.MSELoss();best=float('inf');checkpoint=None;stale=0;history=[];best_epoch=0
    for epoch in range(1,41):
        model.train();total=0.;count=0;clip_count=0;batches=0
        for x,y in training:
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(x),y);loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.,error_if_nonfinite=True)
            clip_count+=int(norm>1);batches+=1
            optimizer.step();total+=loss.item()*len(y);count+=len(y)
        model.eval();valtotal=0.;valcount=0
        with torch.no_grad():
            for x,y in validation:valtotal+=criterion(model(x),y).item()*len(y);valcount+=len(y)
        vl=valtotal/valcount
        history.append(dict(epoch=epoch,train_MSE=total/count,valid_MSE=vl,clipped_batch_fraction=clip_count/batches))
        if vl<best-1e-5:
            best=vl;checkpoint=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        if stale>=7:break
    model.load_state_dict(checkpoint);model.eval()
    return history,best_epoch
def predict(model,data):
    outputs=[];model.eval()
    with torch.no_grad():
        for x,_ in data:outputs.append(model(x).numpy())
    return np.concatenate(outputs)
def metric(name,y,p):
    return dict(model=name,MAE_bp=float(np.mean(np.abs(y-p))*100),
        RMSE_bp=float(np.sqrt(np.mean((y-p)**2))*100),
        Hit_Ratio=float(np.mean(np.sign(y)==np.sign(p))))
results['parameters']=[dict(model=kind,parameters=sum(p.numel() for p in SequenceRegressor(kind).parameters())) for kind in ['RNN','LSTM','GRU']]
# BLOCK models END

# BLOCK walkforward START
results['folds']=[];results['fold_metrics']=[];results['training']=[]
all_predictions=[];lstm_histories={}
target_pct=frame.target_return.to_numpy()*100
raw=frame[FEATURES].to_numpy()
for fold,train_ids,valid_ids,test_ids in walk_forward_splits():
    scaler=StandardScaler().fit(raw[19:train_ids[-1]+1])
    scaled=scaler.transform(raw).astype(np.float32)
    train_loader=loader(scaled,target_pct,train_ids,True)
    valid_loader=loader(scaled,target_pct,valid_ids,False)
    test_loader=loader(scaled,target_pct,test_ids,False)
    bx,by=next(iter(train_loader));assert bx.shape[1:]==(60,6)
    results['folds'].append(dict(fold=fold,train_windows=len(train_ids),valid_windows=len(valid_ids),test_windows=len(test_ids),
        train_last=str(frame.index[train_ids[-1]].date()),train_label_last=str(frame.index[train_ids[-1]+HORIZON].date()),
        valid_first=str(frame.index[valid_ids[0]].date()),valid_last=str(frame.index[valid_ids[-1]].date()),test_first=str(frame.index[test_ids[0]].date()),test_last=str(frame.index[test_ids[-1]].date())))
    block=pd.DataFrame({'date':frame.index[test_ids],'origin':test_ids,'true_return_pct':target_pct[test_ids],'current_close':frame.Close.iloc[test_ids].to_numpy()})
    block['Zero']=0.;block['TrainingMean']=float(np.mean(target_pct[train_ids]))
    block['Past5DayMomentum']=(frame.Close.iloc[test_ids].to_numpy()/frame.Close.iloc[test_ids-HORIZON].to_numpy()-1)*100
    for kind in ['RNN','LSTM','GRU']:
        seed_all(9003+fold)
        model=SequenceRegressor(kind)
        start=time.perf_counter()
        history,best_epoch=fit_model(model,loader(scaled,target_pct,train_ids,True),valid_loader)
        elapsed=time.perf_counter()-start
        block[kind]=predict(model,test_loader)
        results['training'].append(dict(fold=fold,model=kind,best_epoch=best_epoch,epochs_run=len(history),seconds=elapsed))
        if kind=='LSTM':
            lstm_histories[fold]=history
            torch.save(model.state_dict(),OUT/f'lstm_fold{fold}.pt')
            np.savez(OUT/f'scaler_fold{fold}.npz',mean=scaler.mean_,scale=scaler.scale_)
            reloaded=SequenceRegressor('LSTM');reloaded.load_state_dict(torch.load(OUT/f'lstm_fold{fold}.pt',weights_only=True))
            assert np.allclose(predict(reloaded,test_loader),block[kind].to_numpy())
    for name in ['Zero','TrainingMean','Past5DayMomentum','RNN','LSTM','GRU']:
        row=metric(name,block.true_return_pct.to_numpy(),block[name].to_numpy());row['fold']=fold
        results['fold_metrics'].append(row)
    all_predictions.append(block)
oos=pd.concat(all_predictions,ignore_index=True)
assert not oos.date.duplicated().any()
results['overall']=[metric(name,oos.true_return_pct.to_numpy(),oos[name].to_numpy()) for name in ['Zero','TrainingMean','Past5DayMomentum','RNN','LSTM','GRU']]
oos['LSTM_implied_future_price']=oos.current_close*(1+oos.LSTM/100)
oos['actual_future_price']=oos.current_close*(1+oos.true_return_pct/100)
oos.to_csv(OUT/'walkforward_predictions.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,(fold,hist) in zip(axes,lstm_histories.items()):
    h=pd.DataFrame(hist);ax.plot(h.epoch,h.train_MSE,label='Train');ax.plot(h.epoch,h.valid_MSE,label='Validation')
    ax.set(title=f'LSTM fold {fold}',xlabel='Epoch',ylabel='MSE (percentage points squared)');ax.legend()
savefig('fig03_training.png')
fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
axes[0].plot(oos.date,oos.true_return_pct,label='Realized future 5-day return',alpha=.6,lw=.8)
axes[0].plot(oos.date,oos.LSTM,label='LSTM forecast',lw=.8);axes[0].set_ylabel('Return (%)');axes[0].legend()
axes[1].plot(oos.date,oos.actual_future_price,label='Actual price at t+5')
axes[1].plot(oos.date,oos.LSTM_implied_future_price,label='Implied forecast price');axes[1].set_ylabel('Price indexed by forecast origin');axes[1].legend()
savefig('fig04_forecasts.png')
# BLOCK walkforward END

# BLOCK inference START
saved=np.load(OUT/'scaler_fold3.npz')
last_model=SequenceRegressor('LSTM')
last_model.load_state_dict(torch.load(OUT/'lstm_fold3.pt',weights_only=True))
last_window=((raw[-LOOKBACK:]-saved['mean'])/saved['scale']).astype(np.float32)
last_model.eval()
with torch.no_grad():forecast_pct=float(last_model(torch.from_numpy(last_window)[None,:,:]).item())
results['latest']={'asof_date':str(frame.index[-1].date()),'horizon_sessions':HORIZON,
    'forecast_return_pct':forecast_pct,'current_close':float(frame.Close.iloc[-1]),
    'implied_price':float(frame.Close.iloc[-1]*(1+forecast_pct/100)),
    'note':'Uses the third fold model without retraining; the future target is unknown.'}
results['checks']=['BPTT_scalar_autograd_matches_power','manual_LSTM_matches_LSTMCell','manual_GRU_matches_GRUCell',
    'OHLC_consistency','window_shape_B_60_6','label_maturity_separation',
    'no_duplicate_test_origins','saved_LSTM_predictions_match']
results['python']=platform.python_version()
results['versions']={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','matplotlib','torch']}
(OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'requirements.txt').write_text('\n'.join(f'{p}=={v.split("+")[0]}' for p,v in results['versions'].items())+'\n',encoding='utf-8')
(OUT/'model_config.json').write_text(json.dumps({'features':FEATURES,'lookback':LOOKBACK,'horizon':HORIZON,'hidden_size':16,'num_layers':1,'target_units':'percentage points'},indent=2),encoding='utf-8')
print(pd.DataFrame(results['overall']).to_string(index=False))
print('All numerical and chronological checks passed.')
# BLOCK inference END
