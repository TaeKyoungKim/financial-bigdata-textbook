"""Chapter 10: attention, Compact TFT and optional FinBERT.
Default run is offline and uses explicitly synthetic news scores.
Set FINBERT_INPUT to a CSV with timestamp,headline for real FinBERT scoring.
"""
from __future__ import annotations
import copy, json, math, os, platform, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT=Path(__file__).resolve().parent; ASSET=ROOT/'assets'; ASSET.mkdir(exist_ok=True)

# BLOCK ATTENTION START
def scaled_dot_product_attention(Q,K,V,causal=False):
    scores=Q@K.transpose(-2,-1)/math.sqrt(Q.size(-1))
    if causal:
        mask=torch.triu(torch.ones(scores.size(-2),scores.size(-1),dtype=torch.bool,device=scores.device),1)
        scores=scores.masked_fill(mask,float('-inf'))
    weights=torch.softmax(scores,dim=-1)
    return weights@V,weights

Q=torch.randn(2,6,16); K=torch.randn(2,6,16); V=torch.randn(2,6,8)
context,A=scaled_dot_product_attention(Q,K,V,True)
assert context.shape==(2,6,8) and torch.allclose(A.sum(-1),torch.ones(2,6),atol=1e-6)
for dk in (4,16,64,256):
    q=torch.randn(10000,dk); k=torch.randn(10000,dk); raw=(q*k).sum(1)
    print(f'dk={dk:3d} raw_std={raw.std():.3f} scaled_std={(raw/math.sqrt(dk)).std():.3f}')
# BLOCK ATTENTION END

# BLOCK POSITION START
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self,d_model,max_len=512):
        super().__init__(); pe=torch.zeros(max_len,d_model)
        pos=torch.arange(max_len,dtype=torch.float32).unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2,dtype=torch.float32)*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1)]

dates=pd.bdate_range('2020-01-01',periods=10); calendar=pd.DataFrame(index=dates)
calendar['dow_sin']=np.sin(2*np.pi*dates.dayofweek/5); calendar['dow_cos']=np.cos(2*np.pi*dates.dayofweek/5)
calendar['month_end']=dates.is_month_end.astype(float); print(calendar.head())
# BLOCK POSITION END

# BLOCK FINBERT START
def score_headlines_finbert(frame,model_name='ProsusAI/finbert',batch_size=32):
    """Return probabilities and score=p_positive-p_negative; model labels are read safely."""
    try:
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
    except ImportError as e: raise RuntimeError('pip install -r requirements-finbert.txt') from e
    tok=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE).eval()
    labels={int(k):str(v).lower() for k,v in model.config.id2label.items()}; rows=[]
    texts=frame['headline'].fillna('').astype(str).tolist()
    with torch.inference_mode():
        for i in range(0,len(texts),batch_size):
            enc=tok(texts[i:i+batch_size],padding=True,truncation=True,max_length=128,return_tensors='pt')
            p=model(**{k:v.to(DEVICE) for k,v in enc.items()}).logits.softmax(-1).cpu().numpy()
            for probs in p:
                row={f'p_{labels[j]}':float(probs[j]) for j in range(len(probs))}
                row['sentiment_score']=row.get('p_positive',0)-row.get('p_negative',0); rows.append(row)
    return pd.concat([frame.reset_index(drop=True),pd.DataFrame(rows)],axis=1)

def aggregate_news_before_cutoff(scored,cutoff_hour=15):
    x=scored.copy(); x['timestamp']=pd.to_datetime(x['timestamp'])
    x=x[x.timestamp.dt.hour<cutoff_hour].copy(); x['date']=x.timestamp.dt.normalize()
    return x.groupby('date').agg(sentiment=('sentiment_score','mean'),news_count=('headline','size')).reset_index()

finbert_input=os.getenv('FINBERT_INPUT')
if finbert_input:
    score_headlines_finbert(pd.read_csv(finbert_input)).to_csv(ROOT/'finbert_scored.csv',index=False)
# BLOCK FINBERT END

# BLOCK DATA START
def make_synthetic_market(n=1500,seed=42):
    """Synthetic teaching market. sentiment is NOT FinBERT output."""
    rng=np.random.default_rng(seed); dates=pd.bdate_range('2018-01-02',periods=n)
    fomc=np.zeros(n); earnings=np.zeros(n); fomc[np.arange(30,n,42)]=1; earnings[np.arange(50,n,63)]=1
    latent=np.zeros(n); news=np.zeros(n); count=np.zeros(n)
    for t in range(1,n): latent[t]=.88*latent[t-1]+rng.normal(0,.55)
    available=rng.random(n)<.72; news[available]=np.tanh(latent[available]+rng.normal(0,.30,available.sum()))
    count[available]=rng.integers(1,5,available.sum()); ret=np.zeros(n); vol=np.full(n,.009)
    for t in range(1,n):
        event=fomc[t-1]+earnings[t-1]
        vol[t]=np.clip(.0015+.78*vol[t-1]+.13*abs(ret[t-1])+.0025*event,.004,.04)
        ret[t]=.00015+.10*ret[t-1]+.0035*news[t-1]*(1+1.2*event)+rng.normal(0,vol[t])
    close=100*np.exp(np.cumsum(ret)); open_=close*np.exp(rng.normal(0,.002,n)); spread=np.abs(rng.normal(.004,.002,n))
    high=np.maximum(open_,close)*(1+spread); low=np.minimum(open_,close)*(1-spread)
    volume=np.exp(15.5+.18*np.abs(ret)/(vol+1e-6)+.25*(fomc+earnings)+rng.normal(0,.25,n))
    df=pd.DataFrame({'open':open_,'high':high,'low':low,'close':close,'volume':volume,'return_1d':ret,
      'volatility_20d':pd.Series(ret,index=dates).rolling(20).std().bfill(),'sentiment':news,'news_count':count,
      'no_news':(count==0).astype(float),'fomc_flag':fomc,'earnings_flag':earnings},index=dates)
    df['dow_sin']=np.sin(2*np.pi*df.index.dayofweek/5); df['dow_cos']=np.cos(2*np.pi*df.index.dayofweek/5)
    df['month_end']=df.index.is_month_end.astype(float); df['target_5d_pct']=100*(df.close.shift(-5)/df.close-1)
    return df.dropna()

FEATURES=['open','high','low','close','volume','return_1d','volatility_20d','sentiment','news_count','no_news','fomc_flag','earnings_flag','dow_sin','dow_cos','month_end']
LOOKBACK=60; HORIZON=5; df=make_synthetic_market(); n=len(df)
train_end=int(n*.68); valid_start=train_end+HORIZON; valid_end=int(n*.83); test_start=valid_end+HORIZON
scaler=StandardScaler().fit(df.iloc[:train_end+1][FEATURES]); Xall=scaler.transform(df[FEATURES]).astype('float32')
yall=df.target_5d_pct.to_numpy('float32')
class WindowDataset(Dataset):
    def __init__(self,origins): self.origins=np.asarray(origins,dtype=int)
    def __len__(self): return len(self.origins)
    def __getitem__(self,i):
        t=self.origins[i]; return torch.from_numpy(Xall[t-LOOKBACK+1:t+1]),torch.tensor(yall[t]),t
train_idx=np.arange(LOOKBACK-1,train_end-HORIZON+1); valid_idx=np.arange(valid_start,valid_end-HORIZON+1)
test_idx=np.arange(test_start,n-HORIZON); train_ds,valid_ds,test_ds=map(WindowDataset,(train_idx,valid_idx,test_idx))
print('date split',df.index[train_idx[-1]],df.index[valid_idx[0]],df.index[test_idx[0]])
# BLOCK DATA END

# BLOCK TFT START
class VariableSelectionNetwork(nn.Module):
    def __init__(self,n_features,d_model):
        super().__init__(); self.embed=nn.ModuleList([nn.Linear(1,d_model) for _ in range(n_features)])
        self.selector=nn.Sequential(nn.Linear(n_features,d_model),nn.ELU(),nn.Linear(d_model,n_features))
    def forward(self,x):
        w=torch.softmax(self.selector(x),-1); v=torch.stack([f(x[...,j:j+1]) for j,f in enumerate(self.embed)],-2)
        return (w.unsqueeze(-1)*v).sum(-2),w
class GatedResidual(nn.Module):
    def __init__(self,d): super().__init__(); self.v=nn.Linear(d,d); self.g=nn.Linear(d,d); self.n=nn.LayerNorm(d)
    def forward(self,x,res): return self.n(res+torch.sigmoid(self.g(x))*self.v(x))
class CompactTFT(nn.Module):
    """One-horizon educational TFT subset, not the complete multi-horizon TFT."""
    def __init__(self,n_features,d=32,heads=4,dropout=.1):
        super().__init__(); self.vsn=VariableSelectionNetwork(n_features,d); self.pos=SinusoidalPositionEncoding(d,LOOKBACK)
        self.lstm=nn.LSTM(d,d,batch_first=True); self.local=GatedResidual(d)
        self.attn=nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True); self.ag=GatedResidual(d)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*d,d)); self.fg=GatedResidual(d)
        self.out=nn.Linear(d,3)
    def forward(self,x,return_weights=False):
        s,vw=self.vsn(x); z=self.pos(s); l,_=self.lstm(z); l=self.local(l,z); T=x.size(1)
        mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=x.device),1)
        a,aw=self.attn(l,l,l,attn_mask=mask,need_weights=True,average_attn_weights=False)
        z=self.ag(a,l); z=self.fg(self.ff(z),z); raw=self.out(z[:,-1]); q50=raw[:,1]
        pred=torch.stack([q50-nn.functional.softplus(raw[:,0]),q50,q50+nn.functional.softplus(raw[:,2])],1)
        return (pred,vw,aw) if return_weights else pred
def pinball_loss(pred,y,quantiles=(.1,.5,.9)):
    e=y[:,None]-pred; q=torch.tensor(quantiles,device=pred.device)[None,:]
    return torch.maximum(q*e,(q-1)*e).mean()

model=CompactTFT(len(FEATURES)).to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
train_loader=DataLoader(train_ds,64,shuffle=True); valid_loader=DataLoader(valid_ds,128)
best=copy.deepcopy(model.state_dict()); best_loss=float('inf'); stale=0; history=[]
for epoch in range(1,51):
    model.train(); tr=[]
    for xb,yb,_ in train_loader:
        xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(); loss=pinball_loss(model(xb),yb)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tr.append(loss.item())
    model.eval(); va=[]
    with torch.inference_mode():
        for xb,yb,_ in valid_loader: va.append(pinball_loss(model(xb.to(DEVICE)),yb.to(DEVICE)).item())
    row=(epoch,float(np.mean(tr)),float(np.mean(va))); history.append(row)
    if row[2]<best_loss-1e-5: best_loss=row[2]; best=copy.deepcopy(model.state_dict()); stale=0
    else: stale+=1
    if epoch==1 or epoch%5==0: print('epoch/train/valid',row)
    if stale>=8: break
model.load_state_dict(best)
# BLOCK TFT END

# BLOCK EVALUATION START
test_loader=DataLoader(test_ds,128); preds=[]; ys=[]; origins=[]; attns=[]; varws=[]; model.eval()
with torch.inference_mode():
    for xb,yb,idx in test_loader:
        p,v,a=model(xb.to(DEVICE),True); preds.append(p.cpu().numpy()); ys.append(yb.numpy()); origins.append(idx.numpy())
        attns.append(a[:,:,-1,:].cpu().numpy()); varws.append(v.cpu().numpy())
pred=np.concatenate(preds); y=np.concatenate(ys); origins=np.concatenate(origins); attn=np.concatenate(attns); varw=np.concatenate(varws)
med=pred[:,1]; metrics={'MAE_pct_point':mean_absolute_error(y,med),'RMSE_pct_point':mean_squared_error(y,med)**.5,
 'Hit_ratio':np.mean(np.sign(y)==np.sign(med)),'P10_P90_coverage':np.mean((y>=pred[:,0])&(y<=pred[:,2])),
 'Mean_interval_width':np.mean(pred[:,2]-pred[:,0])}; print('TEST',metrics)
importance=pd.Series(varw.mean(axis=(0,1)),index=FEATURES).sort_values(ascending=False); print('VARIABLE SELECTION\n',importance)
event_weights=[]; normal_weights=[]
for b,t in enumerate(origins):
    win=df.iloc[t-LOOKBACK+1:t+1]; event=((win.fomc_flag+win.earnings_flag)>0).to_numpy(); w=attn[b].mean(0)
    event_weights.extend(w[event]); normal_weights.extend(w[~event])
attention_report={'event_mean_weight':float(np.mean(event_weights)),'normal_mean_weight':float(np.mean(normal_weights))}
attention_report['ratio']=attention_report['event_mean_weight']/attention_report['normal_mean_weight']; print('ATTENTION',attention_report)
sent_col=FEATURES.index('sentiment'); counter=[]
with torch.inference_mode():
    for xb,_,_ in test_loader:
        xb=xb.to(DEVICE); xb[:,:,sent_col]=0; counter.append(model(xb)[:,1].cpu().numpy())
counter=np.concatenate(counter); print('sentiment zero mean abs prediction change',np.mean(np.abs(med-counter)))
# BLOCK EVALUATION END

# BLOCK FIGURES START
plt.style.use('seaborn-v0_8-whitegrid'); h=np.asarray(history)
fig,ax=plt.subplots(figsize=(10,4)); ax.plot(h[:,0],h[:,1],label='train'); ax.plot(h[:,0],h[:,2],label='validation'); ax.set(xlabel='epoch',ylabel='pinball loss',title='Compact TFT learning curve'); ax.legend(); fig.tight_layout(); fig.savefig(ASSET/'fig01_learning.png',dpi=170); plt.close(fig)
fig,ax=plt.subplots(figsize=(11,4)); take=min(120,len(y)); d=df.index[origins[:take]]; ax.plot(d,y[:take],label='actual',lw=1); ax.plot(d,med[:take],label='q50',lw=1.3); ax.fill_between(d,pred[:take,0],pred[:take,2],alpha=.2,label='q10-q90'); ax.set_ylabel('%'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(ASSET/'fig02_forecast.png',dpi=170); plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5)); importance.sort_values().plot.barh(ax=ax,color='#4677b5'); ax.set(title='Mean variable-selection weight',xlabel='softmax weight'); fig.tight_layout(); fig.savefig(ASSET/'fig03_variable_selection.png',dpi=170); plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4)); ax.bar(['event','ordinary'],[attention_report['event_mean_weight'],attention_report['normal_mean_weight']],color=['#d65f5f','#4c72b0']); ax.set(ylabel='mean final-query attention',title='Attention diagnostic: association, not causality'); fig.tight_layout(); fig.savefig(ASSET/'fig04_event_attention.png',dpi=170); plt.close(fig)
fig,ax=plt.subplots(figsize=(13,5)); ax.axis('off')
boxes=[(.02,.58,.16,.24,'Headlines / filings\n(timestamped)'),(.22,.58,.16,.24,'FinBERT\np(pos)-p(neg)'),(.02,.15,.16,.24,'OHLCV / volatility\ncalendar / events'),(.43,.36,.16,.28,'Variable selection\n+ position encoding'),(.64,.36,.14,.28,'LSTM\nlocal patterns'),(.82,.36,.16,.28,'Causal MHA\nlong memory'),(.64,.03,.34,.18,'Gated residual + quantile head\nq10, q50, q90')]
for x,y0,w,h0,label in boxes: ax.add_patch(plt.Rectangle((x,y0),w,h0,facecolor='#eaf1f8',edgecolor='#244a6b',lw=1.5)); ax.text(x+w/2,y0+h0/2,label,ha='center',va='center',fontsize=10)
for a,b in [((.18,.70),(.22,.70)),((.38,.70),(.43,.52)),((.18,.27),(.43,.45)),((.59,.50),(.64,.50)),((.78,.50),(.82,.50)),((.90,.36),(.82,.21))]: ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.7,color='#555'))
ax.set_title('Structured prices + unstructured text: multimodal Compact TFT',fontsize=14); fig.tight_layout(); fig.savefig(ASSET/'fig05_architecture.png',dpi=170,bbox_inches='tight'); plt.close(fig)
pd.DataFrame({'date':df.index[origins],'actual':y,'q10':pred[:,0],'q50':med,'q90':pred[:,2]}).to_csv(ROOT/'test_predictions.csv',index=False)
pd.DataFrame(history,columns=['epoch','train_loss','valid_loss']).to_csv(ROOT/'training_history.csv',index=False)
importance.rename('mean_weight').to_csv(ROOT/'variable_selection_weights.csv'); print('saved to',ROOT)
result={'python':platform.python_version(),'torch':torch.__version__,'device':str(DEVICE),'seed':SEED,
 'samples':{'train':len(train_ds),'validation':len(valid_ds),'test':len(test_ds)},
 'date_split':{'train_origin_end':str(df.index[train_idx[-1]].date()),'validation_origin_start':str(df.index[valid_idx[0]].date()),'test_origin_start':str(df.index[test_idx[0]].date())},
 'metrics':{k:float(v) for k,v in metrics.items()},'attention':attention_report,
 'sentiment_zero_mean_abs_change':float(np.mean(np.abs(med-counter))),
 'top_variables':{k:float(v) for k,v in importance.head(8).items()},'epochs':len(history),
 'synthetic_data':True,'finbert_default_run':False}
(ROOT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
# BLOCK FIGURES END
