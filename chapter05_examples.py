# BLOCK setup START
from pathlib import Path
import json, platform, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, LassoCV, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

OUT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
(OUT / 'assets').mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260904)
results = {}
def records(frame):
    return json.loads(frame.to_json(orient='records', date_format='iso'))
def savefig(name):
    plt.tight_layout(); plt.savefig(OUT / 'assets' / name, dpi=150); plt.close()
def metrics(name, y, pred, split='test', scale=10000):
    return dict(model=name, split=split, RMSE=float(np.sqrt(mean_squared_error(y,pred))*scale),
                MAE=float(mean_absolute_error(y,pred)*scale), R2=float(r2_score(y,pred)))
# BLOCK setup END

# BLOCK data START
n = 1700
dates = pd.bdate_range('2018-01-02', periods=n)
def ar_process(shape, rho):
    z = rng.normal(size=shape)
    for t in range(1, shape[0]):
        z[t] = rho*z[t-1] + np.sqrt(1-rho**2)*z[t]
    return z
z = ar_process((n,4), .8)
macro = pd.DataFrame({'GovYield':2.5+.4*z[:,0],
    'CallRate':1.5+.4*z[:,0]+.015*rng.normal(size=n),
    'FX':1250*np.exp(.02*z[:,1]+.005*z[:,0]),
    'Oil':80*np.exp(.15*z[:,2]+.03*z[:,1]),
    'VIX':20*np.exp(.2*z[:,3]+.03*z[:,2])}, index=dates)
factor_names = [f'Factor_{j:02d}' for j in range(1,51)]
true_w = np.zeros(50); true_w[:5] = [.0013,-.0011,.0009,.0007,-.0006]
rf, premium = .00005, .0002
market = rf + premium + .008*rng.normal(size=n)
def add_targets(frame):
    frame = frame.sort_values('date').copy()
    ret = frame['return']
    frame['HistVol20'] = ret.pow(2).rolling(20).mean().pow(.5)*np.sqrt(252)
    frame['Momentum20'] = frame['price'].pct_change(20)
    frame['AbsMomentum20'] = frame['Momentum20'].abs()
    frame['target_return_1d'] = ret.shift(-1)
    future = pd.concat([ret.shift(-k) for k in range(1,21)],axis=1)
    frame['target_vol20'] = np.sqrt(252*future.pow(2).mean(axis=1,skipna=False))
    frame['label_end_return'] = frame['date'].shift(-1)
    frame['label_end_vol'] = frame['date'].shift(-20)
    return frame
panels = []
for j, beta in enumerate([.8,.95,1.1,1.25,1.4]):
    f = ar_process((n,50), .3)
    mu = rf+.0001+beta*premium+f@true_w
    sigma = .006*np.exp(.2*z[:,3]+.15*np.abs(f[:,0]))
    ret = np.zeros(n)
    ret[1:] = rf+.0001+beta*(market[1:]-rf)+f[:-1]@true_w+sigma[:-1]*rng.normal(size=n-1)
    assert np.all(ret > -1)
    d = pd.DataFrame(f,columns=factor_names)
    d['date']=dates; d['stock']=f'LARGE_{j+1:02d}'
    d['return']=ret; d['price']=100*np.cumprod(1+ret)
    d['market_return']=market; d['oracle_mean']=mu; d['true_beta']=beta
    for c in macro: d[c]=macro[c].to_numpy()
    panels.append(add_targets(d))
panel = pd.concat(panels).sort_values(['date','stock']).reset_index(drop=True)
cal = panel[panel.date.between(dates[20],dates[199])].copy()
train = panel[panel.date.between(dates[200],dates[1199])].copy()
test = panel[panel.date.between(dates[1220],dates[n-21])].copy()
assert train.label_end_vol.max() < test.date.min()
assert test[['target_return_1d','target_vol20']].notna().all().all()
scaler = StandardScaler().fit(cal[factor_names])
X = scaler.transform(train[factor_names]); Xt = scaler.transform(test[factor_names])
y = train.target_return_1d.to_numpy(); yt = test.target_return_1d.to_numpy()
def date_folds(frame, gap=1):
    unique_dates = np.sort(frame.date.unique())
    folds=[]
    for a,b in TimeSeriesSplit(n_splits=4,test_size=150,gap=gap).split(unique_dates):
        ia=np.flatnonzero(frame.date.isin(unique_dates[a]))
        ib=np.flatnonzero(frame.date.isin(unique_dates[b]))
        label='label_end_return' if gap==1 else 'label_end_vol'
        assert frame.iloc[ia][label].max() < frame.iloc[ib].date.min()
        folds.append((ia,ib))
    return folds
cv=date_folds(train)
results['splits']=[dict(split=name,rows=len(d),start=str(d.date.min().date()),end=str(d.date.max().date()))
                   for name,d in [('calibration',cal),('development',train),('future_test',test)]]
results['folds']=[dict(fold=k+1,train_rows=len(a),valid_rows=len(b),
    train_end=str(train.iloc[a].date.max().date()),valid_start=str(train.iloc[b].date.min().date())) for k,(a,b) in enumerate(cv)]
panel.to_csv(OUT/'synthetic_stock_panel.csv',index=False)
# BLOCK data END

# BLOCK ols START
one = train.stock.eq('LARGE_01').to_numpy()
A = np.column_stack([np.ones(one.sum()), X[one,:5]])
b = y[one]
w_svd = np.linalg.lstsq(A,b,rcond=None)[0]
w_normal = np.linalg.solve(A.T@A,A.T@b)
assert np.allclose(w_svd,w_normal)
capm=[]
for stock,d in train.groupby('stock'):
    m=d.market_return.to_numpy()-rf; s=d['return'].to_numpy()-rf
    fit=LinearRegression().fit(m[:,None],s)
    beta_cov=np.cov(m,s,ddof=1)[0,1]/np.var(m,ddof=1)
    assert np.isclose(fit.coef_[0],beta_cov)
    capm.append(dict(stock=stock,true_beta=float(d.true_beta.iloc[0]),
        fitted_beta=float(fit.coef_[0]),alpha_daily=float(fit.intercept_)))
results['capm']=capm
results['ols']=[dict(term=t,normal=float(a),SVD=float(b)) for t,a,b in zip(['intercept']+factor_names[:5],w_normal,w_svd)]
# BLOCK ols END

# BLOCK gd START
def loss(w): return float(np.mean((A@w-b)**2)/2)
L=np.linalg.eigvalsh(A.T@A/len(A)).max()
histories={}; weights={}
for method,batch,epochs in [('batch',len(A),150),('SGD',1,150),('minibatch',32,300)]:
    random=np.random.default_rng(42); w=np.zeros(A.shape[1]); curve=[]; step=0
    for epoch in range(epochs):
        updates=1 if method=='batch' else int(np.ceil(len(A)/batch))
        for _ in range(updates):
            if method=='batch':
                gradient=A.T@(A@w-b)/len(A); eta=1/L
            else:
                idx=random.integers(0,len(A),size=batch)
                gradient=A[idx].T@(A[idx]@w-b[idx])/batch
                eta=(.02/L)/(1+step/100)**.75
            w-=eta*gradient; step+=1
        curve.append(loss(w))
    histories[method]=curve; weights[method]=w
results['gd']=[dict(method=k,objective=loss(v),distance_to_OLS=float(np.linalg.norm(v-w_svd))) for k,v in weights.items()]
results['gd'].append(dict(method='OLS',objective=loss(w_svd),distance_to_OLS=0.))
for name,h in histories.items(): plt.plot(np.arange(1,len(h)+1),h,label=name)
plt.axhline(loss(w_svd),color='black',ls='--',label='OLS optimum')
plt.xlabel('Epoch');plt.ylabel('RSS / (2n)');plt.legend();savefig('fig01_gradient.png')
# BLOCK gd END

# BLOCK vif START
macro_names=list(macro.columns)
def vif_table(frame):
    standardized=(frame-frame.mean())/frame.std(ddof=0)
    design=np.column_stack([np.ones(len(frame)),standardized.to_numpy()])
    return pd.DataFrame({'feature':frame.columns,
        'VIF':[float(variance_inflation_factor(design,i+1)) for i in range(frame.shape[1])]})
class VIFSelector(TransformerMixin,BaseEstimator):
    def __init__(self,threshold=10.): self.threshold=threshold
    def fit(self,X,y=None):
        X=pd.DataFrame(X).copy()
        self.columns_=[c for c in X if X[c].std(ddof=0)>1e-12]
        if not self.columns_: raise ValueError('All columns are constant')
        self.history_=[]
        while len(self.columns_)>1:
            table=vif_table(X[self.columns_]); row=table.loc[table.VIF.idxmax()]
            if row.VIF<=self.threshold: break
            self.history_.append(dict(removed=str(row.feature),VIF=float(row.VIF)))
            self.columns_.remove(row.feature)
        return self
    def transform(self,X): return pd.DataFrame(X)[self.columns_].copy()
macro_train=train.loc[one,macro_names]
macro_test=test.loc[test.stock.eq('LARGE_01'),macro_names]
vif_pipe=Pipeline([('impute',SimpleImputer().set_output(transform='pandas')),
    ('vif',VIFSelector(10)),('scale',StandardScaler()),('ols',LinearRegression())])
vif_pipe.fit(macro_train,b)
kept=vif_pipe.named_steps['vif'].columns_
results['vif_before']=records(vif_table(macro_train))
results['vif_removed']=vif_pipe.named_steps['vif'].history_
results['vif_after']=records(vif_table(macro_train[kept]))
results['vif_test']=[metrics('VIF + OLS',test.loc[test.stock.eq('LARGE_01'),'target_return_1d'],vif_pipe.predict(macro_test))]
Z=StandardScaler().fit_transform(macro_train)
delta=np.random.default_rng(7).normal(0,.0001,len(b))
results['stability']=[]
for name,model in [('OLS',LinearRegression()),('Ridge',Ridge(alpha=100))]:
    model.fit(Z,b); before=model.coef_.copy(); model.fit(Z,b+delta)
    results['stability'].append(dict(model=name,coefficient_change=float(np.linalg.norm(model.coef_-before))))
results['condition']={'X':float(np.linalg.cond(Z)),'XtX':float(np.linalg.cond(Z.T@Z)),
    'ridge_matrix':float(np.linalg.cond(Z.T@Z+100*np.eye(5)))}
plt.figure(figsize=(7,5));plt.imshow(macro_train.corr(),vmin=-1,vmax=1,cmap='coolwarm')
plt.xticks(range(5),macro_names);plt.yticks(range(5),macro_names);plt.colorbar(label='Correlation')
savefig('fig02_macro.png')
# BLOCK vif END

# BLOCK factors START
alphas=np.logspace(-5,-2,50)
lasso=LassoCV(alphas=alphas,cv=cv,max_iter=20000,tol=1e-8,n_jobs=-1).fit(X,y)
elastic=ElasticNetCV(l1_ratio=[.1,.5,.9,1.],alphas=alphas,cv=cv,max_iter=20000,tol=1e-8,n_jobs=-1).fit(X,y)
ridge=GridSearchCV(Ridge(),{'alpha':np.logspace(-2,4,20)},cv=cv,
    scoring='neg_mean_squared_error',n_jobs=-1).fit(X,y).best_estimator_
models={'OLS50':LinearRegression().fit(X,y),'Ridge':ridge,'LassoCV':lasso,'ElasticNetCV':elastic}
results['returns']=[metrics('Historical mean',yt,np.repeat(y.mean(),len(yt)))]
for name,model in models.items(): results['returns'].append(metrics(name,yt,model.predict(Xt)))
raw_coef=lasso.coef_/scaler.scale_
raw_intercept=float(lasso.intercept_-np.dot(raw_coef,scaler.mean_))
assert np.allclose(lasso.predict(Xt),test[factor_names].to_numpy()@raw_coef+raw_intercept)
coef=pd.DataFrame({'factor':factor_names,'standardized_coef':lasso.coef_,
    'raw_coef':raw_coef,'true_coef':true_w})
selected=coef[coef.standardized_coef.ne(0)].copy()
selected=selected.loc[selected.standardized_coef.abs().sort_values(ascending=False).index]
print('\nALL NONZERO LASSO COEFFICIENTS\n',selected.to_string(index=False))
results['selected']=records(selected)
chosen=lasso.coef_!=0; truth=true_w!=0
results['selection']={'lasso_alpha':float(lasso.alpha_),'elastic_alpha':float(elastic.alpha_),
    'l1_ratio':float(elastic.l1_ratio_),'ridge_alpha':float(ridge.alpha),
    'TP':int(np.sum(chosen&truth)),'FP':int(np.sum(chosen&~truth)),'FN':int(np.sum(~chosen&truth)),
    'raw_intercept':raw_intercept}
coef.to_csv(OUT/'factor_coefficients.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].bar(np.arange(50),lasso.coef_*10000,label='Lasso');ax[0].plot(np.arange(50),true_w*scaler.scale_*10000,'ro',ms=3,label='Truth in scaled units')
ax[0].set(xlabel='Factor index (0-based)',ylabel='Return contribution (bp)');ax[0].legend()
ax[1].semilogx(lasso.alphas_,lasso.mse_path_.mean(axis=1));ax[1].axvline(lasso.alpha_,ls='--')
ax[1].set(xlabel='Alpha',ylabel='Mean chronological CV MSE');savefig('fig03_lasso.png')
# BLOCK factors END

# BLOCK trees START
vol_features=['VIX','HistVol20','AbsMomentum20','GovYield','FX','Oil']
VX=train[vol_features]; VXt=test[vol_features]
vy=train.target_vol20; vyt=test.target_vol20
vol_models={'Linear':LinearRegression(),
    'Tree depth3':DecisionTreeRegressor(max_depth=3,min_samples_leaf=20,random_state=42),
    'Tree depth12':DecisionTreeRegressor(max_depth=12,min_samples_leaf=1,random_state=42),
    'GBM':GradientBoostingRegressor(loss='squared_error',n_estimators=100,max_depth=2,
        min_samples_leaf=20,learning_rate=.05,random_state=42)}
results['volatility']=[]
for name,model in vol_models.items():
    model.fit(VX,vy)
    for split,xx,yy in [('train',VX,vy),('test',VXt,vyt)]:
        row=metrics(name,yy,model.predict(xx),split,scale=100)
        row['negative_predictions']=int(np.sum(model.predict(xx)<0));results['volatility'].append(row)
tree=vol_models['Tree depth3']; leaves=tree.apply(VX)
assert all(np.isclose(tree.predict(VX)[leaves==leaf][0],vy.to_numpy()[leaves==leaf].mean()) for leaf in np.unique(leaves))
gbm=vol_models['GBM']; residual=vy.to_numpy()-float(gbm.init_.constant_[0,0])
first=gbm.estimators_[0,0]; leaf_id=first.apply(VX.to_numpy())
assert all(np.isclose(first.predict(VX.to_numpy())[leaf_id==leaf][0],residual[leaf_id==leaf].mean()) for leaf in np.unique(leaf_id))
focus=test.stock.eq('LARGE_01').to_numpy()
plt.figure(figsize=(11,4));plt.plot(test.loc[focus,'date'],vyt[focus]*100,label='Future 20d realized RMS',alpha=.7)
plt.plot(test.loc[focus,'date'],gbm.predict(VXt[focus])*100,label='GBM forecast')
plt.ylabel('Annualized volatility (%)');plt.legend();savefig('fig04_volatility.png')
# BLOCK trees END

# BLOCK curves START
random=np.random.default_rng(70)
signal=random.uniform(-3,3,700)
expected=.002*signal+.004*np.sin(1.5*signal)
observed=expected+random.normal(0,.008,700)
grid=np.linspace(-3,3,600)[:,None]
curve_models={'Linear':LinearRegression(),
 'Tree depth2':DecisionTreeRegressor(max_depth=2,random_state=42),
 'Tree depth12':DecisionTreeRegressor(max_depth=12,random_state=42),
 'GBM':GradientBoostingRegressor(n_estimators=100,max_depth=2,learning_rate=.05,random_state=42)}
results['curves']=[]
fig,axes=plt.subplots(1,2,figsize=(13,4))
axes[0].scatter(signal[:400],observed[:400]*10000,s=8,alpha=.15,label='Training returns')
truth_grid=.002*grid[:,0]+.004*np.sin(1.5*grid[:,0])
for ax in axes: ax.set_xlabel('Signal known at forecast time')
axes[0].plot(grid[:,0],truth_grid*10000,'k--',label='Conditional mean')
axes[1].plot(grid[:,0],100*(1+truth_grid),'k--',label='Conditional expected price')
for name,model in curve_models.items():
    model.fit(signal[:400,None],observed[:400]); prediction=model.predict(grid)
    for split,sl in [('train',slice(0,400)),('test',slice(400,None))]:
        results['curves'].append(metrics(name,observed[sl],model.predict(signal[sl,None]),split))
    axes[0].plot(grid[:,0],prediction*10000,label=name,alpha=.8)
    axes[1].plot(grid[:,0],100*(1+prediction),label=name,alpha=.8)
axes[0].set_ylabel('Next return (bp)');axes[1].set_ylabel('Expected next price, current price = 100')
axes[0].legend(fontsize=7);axes[1].legend(fontsize=7);savefig('fig05_steps.png')
# BLOCK curves END

# BLOCK export START
results['versions']={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','statsmodels','matplotlib']}
results['python']=platform.python_version()
results['checks']=['date_grouped_CV','mature_labels_before_validation','OLS_matches_SVD',
    'CAPM_matches_covariance','raw_scale_predictions_match','tree_leaf_mean','GBM_residual_leaf_mean']
(OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in results['versions'].items())+'\n',encoding='utf-8')
print('\nRETURN TEST METRICS (bp)\n',pd.DataFrame(results['returns']).to_string(index=False))
print('\nSELECTION AUDIT\n',results['selection'])
print('\nAll numerical and chronological checks passed.')
# BLOCK export END
