# BLOCK setup START
from pathlib import Path
import json, platform, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

OUT=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
(OUT/'assets').mkdir(parents=True,exist_ok=True)
results={}
def records(frame): return json.loads(frame.to_json(orient='records',double_precision=12))
def savefig(name):
    plt.tight_layout();plt.savefig(OUT/'assets'/name,dpi=160);plt.close()
# BLOCK setup END

# BLOCK singular START
rng=np.random.default_rng(6001)
T,N=60,100
returns=rng.normal(0,.01,(T,N))
centered=returns-returns.mean(axis=0)
cov=centered.T@centered/(T-1)
eigenvalues,eigenvectors=np.linalg.eigh(cov)
rank=int(np.linalg.matrix_rank(cov))
assert rank==T-1
null_weight=eigenvectors[:,0];null_weight/=np.abs(null_weight).sum()
future=rng.normal(0,.01,(3000,N))
results['singular']=[dict(T=T,N=N,rank=rank,nullity=N-rank,
    sample_variance=float(null_weight@cov@null_weight),
    true_variance=float(.01**2*(null_weight@null_weight)),
    future_variance=float(np.var(future@null_weight,ddof=1)))]
results['distances']=[]
for p in [2,10,100,1000]:
    differences=rng.normal(0,np.sqrt(2),(2000,p))
    squared=np.sum(differences**2,axis=1)
    results['distances'].append(dict(dimension=p,squared_distance_CV=float(squared.std()/squared.mean()),
        theoretical_CV=float(np.sqrt(2/p))))
plt.figure(figsize=(9,3.8));plt.plot(np.arange(1,N+1),eigenvalues[::-1],'o',ms=3)
plt.axhline(0,color='black',lw=.8);plt.xlabel('Eigenvalue rank');plt.ylabel('Sample covariance eigenvalue')
plt.title('100 assets, 60 observations: rank at most 59');savefig('fig01_singular.png')
# BLOCK singular END

# BLOCK yield_data START
rng=np.random.default_rng(6002)
maturities=['3M','1Y','3Y','5Y','10Y','30Y']
years=np.array([.25,1,3,5,10,30])
templates=np.column_stack([np.ones(6),[-1,-.7,-.3,.1,.6,1],[-1,-.1,.9,1,.1,-1]])
basis,_=np.linalg.qr(templates)
for j in range(3):
    if basis[:,j]@templates[:,j]<0: basis[:,j]*=-1
assert np.allclose(basis.T@basis,np.eye(3))
days=1000
factors=rng.normal(size=(days,3))*np.array([5.,2.,1.])
changes=factors@basis.T+rng.normal(0,.15,(days,6))
dates=pd.bdate_range('2020-01-02',periods=days+1)
initial=np.array([2.3,2.4,2.6,2.8,3.,3.2])
levels=np.vstack([initial,initial+np.cumsum(changes,axis=0)/100])
yield_frame=pd.DataFrame(levels,index=dates,columns=maturities)
yield_frame.index.name='date'
delta=yield_frame.diff().dropna()*100
train=delta.iloc[:750].copy();test=delta.iloc[750:].copy()
assert train.index.max()<test.index.min()
yield_frame.to_csv(OUT/'synthetic_yields_percent.csv')
delta.to_csv(OUT/'yield_changes_bp.csv')
results['splits']=[dict(split=k,rows=len(d),start=str(d.index.min().date()),end=str(d.index.max().date()))
    for k,d in [('train',train),('future_test',test)]]
fig,axes=plt.subplots(1,2,figsize=(12,4))
for c in maturities: axes[0].plot(yield_frame.index,yield_frame[c],label=c,lw=.8)
axes[0].set_ylabel('Yield (%)');axes[0].legend(ncol=3,fontsize=8)
for idx in [0,250,500,750,1000]:
    axes[1].plot(years,yield_frame.iloc[idx],marker='o',label=str(yield_frame.index[idx].date()))
axes[1].set(xlabel='Maturity (years)',ylabel='Yield (%)');axes[1].legend(fontsize=8)
savefig('fig02_yields.png')
# BLOCK yield_data END

# BLOCK pca START
pca=PCA(n_components=6,svd_solver='full',whiten=False).fit(train)
vectors=pca.components_.T.copy()
for j in range(3):
    if vectors[:,j]@templates[:,j]<0: vectors[:,j]*=-1
Xc=train.to_numpy()-pca.mean_
scores=Xc@vectors
test_scores=(test.to_numpy()-pca.mean_)@vectors
sample_cov=Xc.T@Xc/(len(train)-1)
evalues=np.linalg.eigvalsh(sample_cov)[::-1]
assert np.allclose(evalues,pca.explained_variance_)
assert np.allclose(sample_cov@vectors,vectors*evalues)
assert np.allclose(np.cov(scores,rowvar=False),np.diag(evalues),atol=1e-10)
assert np.all(vectors[:,0]>0)
assert vectors[0,1]<0<vectors[-1,1]
assert vectors[0,2]<0 and vectors[-1,2]<0 and vectors[2,2]>0 and vectors[3,2]>0
ratios=pca.explained_variance_ratio_
results['explained']=[dict(PC=f'PC{j+1}',eigenvalue_bp2=float(evalues[j]),
    explained_pct=float(100*ratios[j]),cumulative_pct=float(100*ratios[:j+1].sum())) for j in range(6)]
loadings=pd.DataFrame(vectors[:,:3],index=maturities,columns=['PC1_Level','PC2_Slope','PC3_Curvature'])
results['loadings']=records(loadings.reset_index(names='maturity'))
loadings.to_csv(OUT/'yield_pca_axes.csv',index_label='maturity')
shock=vectors[:,:3]*np.sqrt(evalues[:3])
results['shocks']=records(pd.DataFrame(shock,index=maturities,columns=['Level_1SD_bp','Slope_1SD_bp','Curvature_1SD_bp']).reset_index(names='maturity'))
reconstruction=scores[:,:3]@vectors[:,:3].T+pca.mean_
test_reconstruction=test_scores[:,:3]@vectors[:,:3].T+pca.mean_
results['reconstruction']=[dict(split=k,RMSE_bp=float(np.sqrt(np.mean((a-b)**2))))
    for k,a,b in [('train',train.to_numpy(),reconstruction),('future_test',test.to_numpy(),test_reconstruction)]]
fig,axes=plt.subplots(1,3,figsize=(14,4))
for j,name in enumerate(['Level','Slope','Curvature']):
    axes[0].plot(np.arange(6),vectors[:,j],marker='o',label=f'PC{j+1} {name}')
    axes[1].plot(np.arange(6),shock[:,j],marker='o',label=name)
for ax in axes[:2]:ax.set_xticks(np.arange(6),maturities);ax.axhline(0,color='gray',lw=.7);ax.legend(fontsize=7)
axes[0].set_ylabel('Unit eigenvector coefficient');axes[1].set_ylabel('Positive 1-SD shock (bp)')
axes[2].bar(np.arange(1,7),ratios*100,label='Individual');axes[2].plot(np.arange(1,7),np.cumsum(ratios)*100,'ro-',label='Cumulative')
axes[2].set(xlabel='Principal component',ylabel='Explained variance (%)');axes[2].legend()
savefig('fig03_components.png')
# BLOCK pca END

# BLOCK risk START
portfolios={'Long_bonds':np.array([1000,2000,7000,10000,14000,18000]),
    'Butterfly':np.array([-3000,-6000,22000,18000,-8000,-18000])}
results['risk']=[]
results['risk_summary']=[]
for name,dv01 in portfolios.items():
    exposure=-vectors.T@dv01
    contribution=evalues*exposure**2
    direct=float(dv01@sample_cov@dv01)
    assert np.isclose(contribution.sum(),direct)
    for j in range(6):results['risk'].append(dict(portfolio=name,PC=f'PC{j+1}',
        exposure_currency_per_bp=float(exposure[j]),variance_share_pct=float(100*contribution[j]/direct)))
    results['risk_summary'].append(dict(portfolio=name,full_daily_SD=float(np.sqrt(direct)),
        top3_daily_SD=float(np.sqrt(contribution[:3].sum())),
        omitted_variance_pct=float(100*contribution[3:].sum()/direct),
        future_realized_SD=float(np.std(-test.to_numpy()@dv01,ddof=1))))
pd.DataFrame(results['risk']).to_csv(OUT/'portfolio_risk_components.csv',index=False)
# BLOCK risk END

# BLOCK svd START
U,s,Vt=np.linalg.svd(Xc,full_matrices=False)
assert np.allclose(U.T@U,np.eye(6)) and np.allclose(Vt@Vt.T,np.eye(6))
assert np.allclose((U*s)@Vt,Xc)
assert np.allclose(s**2/(len(train)-1),evalues)
rank3=(U[:,:3]*s[:3])@Vt[:3]
frobenius_squared=float(np.linalg.norm(Xc-rank3,'fro')**2)
assert np.isclose(frobenius_squared,np.sum(s[3:]**2))
svd=TruncatedSVD(n_components=3,algorithm='randomized',n_iter=10,random_state=42).fit(Xc)
projector=svd.components_.T@svd.components_
pca_projector=vectors[:,:3]@vectors[:,:3].T
assert np.allclose(projector,pca_projector,atol=1e-8)
raw_svd=TruncatedSVD(n_components=3,random_state=42).fit(yield_frame.iloc[:751])
centered_level_pca=PCA(n_components=3).fit(yield_frame.iloc[:751])
cosine=float(abs(raw_svd.components_[0]@centered_level_pca.components_[0]))
results['svd']=[dict(reconstruction_error_squared=frobenius_squared,
    discarded_singular_values_squared=float(np.sum(s[3:]**2)),
    projector_max_difference=float(np.max(np.abs(projector-pca_projector))),
    raw_SVD_vs_centered_level_PCA_axis_cosine=cosine)]
pd.DataFrame(scores[:,:3],index=train.index,columns=['Level','Slope','Curvature']).to_csv(OUT/'yield_pca_train_scores.csv')
# BLOCK svd END

# BLOCK rmt START
rng=np.random.default_rng(6003)
obs,assets=500,100
q=assets/(obs-1)
lower=(1-np.sqrt(q))**2;upper=(1+np.sqrt(q))**2
null_data=rng.normal(size=(obs,assets))
null_values=np.linalg.eigvalsh(np.corrcoef(null_data,rowvar=False))
groups=np.arange(assets)//20
true_corr=.15*np.ones((assets,assets))+.15*(groups[:,None]==groups[None,:])+.7*np.eye(assets)
chol=np.linalg.cholesky(true_corr)
raw=rng.normal(size=(obs,assets))@chol.T*.01
future_raw=rng.normal(size=(2000,assets))@chol.T*.01
std=raw.std(axis=0,ddof=1)
C=np.corrcoef(raw,rowvar=False)
lam,V=np.linalg.eigh(C)
signal=lam>upper
clean_lam=lam.copy()
if (~signal).any(): clean_lam[~signal]=lam[~signal].mean()
clean_pre=(V*clean_lam)@V.T
diagonal=np.sqrt(np.diag(clean_pre))
clean=clean_pre/np.outer(diagonal,diagonal)
assert np.allclose(np.diag(clean),1)
assert np.linalg.eigvalsh(clean).min()>0
assert np.isclose(clean_lam.sum(),lam.sum())
cov_raw=C*np.outer(std,std);cov_clean=clean*np.outer(std,std)
results['rmt']=[dict(q=q,MP_lower=lower,MP_upper=upper,
    candidate_signal_count=int(signal.sum()),noise_eigenvalue_mean=float(clean_lam[~signal][0]),
    null_outside_count=int(np.sum((null_values<lower)|(null_values>upper))))]
results['denoising']=[]
for name,corr,covariance in [('Sample',C,cov_raw),('Filtered',clean,cov_clean)]:
    ones=np.ones(assets);w=np.linalg.solve(covariance,ones);w/=ones@w
    results['denoising'].append(dict(model=name,correlation_Frobenius_error=float(np.linalg.norm(corr-true_corr,'fro')),
        condition_number=float(np.linalg.cond(covariance)),gross_exposure=float(np.abs(w).sum()),
        estimated_daily_SD_bp=float(np.sqrt(w@covariance@w)*10000),
        true_daily_SD_bp=float(np.sqrt(w@(.01**2*true_corr)@w)*10000),
        future_daily_SD_bp=float(np.std(future_raw@w,ddof=1)*10000)))
fig,axes=plt.subplots(1,2,figsize=(12,4))
grid=np.linspace(lower+1e-6,upper-1e-6,400)
density=np.sqrt((upper-grid)*(grid-lower))/(2*np.pi*q*grid)
axes[0].hist(null_values,bins=20,density=True,alpha=.5,label='IID sample correlation')
axes[0].plot(grid,density,'r',label='MP null density, variance = 1')
axes[0].set(xlabel='Eigenvalue',ylabel='Density');axes[0].legend(fontsize=8)
axes[1].semilogy(np.arange(1,assets+1),lam[::-1],label='Sample')
axes[1].semilogy(np.arange(1,assets+1),np.linalg.eigvalsh(clean)[::-1],label='Filtered + unit diagonal')
axes[1].axhline(upper,color='red',ls='--',label='MP null upper edge')
axes[1].set(xlabel='Eigenvalue rank',ylabel='Eigenvalue (log)');axes[1].legend(fontsize=8)
savefig('fig04_rmt.png')
pd.DataFrame(clean).to_csv(OUT/'denoised_correlation.csv',index=False)
# BLOCK rmt END

# BLOCK lda START
rng=np.random.default_rng(6004)
count=1500
rating=rng.integers(0,3,count)
rating_names=np.array(['AAA','BBB','B'])
centers=np.array([[-1.2,-.6],[0,1.2],[1.2,-.6]])
credit=centers[rating]+rng.normal(0,.45,(count,2))
size=rng.normal(0,3,count);cycle=rng.normal(0,3,count)
features=pd.DataFrame({
    'DebtRatio':120+20*size+8*credit[:,0],
    'InterestBurden':30+5*size-2*credit[:,0],
    'CurrentRatio':180+25*cycle+10*credit[:,1],
    'CashCoverage':60+8*cycle-3.2*credit[:,1],
    'LogAssets':25+size+rng.normal(0,.1,count),
    'SalesCycleScore':cycle+rng.normal(0,.1,count)})
features['date']=pd.bdate_range('2017-01-02',periods=count)
features['rating']=rating_names[rating]
columns=list(features.columns[:6])
a=features.iloc[:1000];b=features.iloc[1000:]
rating_models={
 'All six features':Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(C=10,max_iter=3000))]),
 'PCA2 + logistic':Pipeline([('scale',StandardScaler()),('reduce',PCA(n_components=2)),('classifier',LogisticRegression(C=10,max_iter=3000))]),
 'LDA2 + logistic':Pipeline([('scale',StandardScaler()),('reduce',LinearDiscriminantAnalysis(n_components=2,solver='svd')),('classifier',LogisticRegression(C=10,max_iter=3000))])}
results['credit']=[]
for name,model in rating_models.items():
    model.fit(a[columns],a.rating)
    prediction=model.predict(b[columns])
    results['credit'].append(dict(model=name,accuracy=float(accuracy_score(b.rating,prediction)),
        balanced_accuracy=float(balanced_accuracy_score(b.rating,prediction)),
        macro_F1=float(f1_score(b.rating,prediction,average='macro'))))
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,name in zip(axes,['PCA2 + logistic','LDA2 + logistic']):
    model=rating_models[name]
    projection=model[:-1].transform(b[columns])
    for label in rating_names:
        mask=b.rating.eq(label).to_numpy();ax.scatter(projection[mask,0],projection[mask,1],s=12,alpha=.5,label=label)
    ax.set(title=name,xlabel='Axis 1',ylabel='Axis 2');ax.legend()
savefig('fig05_lda.png')
S=StandardScaler().fit_transform(a[columns]);labels=a.rating.to_numpy();mean=S.mean(axis=0)
SW=np.zeros((6,6));SB=np.zeros((6,6))
for label in np.unique(labels):
    group=S[labels==label];mu=group.mean(axis=0);diff=group-mu
    SW+=diff.T@diff;SB+=len(group)*np.outer(mu-mean,mu-mean)
assert np.allclose(SW+SB,(S-mean).T@(S-mean))
results['lda_diagnostics']=[dict(classes=3,SB_rank=int(np.linalg.matrix_rank(SB)),
    maximum_discriminant_dimensions=2,
    PCA2_explained_pct=float(rating_models['PCA2 + logistic'].named_steps['reduce'].explained_variance_ratio_.sum()*100))]
cm=confusion_matrix(b.rating,rating_models['LDA2 + logistic'].predict(b[columns]),labels=rating_names)
results['credit_confusion']=records(pd.DataFrame(cm,columns=['pred_AAA','pred_BBB','pred_B']).assign(actual=rating_names))
features.to_csv(OUT/'synthetic_credit_ratings.csv',index=False)
# BLOCK lda END

# BLOCK export START
results['python']=platform.python_version()
results['versions']={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','matplotlib']}
results['checks']=['centered_rank_T_minus_1','PCA_eigen_equation','PCA_scores_uncorrelated',
    'Level_Slope_Curvature_sign_patterns','risk_variance_additivity','SVD_reconstruction',
    'Eckart_Young_error','PCA_SVD_projector_equivalence','denoised_positive_definite',
    'denoised_unit_diagonal','LDA_scatter_decomposition','chronological_holdouts']
(OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in results['versions'].items())+'\n',encoding='utf-8')
for key in ['explained','loadings','risk_summary','rmt','denoising','credit']:
    print('\n'+key+'\n'+pd.DataFrame(results[key]).to_string(index=False))
print('\nAll numerical checks passed.')
# BLOCK export END
