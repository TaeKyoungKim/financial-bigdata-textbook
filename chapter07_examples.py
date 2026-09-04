# BLOCK setup START
from pathlib import Path
import json, platform, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

OUT=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
(OUT/'assets').mkdir(parents=True,exist_ok=True)
results={}
def records(frame):return json.loads(frame.to_json(orient='records',double_precision=12))
def savefig(name):
    plt.tight_layout();plt.savefig(OUT/'assets'/name,dpi=160);plt.close()
# BLOCK setup END

# BLOCK assets START
rng=np.random.default_rng(7001)
T,N=750,20
tickers=[f'STOCK_{i+1:02d}' for i in range(N)]
groups=np.repeat(np.arange(4),5)
market=rng.normal(size=(T,1));sector=rng.normal(size=(T,4));noise=rng.normal(size=(T,N))
vol=np.repeat([.012,.018,.009,.016],5)
R=(np.sqrt(.15)*market+np.sqrt(.65)*sector[:,groups]+np.sqrt(.20)*noise)*vol+.0003
R[:,1]=R[:,0]+rng.normal(0,.0002,T)
returns=pd.DataFrame(R,index=pd.bdate_range('2019-01-02',periods=T),columns=tickers)
train=returns.iloc[:500];test=returns.iloc[500:]
assert train.index.max()<test.index.min()
correlation=train.corr().to_numpy();cov=train.cov().to_numpy()
D=np.sqrt(np.clip((1-correlation)/2,0,1));np.fill_diagonal(D,0)
asset_vectors=train.to_numpy().T.copy()
asset_vectors-=asset_vectors.mean(axis=1,keepdims=True)
asset_vectors/=2*np.linalg.norm(asset_vectors,axis=1,keepdims=True)
assert np.allclose(squareform(pdist(asset_vectors)),D,atol=1e-8)
results['k_search']=[];fits={}
for k in range(2,9):
    model=KMeans(n_clusters=k,n_init=50,random_state=42).fit(asset_vectors)
    score=silhouette_score(D,model.labels_,metric='precomputed')
    fits[k]=model
    results['k_search'].append(dict(k=k,inertia=float(model.inertia_),silhouette=float(score)))
best=max(results['k_search'],key=lambda row:(row['silhouette'],-row['k']))['k']
labels=fits[best].labels_
results['cluster_members']=records(pd.DataFrame({'stock':tickers,'cluster':labels,'synthetic_group_audit':groups}))
results['cluster_summary']={'best_k':int(best),'silhouette':float(silhouette_score(D,labels,metric='precomputed')),
    'ARI_against_generation':float(adjusted_rand_score(groups,labels))}
results['splits']=[dict(dataset='stocks',split=name,rows=len(d),start=str(d.index.min().date()),end=str(d.index.max().date())) for name,d in [('train',train),('test',test)]]
returns.to_csv(OUT/'synthetic_stock_returns.csv',index_label='date')
fig,ax=plt.subplots(1,2,figsize=(11,4));search=pd.DataFrame(results['k_search'])
ax[0].plot(search.k,search.inertia,'o-');ax[0].set(xlabel='k',ylabel='Inertia')
ax[1].plot(search.k,search.silhouette,'o-');ax[1].axvline(best,ls='--',color='red');ax[1].set(xlabel='k',ylabel='Mean silhouette (correlation distance)')
savefig('fig01_k_selection.png')
# BLOCK assets END

# BLOCK allocation START
tree=linkage(squareform(D,checks=True),method='single',optimal_ordering=True)
order=leaves_list(tree)
def cluster_variance(indices,covariance):
    sub=covariance[np.ix_(indices,indices)]
    w=1/np.diag(sub);w/=w.sum()
    return float(w@sub@w)
def hierarchical_weights(covariance,ordered_indices):
    w=np.ones(len(covariance));pending=[list(ordered_indices)]
    while pending:
        cluster=pending.pop()
        if len(cluster)<2:continue
        middle=len(cluster)//2;left=cluster[:middle];right=cluster[middle:]
        vl=cluster_variance(left,covariance);vr=cluster_variance(right,covariance)
        alpha=vr/(vl+vr)
        w[left]*=alpha;w[right]*=1-alpha
        pending.extend([left,right])
    return w
hrp=hierarchical_weights(cov,order)
mu=train.mean().to_numpy();gamma=10.
ones=np.ones(N)
gmv=np.linalg.solve(cov,ones);gmv/=gmv.sum()
def unrestricted_mv(expected):
    inv_mu=np.linalg.solve(cov,expected);inv_one=np.linalg.solve(cov,ones)
    lagrange=(ones@inv_mu-gamma)/(ones@inv_one)
    return (inv_mu-lagrange*inv_one)/gamma
def long_only_mv(expected):
    fit=minimize(lambda w:float(gamma/2*w@cov@w-expected@w),np.ones(N)/N,
        jac=lambda w:gamma*cov@w-expected,method='SLSQP',bounds=[(0,1)]*N,
        constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:ones},
        options={'ftol':1e-12,'maxiter':2000})
    if not fit.success:raise RuntimeError(fit.message)
    return fit.x
weights={'Equal':np.ones(N)/N,'GMV unrestricted':gmv,'MeanVariance unrestricted':unrestricted_mv(mu),
    'MeanVariance long-only':long_only_mv(mu),'Hierarchical':hrp}
results['allocation']=[]
for name,w in weights.items():
    assert np.isclose(w.sum(),1)
    future=test.to_numpy()@w
    results['allocation'].append(dict(model=name,max_absolute_weight=float(np.abs(w).max()),
        gross_exposure=float(np.abs(w).sum()),train_annual_vol_pct=float(np.sqrt(w@cov@w*252)*100),
        test_annual_vol_pct=float(future.std(ddof=1)*np.sqrt(252)*100)))
assert np.all(hrp>=0)
perturbed=mu.copy();perturbed[0]+=.0001
results['sensitivity']=[dict(model=name,L1_weight_change=float(np.abs(fun(perturbed)-fun(mu)).sum()))
    for name,fun in [('MeanVariance unrestricted',unrestricted_mv),('MeanVariance long-only',long_only_mv)]]
results['sensitivity'].append(dict(model='Hierarchical (does not use mean)',L1_weight_change=0.))
results['cov_condition']=float(np.linalg.cond(cov))
pd.DataFrame(weights,index=tickers).to_csv(OUT/'portfolio_weights.csv',index_label='stock')
results['weights']=records(pd.DataFrame({'stock':tickers,'hierarchical_weight':hrp,'long_only_MV_weight':weights['MeanVariance long-only']}))
fig,axes=plt.subplots(1,2,figsize=(13,5))
dendrogram(tree,labels=tickers,ax=axes[0],leaf_rotation=90);axes[0].set_ylabel('Correlation distance, single linkage')
im=axes[1].imshow(correlation[np.ix_(order,order)],vmin=-1,vmax=1,cmap='coolwarm')
axes[1].set_xticks(range(N),np.array(tickers)[order],rotation=90,fontsize=6);axes[1].set_yticks(range(N),np.array(tickers)[order],fontsize=6)
fig.colorbar(im,ax=axes[1]);savefig('fig02_hierarchy.png')
plt.figure(figsize=(11,4));x=np.arange(N)
plt.bar(x-.2,weights['MeanVariance long-only'],.4,label='Mean-variance long-only')
plt.bar(x+.2,hrp,.4,label='Hierarchical')
plt.xticks(x,tickers,rotation=90);plt.ylabel('Capital weight');plt.legend();savefig('fig03_allocation.png')
# BLOCK allocation END

# BLOCK market START
rng=np.random.default_rng(7002)
days=1700
transition_true=np.array([[.98,.018,.002],[.02,.965,.015],[.02,.05,.93]])
latent=np.zeros(days,dtype=int)
for t in range(1,days):latent[t]=rng.choice(3,p=transition_true[latent[t-1]])
daily_mu=np.array([.0012,.0,-.007]);daily_sigma=np.array([.003,.009,.028])
market_returns=daily_mu[latent]+daily_sigma[latent]*rng.normal(size=days)
market=pd.DataFrame({'return':market_returns,'latent_audit':latent},index=pd.bdate_range('2017-01-02',periods=days))
assert (market['return']>-1).all()
market['price']=100*(1+market['return']).cumprod()
market['return_pct']=market['return']*100
market['vol20_annual_pct']=market['return'].rolling(20).std(ddof=1)*np.sqrt(252)*100
market=market.dropna().copy()
development=market.iloc[:1100];future_market=market.iloc[1100:]
feature_names=['return_pct','vol20_annual_pct']
scaler=StandardScaler().fit(development[feature_names])
X=scaler.transform(development[feature_names]);Xfuture=scaler.transform(future_market[feature_names])
results['splits'] += [dict(dataset='market',split=name,rows=len(d),start=str(d.index.min().date()),end=str(d.index.max().date()))
    for name,d in [('train',development),('test',future_market)]]
market.to_csv(OUT/'synthetic_market_regimes.csv',index_label='date')
# BLOCK market END

# BLOCK kde START
sample=development.return_pct.to_numpy()
grid=np.linspace(sample.min()-2,sample.max()+2,1200)
bandwidths=[.15,.5,1.5]
results['bandwidth']=[]
fig,axes=plt.subplots(1,2,figsize=(12,4))
for h in bandwidths:
    u=(grid[:,None]-sample[None,:])/h
    density=np.exp(-u**2/2).mean(axis=1)/(h*np.sqrt(2*np.pi))
    axes[0].plot(grid,density,label=f'h={h} percentage points')
    tail=float(ndtr((-3-sample)/h).mean())
    ms=MeanShift(bandwidth=h,bin_seeding=True,max_iter=500).fit(sample[:,None])
    results['bandwidth'].append(dict(bandwidth_percentage_points=h,
        flat_kernel_MeanShift_clusters=int(len(ms.cluster_centers_)),
        Gaussian_KDE_probability_below_minus3pct=tail))
    axes[1].scatter(ms.cluster_centers_[:,0],np.repeat(h,len(ms.cluster_centers_)),s=35,label=f'h={h}')
def gaussian_mean_shift(seed,values,h,max_iter=1000,tol=1e-7):
    x=float(seed)
    for iteration in range(max_iter):
        logw=-.5*((x-values)/h)**2;w=np.exp(logw-logw.max())
        new=float(w@values/w.sum())
        if abs(new-x)<tol:return new,iteration+1,True
        x=new
    return x,max_iter,False
results['gaussian_modes']=[]
for seed in [-3.,0.,3.]:
    mode,iterations,converged=gaussian_mean_shift(seed,sample,.5)
    results['gaussian_modes'].append(dict(start_return_pct=seed,mode_return_pct=mode,iterations=iterations,converged=converged))
results['empirical_left_tail']=float(np.mean(sample<-3))
axes[0].set(xlabel='Daily market return (%)',ylabel='Gaussian KDE density');axes[0].legend(fontsize=7)
axes[1].set(xlabel='Flat-kernel MeanShift mode, daily return (%)',ylabel='Bandwidth (percentage points)')
axes[1].legend();savefig('fig04_bandwidth.png')
# BLOCK kde END

# BLOCK em START
def expectation(data,pi,means,covariances):
    log_joint=np.column_stack([np.log(pi[k])+multivariate_normal.logpdf(data,mean=means[k],cov=covariances[k]) for k in range(len(pi))])
    normalizer=logsumexp(log_joint,axis=1)
    return np.exp(log_joint-normalizer[:,None]),float(normalizer.sum())
def maximization(data,responsibilities):
    nk=responsibilities.sum(axis=0);pi=nk/len(data)
    means=responsibilities.T@data/nk[:,None];covs=[]
    for k in range(len(nk)):
        diff=data-means[k]
        covs.append((diff.T*responsibilities[:,k])@diff/nk[k])
    return pi,means,np.array(covs)
initial_means=KMeans(n_clusters=3,n_init=20,random_state=8).fit(X).cluster_centers_
pi=np.ones(3)/3;means=initial_means;covs=np.repeat(np.eye(2)[None,:,:],3,axis=0)
results['manual_em']=[]
resp,old_ll=expectation(X,pi,means,covs)
for iteration in range(1,31):
    pi,means,covs=maximization(X,resp)
    resp,new_ll=expectation(X,pi,means,covs)
    assert new_ll>=old_ll-1e-7
    assert np.allclose(resp.sum(axis=1),1)
    results['manual_em'].append(dict(iteration=iteration,log_likelihood=new_ll,improvement=new_ll-old_ll))
    old_ll=new_ll
# BLOCK em END

# BLOCK gmm START
gmm=GaussianMixture(n_components=3,covariance_type='full',n_init=20,max_iter=1000,
    reg_covar=1e-6,tol=1e-5,random_state=42).fit(X)
assert gmm.converged_
raw_means=scaler.inverse_transform(gmm.means_)
raw_covs=gmm.covariances_*scaler.scale_[None,:,None]*scaler.scale_[None,None,:]
crash=int(np.argmax(raw_means[:,1]));remaining=[k for k in range(3) if k!=crash]
bull=max(remaining,key=lambda k:raw_means[k,0]);side=next(k for k in remaining if k!=bull)
ordered=[bull,side,crash];names=['LowVol_Bull','Range','HighVol_Crash']
results['regime_fit']=[]
for name,k in zip(names,ordered):
    results['regime_fit'].append(dict(regime=name,component=int(k),weight=float(gmm.weights_[k]),
        mean_daily_return_pct=float(raw_means[k,0]),mean_annual_vol_pct=float(raw_means[k,1]),
        variance_return_pct2=float(raw_covs[k,0,0]),variance_vol_pct2=float(raw_covs[k,1,1]),
        covariance_return_vol_pct2=float(raw_covs[k,0,1])))
train_prob=gmm.predict_proba(X)[:,ordered];future_prob=gmm.predict_proba(Xfuture)[:,ordered]
assert np.allclose(future_prob.sum(axis=1),1)
train_state=train_prob.argmax(axis=1);future_state=future_prob.argmax(axis=1)
results['regime_empirical']=[]
for split,frame,hard in [('train',development,train_state),('test',future_market,future_state)]:
    for j,name in enumerate(names):
        selected=frame.iloc[np.flatnonzero(hard==j)]
        results['regime_empirical'].append(dict(split=split,regime=name,days=len(selected),fraction=len(selected)/len(frame),
            mean_daily_return_pct=float(selected.return_pct.mean()),variance_return_pct2=float(selected.return_pct.var(ddof=1)),
            mean_annual_vol_pct=float(selected.vol20_annual_pct.mean())))
results['gmm_diagnostics']={'converged':bool(gmm.converged_),'iterations':int(gmm.n_iter_),
    'train_average_loglik_standardized':float(gmm.score(X)),'test_average_loglik_standardized':float(gmm.score(Xfuture)),
    'future_uncertain_fraction':float(np.mean(future_prob.max(axis=1)<.6)),
    'label_shapes_match':bool(raw_means[bull,0]>0 and raw_means[crash,0]<0 and raw_means[bull,1]<raw_means[side,1]<raw_means[crash,1])}
prediction=future_market.copy()
for j,name in enumerate(names):prediction['prob_'+name]=future_prob[:,j]
prediction['regime']=np.array(names)[future_state]
prediction.to_csv(OUT/'future_regime_probabilities.csv',index_label='date')
fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
colors=['#2878b5','#d89521','#c44242']
axes[0].plot(future_market.index,future_market.price,color='gray',lw=.8)
for j,name in enumerate(names):
    mask=future_state==j;axes[0].scatter(future_market.index[mask],future_market.price[mask],s=9,color=colors[j],label=name)
axes[0].set_ylabel('Market index');axes[0].legend(ncol=3,fontsize=8)
axes[1].plot(future_market.index,future_market.vol20_annual_pct);axes[1].set_ylabel('Trailing volatility (%)')
axes[2].stackplot(future_market.index,future_prob.T,labels=names,colors=colors,alpha=.8);axes[2].set_ylabel('Posterior probability')
savefig('fig05_regimes.png')
# BLOCK gmm END

# BLOCK transitions START
counts=np.zeros((3,3),dtype=int)
for a,b in zip(train_state[:-1],train_state[1:]):counts[a,b]+=1
transition=(counts+1)/(counts.sum(axis=1,keepdims=True)+3)
assert np.allclose(transition.sum(axis=1),1)
results['transition']=records(pd.DataFrame(transition,columns=names).assign(from_regime=names))
results['transition_counts']=records(pd.DataFrame(counts,columns=names).assign(from_regime=names))
results['durations']=[dict(regime=name,geometric_expected_days=float(1/(1-transition[i,i]))) for i,name in enumerate(names)]
prior=np.bincount(train_state,minlength=3)/len(train_state)
one_step=future_prob[:-1]@transition
targets=future_state[1:]
nll=float(-np.log(np.maximum(one_step[np.arange(len(targets)),targets],1e-15)).mean())
base_nll=float(-np.log(prior[targets]).mean())
results['transition_evaluation']=[dict(model='Soft current GMM x hard-label transition',next_label_logloss=nll),
    dict(model='Training state frequencies',next_label_logloss=base_nll)]
results['warning']='Transition evaluation predicts the next GMM label, not an independently verified economic state.'
fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(transition,vmin=0,vmax=1,cmap='Blues')
for i in range(3):
    for j in range(3):ax.text(j,i,f'{transition[i,j]:.3f}',ha='center',va='center',color='white' if transition[i,j]>.5 else 'black')
ax.set_xticks(range(3),names,rotation=20,fontsize=8);ax.set_yticks(range(3),names,fontsize=8)
ax.set(xlabel='Next GMM state',ylabel='Current GMM state');fig.colorbar(im,ax=ax);savefig('fig06_transitions.png')
# BLOCK transitions END

# BLOCK export START
results['python']=platform.python_version()
results['versions']={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib']}
results['checks']=['correlation_distance_equals_normalized_Euclidean','chronological_splits',
    'portfolio_weights_sum_one','hierarchical_nonnegative','manual_EM_likelihood_monotone',
    'posterior_rows_sum_one','GMM_converged','transition_rows_sum_one']
(OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in results['versions'].items())+'\n',encoding='utf-8')
for key in ['k_search','allocation','bandwidth','regime_fit','transition']:
    print('\n'+key+'\n'+pd.DataFrame(results[key]).to_string(index=False))
print('\nGMM diagnostics:',results['gmm_diagnostics'])
print('All numerical checks passed.')
# BLOCK export END
