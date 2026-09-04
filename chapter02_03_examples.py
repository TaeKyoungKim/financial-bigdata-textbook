"""Financial ML Chapters 02-03. All data are synthetic. Run top to bottom."""
# BLOCK imports START
from pathlib import Path
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, Binarizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve, brier_score_loss)

OUT = Path(__file__).resolve().parent
ASSETS = OUT / 'assets'
ASSETS.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
    'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'savefig.dpi': 160})
FEATURES = ['Debt_Ratio', 'Current_Ratio', 'Operating_Margin', 'Interest_Coverage']
def save(fig, filename):
    fig.tight_layout(pad=2)
    fig.savefig(ASSETS / filename)
    plt.close(fig)
def emit(name, value):
    print(name)
    print(value.to_string(index=False) if isinstance(value, pd.DataFrame) else value)
# BLOCK imports END

# BLOCK companies START
def make_companies(n_companies=200, n_periods=12, seed=2026):
    rng = np.random.default_rng(seed)
    company_risk = rng.normal(0, 0.7, n_companies)
    frames = []
    for period in range(n_periods):
        risk = company_risk + rng.normal(0, 0.5, n_companies) + 0.035 * period
        equity = rng.lognormal(5, 0.7, n_companies)  # positive, billion KRW
        debt_ratio = 55 * (1 + rng.pareto(1.8, n_companies)) * np.exp(0.3*risk)
        liabilities = equity * debt_ratio / 100
        current_liabilities = liabilities * rng.uniform(0.25, 0.65, n_companies)
        current_ratio = 100 * np.exp(0.5 - 0.3*risk + rng.normal(0, 0.35, n_companies))
        current_assets = current_liabilities * current_ratio / 100
        revenue = (equity + liabilities) * rng.uniform(0.35, 1.1, n_companies)
        op_margin = 7 - 3.2*risk + rng.normal(0, 4, n_companies)
        operating_profit = revenue * op_margin / 100
        interest_expense = liabilities * rng.uniform(0.02, 0.07, n_companies)
        coverage = operating_profit / interest_expense
        score = (-1.8 + 0.8*np.log(debt_ratio/100)
                 - 0.9*np.log(current_ratio/100) - 0.09*op_margin
                 - 0.35*np.arcsinh(coverage) + 0.04*period)
        probability = 1 / (1 + np.exp(-score))
        distress = rng.binomial(1, probability)
        available = pd.Timestamp(year=2012+period, month=5, day=1)
        frame = pd.DataFrame({
            'company_id': [f'SYN{i:03d}' for i in range(n_companies)],
            'period': period, 'available_at': available,
            'label_end': available + pd.DateOffset(years=1),
            'Equity': equity, 'Liabilities': liabilities,
            'Current_Liabilities': current_liabilities, 'Current_Assets': current_assets,
            'Revenue': revenue, 'Operating_Profit': operating_profit,
            'Interest_Expense': interest_expense,
            'Debt_Ratio': 100*liabilities/equity,
            'Current_Ratio': 100*current_assets/current_liabilities,
            'Operating_Margin': 100*operating_profit/revenue,
            'Interest_Coverage': coverage, 'distress_next_year': distress})
        # Simulated missing inputs, applied after latent outcome generation.
        for col in FEATURES:
            frame.loc[rng.random(n_companies) < 0.025, col] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ['available_at', 'company_id']).reset_index(drop=True)

companies = make_companies()
snapshot = companies.loc[companies['period'] == 11].copy()
assert snapshot['company_id'].nunique() == 200 and len(snapshot) == 200
assert len(companies) == 2400
companies.to_csv(OUT/'companies_panel_2400.csv', index=False, encoding='utf-8-sig')
snapshot.to_csv(OUT/'companies_snapshot_200.csv', index=False, encoding='utf-8-sig')
company_summary = companies.groupby('period', as_index=False).agg(
    rows=('company_id','size'), distressed=('distress_next_year','sum'))
emit('Company panel', company_summary)
# BLOCK companies END

# BLOCK folds START
# Final test: periods 10,11. Period 9 excluded because its label ends at test start.
development = companies.loc[companies['period'] <= 8].reset_index(drop=True)
company_test = companies.loc[companies['period'] >= 10].reset_index(drop=True)
unique_dates = np.sort(development['available_at'].unique())
time_cv = TimeSeriesSplit(n_splits=3, test_size=2, gap=1)
folds, fold_rows = [], []
for fold, (past_dates, next_dates) in enumerate(time_cv.split(unique_dates), start=1):
    train_rows = np.flatnonzero(development['available_at'].isin(unique_dates[past_dates]))
    valid_rows = np.flatnonzero(development['available_at'].isin(unique_dates[next_dates]))
    train = development.iloc[train_rows]
    valid = development.iloc[valid_rows]
    assert train['available_at'].max() < valid['available_at'].min()
    assert train['label_end'].max() < valid['available_at'].min()
    assert set(train['available_at']).isdisjoint(set(valid['available_at']))
    assert train['distress_next_year'].nunique() == 2
    assert valid['distress_next_year'].nunique() == 2
    folds.append((train_rows, valid_rows))
    fold_rows.append({'fold':fold, 'train_periods':','.join(map(str,past_dates)),
        'valid_periods':','.join(map(str,next_dates)),
        'train_rows':len(train_rows), 'valid_rows':len(valid_rows)})
assert development['label_end'].max() < company_test['available_at'].min()
fold_table = pd.DataFrame(fold_rows)
emit('Expanding window', fold_table)

# Audit a shuffled K-Fold; this is a deliberately invalid deployment evaluation.
bad_cv = KFold(n_splits=3, shuffle=True, random_state=2026)
bad_rows=[]
for fold,(tr,va) in enumerate(bad_cv.split(development),start=1):
    train_dates=development.iloc[tr]['available_at'].to_numpy()
    valid_dates=development.iloc[va]['available_at'].to_numpy()
    future_pairs = int((train_dates[:,None] > valid_dates[None,:]).sum())
    immature_pairs = int((development.iloc[tr]['label_end'].to_numpy()[:,None]
                          >= valid_dates[None,:]).sum())
    bad_rows.append({'fold':fold,'future_feature_pairs':future_pairs,
        'unavailable_label_pairs':immature_pairs,'all_pairs':len(tr)*len(va)})
bad_table=pd.DataFrame(bad_rows)
emit('Shuffled KFold time violations',bad_table)
# BLOCK folds END

# BLOCK leakage START
def ar1_leakage_demo(n=100000, rho=0.8, sigma=1.0, seed=812):
    rng=np.random.default_rng(seed)
    previous=rng.normal(0,sigma/np.sqrt(1-rho*rho),n)
    current=rho*previous+rng.normal(0,sigma,n)
    future=rho*current+rng.normal(0,sigma,n)
    past_prediction=rho*previous
    leaked_prediction=rho/(1+rho*rho)*(previous+future)
    return pd.DataFrame({
        'estimator':['past only','past + forbidden future'],
        'theoretical_MSE':[sigma*sigma,sigma*sigma/(1+rho*rho)],
        'empirical_MSE':[np.mean((current-past_prediction)**2),
                         np.mean((current-leaked_prediction)**2)]})
leakage_table=ar1_leakage_demo()
emit('AR1 conditional error',leakage_table)
# BLOCK leakage END

# BLOCK scalers START
def scaling_demo(seed=92):
    rng=np.random.default_rng(seed)
    clean=50*(1+rng.pareto(1.5,200))
    contaminated=clean.copy()
    contaminated[0]=1_000_000
    rows=[]
    probes=np.array([50.,100.,200.,1_000_000.]).reshape(-1,1)
    for label,values in [('clean',clean),('one extreme',contaminated)]:
        for name,scaler in [('Standard',StandardScaler()),('Robust',RobustScaler())]:
            scaler.fit(values.reshape(-1,1))
            center=scaler.mean_[0] if name=='Standard' else scaler.center_[0]
            transformed=scaler.transform(probes).ravel()
            rows.append({'sample':label,'scaler':name,'center':center,
                'scale':scaler.scale_[0],'z50':transformed[0],
                'z100':transformed[1],'z200':transformed[2],
                'z1000000':transformed[3],
                'middle_gap':transformed[2]-transformed[0]})
    return clean,contaminated,pd.DataFrame(rows)
clean_tail,dirty_tail,scaling_table=scaling_demo()
emit('Scaler sensitivity',scaling_table)
# BLOCK scalers END

# BLOCK credit_model START
credit_pipeline=Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scale',RobustScaler()),
    ('model',LogisticRegression(C=1.0,max_iter=4000,solver='lbfgs'))])
# A fresh imputer and scaler are fitted within every training fold.
credit_search=GridSearchCV(
    estimator=credit_pipeline,
    param_grid={'scale':[StandardScaler(),RobustScaler()],
                'model__C':[0.1,1.0,10.0]},
    scoring='average_precision',cv=folds,refit=True,n_jobs=1,error_score='raise')
credit_search.fit(development[FEATURES],development['distress_next_year'])
credit_cv=pd.DataFrame(credit_search.cv_results_)
credit_cv['scaler']=credit_cv['param_scale'].map(lambda v:type(v).__name__)
credit_cv_table=credit_cv[['scaler','param_model__C','mean_test_score','std_test_score']]
credit_prob=credit_search.predict_proba(company_test[FEATURES])[:,1]
credit_y=company_test['distress_next_year'].to_numpy()
credit_result={'best_scaler':type(credit_search.best_estimator_.named_steps['scale']).__name__,
    'best_C':float(credit_search.best_params_['model__C']),
    'test_AP':float(average_precision_score(credit_y,credit_prob)),
    'test_ROC_AUC':float(roc_auc_score(credit_y,credit_prob)),
    'test_prevalence':float(credit_y.mean())}
emit('Credit cross validation',credit_cv_table)
emit('Credit final test',credit_result)
# BLOCK credit_model END

# BLOCK fds_data START
FDS_FEATURES=['amount_log_z','velocity_z','distance_z','device_risk_z']
def make_fds_block(n=100000,seed=1,start='2025-01-01'):
    if n%1000 != 0:
        raise ValueError('Use a multiple of 1000 for exactly 0.1% fraud.')
    rng=np.random.default_rng(seed)
    y=np.zeros(n,dtype=int)
    y[rng.choice(n,size=n//1000,replace=False)]=1
    # Equal-covariance Gaussian class conditionals; no outcome-derived inputs.
    shift=np.array([2.0,1.4,1.2,0.8])
    X=rng.normal(size=(n,4))+y[:,None]*shift
    frame=pd.DataFrame(X,columns=FDS_FEATURES)
    frame['transaction_at']=pd.date_range(start,periods=n,freq='min')
    frame['label_available_at']=frame['transaction_at']+pd.Timedelta(days=7)
    frame['fraud']=y
    return frame
fds_train=make_fds_block(seed=11,start='2025-01-01')
fds_valid=make_fds_block(seed=22,start='2025-04-01')
fds_test=make_fds_block(seed=33,start='2025-07-01')
assert fds_train['label_available_at'].max() < fds_valid['transaction_at'].min()
assert fds_valid['label_available_at'].max() < fds_test['transaction_at'].min()
for name,frame in [('train',fds_train),('validation',fds_valid),('test',fds_test)]:
    assert frame['fraud'].sum()==100
    frame.to_csv(OUT/f'fds_{name}.csv',index=False,encoding='utf-8-sig')
# BLOCK fds_data END

# BLOCK metrics START
C_FP=20_000
C_FN=1_000_000
def classification_report_row(y_true,y_pred,probability,name):
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    return {'policy':name,'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),
        'accuracy':accuracy_score(y_true,y_pred),
        'precision':precision_score(y_true,y_pred,zero_division=0),
        'recall':recall_score(y_true,y_pred,zero_division=0),
        'F1':f1_score(y_true,y_pred,zero_division=0),
        'balanced_accuracy':balanced_accuracy_score(y_true,y_pred),
        'ROC_AUC':roc_auc_score(y_true,probability),
        'AP':average_precision_score(y_true,probability),
        'loss_KRW':int(C_FP*fp+C_FN*fn),
        'alerts':int(tp+fp)}
dummy=DummyClassifier(strategy='most_frequent')
dummy.fit(fds_train[FDS_FEATURES],fds_train['fraud'])
dummy_pred=dummy.predict(fds_test[FDS_FEATURES])
dummy_prob=dummy.predict_proba(fds_test[FDS_FEATURES])[:,1]
dummy_row=classification_report_row(fds_test['fraud'],dummy_pred,dummy_prob,'Dummy')
assert dummy_row['accuracy']==0.999
assert dummy_row['FN']==100 and dummy_row['TP']==0
assert dummy_row['loss_KRW']==100_000_000
emit('Dummy final test',dummy_row)
# BLOCK metrics END

# BLOCK threshold START
fds_model=Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scale',StandardScaler()),
    ('model',LogisticRegression(C=1.0,class_weight=None,max_iter=2000))])
fds_model.fit(fds_train[FDS_FEATURES],fds_train['fraud'])
valid_p=fds_model.predict_proba(fds_valid[FDS_FEATURES])[:,1]
valid_y=fds_valid['fraud'].to_numpy()
bayes_threshold=C_FP/(C_FP+C_FN)
thresholds=np.unique(np.r_[np.linspace(0,1,1001),
                          np.geomspace(1e-6,0.1,201),bayes_threshold])
grid_rows=[]
for threshold in thresholds:
    predicted=Binarizer(threshold=float(threshold)).transform(
        valid_p.reshape(-1,1)).astype(int).ravel()
    fp=int(np.sum((valid_y==0)&(predicted==1)))
    fn=int(np.sum((valid_y==1)&(predicted==0)))
    tp=int(np.sum((valid_y==1)&(predicted==1)))
    grid_rows.append({'threshold':float(threshold),'FP':fp,'FN':fn,'TP':tp,
        'loss_KRW':C_FP*fp+C_FN*fn,'alerts':int(predicted.sum())})
grid=pd.DataFrame(grid_rows)
# Predeclared tie break: minimum loss, fewer alerts, then larger threshold.
ranked_grid=grid.sort_values(['loss_KRW','alerts','threshold'],
                            ascending=[True,True,False])
selected_threshold=float(ranked_grid.iloc[0]['threshold'])
grid.to_csv(OUT/'threshold_grid_validation.csv',index=False,encoding='utf-8-sig')
emit('Validation top 10',ranked_grid.head(10))
# No refit after threshold selection: keep the same scoring model.
test_p=fds_model.predict_proba(fds_test[FDS_FEATURES])[:,1]
test_y=fds_test['fraud'].to_numpy()
test_rows=[dummy_row]
for name,threshold in [('fixed 0.5',0.5),('Bayes cost',bayes_threshold),
                       ('validation selected',selected_threshold)]:
    test_pred=Binarizer(threshold=threshold).transform(
        test_p.reshape(-1,1)).astype(int).ravel()
    row=classification_report_row(test_y,test_pred,test_p,name)
    row['threshold']=threshold
    test_rows.append(row)
fds_results=pd.DataFrame(test_rows)
assert np.array_equal(Binarizer(threshold=0.5).transform(
    np.array([[0.49],[0.5],[0.51]])).ravel(),[0,0,1])
emit('Frozen policies final test',fds_results)
emit('Validation Brier score',brier_score_loss(valid_y,valid_p))
# BLOCK threshold END

# BLOCK figures START
fig,axes=plt.subplots(1,2,figsize=(12,4))
for k,(tr,va) in enumerate(folds):
    axes[0].scatter(development.iloc[tr]['period'],np.full(len(tr),k),s=28,c='#235bb5')
    axes[0].scatter(development.iloc[va]['period'],np.full(len(va),k),s=28,c='#208980')
axes[0].set(xlabel='연간 관측 시점',ylabel='폴드',title='과거 학습(파랑), 이후 검증(초록)',yticks=[0,1,2])
axes[1].bar(leakage_table['estimator'],leakage_table['empirical_MSE'],color=['#235bb5','#d8794e'])
axes[1].set(ylabel='평균제곱오차',title='미래를 사용하면 평가 문제가 달라진다')
save(fig,'fig_02_01.png')
fig,axes=plt.subplots(1,2,figsize=(12,4))
v=np.sort(clean_tail)
axes[0].loglog(v,np.arange(len(v),0,-1)/len(v),'.',color='#235bb5')
axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
axes[0].set(xlabel='합성 부채비율',ylabel='경험적 초과확률',title='파레토 꼬리: 로그-로그 좌표')
subset=scaling_table.loc[scaling_table['sample']=='one extreme']
axes[1].bar(subset['scaler'],subset['middle_gap'],color=['#d8794e','#208980'])
axes[1].set(ylabel='변환 후 200과 50 사이 거리',title='한 개의 극단치가 중앙부 간격에 미치는 영향')
save(fig,'fig_02_02.png')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,row,title in [(axes[0],dummy_row,'항상 정상 예측'),
    (axes[1],test_rows[-1],'검증에서 선택한 임곗값')]:
    matrix=np.array([[row['TN'],row['FP']],[row['FN'],row['TP']]])
    ax.imshow(np.log1p(matrix),cmap='Blues')
    for (i,j),value in np.ndenumerate(matrix):
        ax.text(j,i,f'{value:,}',ha='center',va='center',fontsize=18,
                color='white' if np.log1p(value)>7 else '#172941')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['정상 예측','사기 예측'],
           yticklabels=['실제 정상','실제 사기'],title=title)
save(fig,'fig_03_01.png')
fig,axes=plt.subplots(1,2,figsize=(12,4))
precision,recall,_=precision_recall_curve(test_y,test_p)
fpr,tpr,_=roc_curve(test_y,test_p)
axes[0].plot(fpr,tpr,color='#235bb5');axes[0].plot([0,1],[0,1],'--',color='#aaa')
axes[0].set(xlabel='FPR',ylabel='TPR / Recall',title='최종 평가 ROC')
axes[1].plot(recall,precision,color='#208980')
axes[1].axhline(0.001,color='#d8794e',linestyle='--',label='사기 발생률 0.1%')
axes[1].set(xlabel='Recall',ylabel='Precision',title='최종 평가 PR');axes[1].legend()
save(fig,'fig_03_02.png')
fig,axes=plt.subplots(1,2,figsize=(12,4))
display=grid.loc[grid['threshold']>0].sort_values('threshold')
axes[0].semilogx(display['threshold'],display['loss_KRW']/1e6,color='#235bb5')
axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
axes[0].axvline(selected_threshold,color='#208980',linestyle='--',label='검증 선택')
axes[0].axvline(bayes_threshold,color='#d8794e',linestyle=':',label='이론 임곗값')
axes[0].set(xlabel='결정 임곗값',ylabel='검증 손실 (백만원)',title='검증 데이터에서만 임곗값 선택');axes[0].legend()
axes[1].bar(['Dummy','0.5','비용 이론','검증 선택'],fds_results['loss_KRW']/1e6,
            color=['#a3afbd','#235bb5','#d8794e','#208980'])
axes[1].set(ylabel='최종 평가 손실 (백만원)',title='정책을 고정한 뒤 최종 평가')
save(fig,'fig_03_03.png')
# BLOCK figures END

# BLOCK outputs START
result={'companies':company_summary.to_dict('records'),'folds':fold_table.to_dict('records'),
    'bad_folds':bad_table.to_dict('records'),'leakage':leakage_table.to_dict('records'),
    'scaling':scaling_table.to_dict('records'),'credit_cv':credit_cv_table.to_dict('records'),
    'credit_test':credit_result,'dummy':dummy_row,'selected_threshold':selected_threshold,
    'bayes_threshold':bayes_threshold,'grid_size':len(grid),
    'validation_top10':ranked_grid.head(10).to_dict('records'),
    'fds_test':fds_results.fillna(-1).to_dict('records'),
    'brier_validation':float(brier_score_loss(valid_y,valid_p)),
    'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','matplotlib']},
    'python':platform.python_version(),'checks':'passed'}
(OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'requirements.txt').write_text('\n'.join(k+'=='+v for k,v in result['versions'].items())+'\n',encoding='utf-8')
fds_results.to_csv(OUT/'fds_test_metrics.csv',index=False,encoding='utf-8-sig')
emit('Checks','passed')
# BLOCK outputs END
