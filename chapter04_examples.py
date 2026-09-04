"""Chapter 04: six-feature synthetic corporate credit classification."""
# BLOCK imports START
from pathlib import Path
from itertools import combinations
from math import factorial, comb
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, brier_score_loss, log_loss)
import shap

OUT=Path(__file__).resolve().parent
ASSETS=OUT/'assets'
ASSETS.mkdir(exist_ok=True)
BASE_FEATURES=['Debt_Ratio','Current_Ratio','Operating_Margin','Interest_Coverage']
FEATURES=BASE_FEATURES+['ROA','Cashflow_to_Debt']
FEATURE_LABELS=['부채비율','유동비율','영업이익률','이자보상배율','영업이익/총자산','영업현금흐름/부채']
POLICY_THRESHOLD=0.20  # illustrative, declared before model fitting
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,
    'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
def emit(title,value):
    print(title)
    print(value.to_string(index=False) if isinstance(value,pd.DataFrame) else value)
def save(fig,name):
    fig.tight_layout(pad=2)
    fig.savefig(ASSETS/name)
    plt.close(fig)
# BLOCK imports END

# BLOCK dataset START
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
        for col in BASE_FEATURES:
            frame.loc[rng.random(n_companies) < 0.025, col] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ['available_at', 'company_id']).reset_index(drop=True)

companies=make_companies()
rng=np.random.default_rng(404)
assets=companies['Equity']+companies['Liabilities']
companies['ROA']=100*companies['Operating_Profit']/assets
companies['Operating_Cashflow']=companies['Operating_Profit']+rng.normal(0,0.035,len(companies))*assets
companies['Cashflow_to_Debt']=100*companies['Operating_Cashflow']/companies['Liabilities']
for col in ['ROA','Cashflow_to_Debt']:
    companies.loc[rng.random(len(companies))<0.025,col]=np.nan
train=companies.loc[companies['period']<=5].reset_index(drop=True)
valid=companies.loc[companies['period'].isin([7,8])].reset_index(drop=True)
test=companies.loc[companies['period'].isin([10,11])].reset_index(drop=True)
assert companies['company_id'].nunique()==200 and len(companies)==2400
assert train['label_end'].max()<valid['available_at'].min()
assert valid['label_end'].max()<test['available_at'].min()
imputer=SimpleImputer(strategy='median')
X_train=pd.DataFrame(imputer.fit_transform(train[FEATURES]),columns=FEATURES)
X_valid=pd.DataFrame(imputer.transform(valid[FEATURES]),columns=FEATURES)
X_test=pd.DataFrame(imputer.transform(test[FEATURES]),columns=FEATURES)
y_train=train['distress_next_year'].to_numpy()
y_valid=valid['distress_next_year'].to_numpy()
y_test=test['distress_next_year'].to_numpy()
split_summary=pd.DataFrame([{'split':name,'rows':len(frame),
    'start':str(frame['available_at'].min().date()),
    'end':str(frame['available_at'].max().date()),
    'distressed':int(frame['distress_next_year'].sum())}
    for name,frame in [('train',train),('validation',valid),('test',test)]])
companies.to_csv(OUT/'companies_six_features.csv',index=False,encoding='utf-8-sig')
emit('Data split',split_summary)
# BLOCK dataset END

# BLOCK impurity START
def impurity(counts,criterion='gini'):
    p=np.asarray(counts,dtype=float)/np.sum(counts)
    if criterion=='gini':
        return float(1-np.sum(p*p))
    p=p[p>0]
    return float(-np.sum(p*np.log2(p)))
split_example=[]
for criterion in ['gini','entropy']:
    parent=impurity([80,20],criterion)
    left=impurity([60,0],criterion)
    right=impurity([20,20],criterion)
    after=0.6*left+0.4*right
    split_example.append({'criterion':criterion,'parent':parent,
        'left':left,'right':right,'weighted_children':after,'gain':parent-after})
split_example=pd.DataFrame(split_example)
emit('Impurity example',split_example)
# BLOCK impurity END

# BLOCK trees START
def metrics(model,X,y,model_name,split):
    positive=int(np.flatnonzero(model.classes_==1)[0])
    p=model.predict_proba(X)[:,positive]
    predicted=model.predict(X)
    return {'model':model_name,'split':split,'accuracy':accuracy_score(y,predicted),
        'precision':precision_score(y,predicted,zero_division=0),
        'recall':recall_score(y,predicted,zero_division=0),
        'AP':average_precision_score(y,p),'ROC_AUC':roc_auc_score(y,p),
        'Brier':brier_score_loss(y,p),'log_loss':log_loss(y,p,labels=[0,1])}
unrestricted=DecisionTreeClassifier(random_state=404)
regulated=DecisionTreeClassifier(max_depth=3,min_samples_split=40,
    min_samples_leaf=20,random_state=404)
tree_models={'unrestricted_tree':unrestricted,'regulated_tree':regulated}
all_metrics=[]
tree_structure=[]
for name,model in tree_models.items():
    model.fit(X_train,y_train)
    tree_structure.append({'model':name,'depth':model.get_depth(),
        'leaves':model.get_n_leaves(),'nodes':model.tree_.node_count})
    for split,X,y in [('train',X_train,y_train),('validation',X_valid,y_valid),('test',X_test,y_test)]:
        all_metrics.append(metrics(model,X,y,name,split))
tree_structure=pd.DataFrame(tree_structure)
assert regulated.get_depth()<=3
assert min(regulated.tree_.n_node_samples[regulated.tree_.children_left==-1])>=20
emit('Tree structure',tree_structure)
emit('Tree metrics',pd.DataFrame(all_metrics))
# BLOCK trees END

# BLOCK rules START
default_rules=export_text(regulated,feature_names=FEATURES,
    max_depth=regulated.get_depth(),decimals=6,show_weights=True)
(OUT/'tree_default_class_rules.txt').write_text(default_rules,encoding='utf-8')
tree=regulated.tree_
class_index=int(np.flatnonzero(regulated.classes_==1)[0])
def probability_at_leaf(node):
    values=tree.value[node][0]
    return float(values[class_index]/values.sum())
def policy_rule_lines(node=0,depth=0):
    prefix='    '*depth
    if tree.children_left[node]==-1:
        p=probability_at_leaf(node)
        return [prefix+f'return {int(p>POLICY_THRESHOLD)}  # node={node}, '
            f'PD={p:.10f}, samples={tree.n_node_samples[node]}, '
            f'policy: PD > {POLICY_THRESHOLD:.2f}']
    feature=FEATURES[tree.feature[node]]
    threshold=repr(float(tree.threshold[node]))
    return ([prefix+f'if x[{feature!r}] <= {threshold}:']
        +policy_rule_lines(tree.children_left[node],depth+1)
        +[prefix+'else:']+policy_rule_lines(tree.children_right[node],depth+1))
policy_rules='\n'.join(policy_rule_lines())
(OUT/'credit_policy_if_else.txt').write_text(policy_rules,encoding='utf-8')
def traverse_policy(row):
    # sklearn trees cast input features to float32 before comparison.
    values=np.asarray(row,dtype=np.float32)
    node=0
    while tree.children_left[node]!=-1:
        left=values[tree.feature[node]]<=tree.threshold[node]
        node=tree.children_left[node] if left else tree.children_right[node]
    return int(probability_at_leaf(node)>POLICY_THRESHOLD)
rule_predictions=np.array([traverse_policy(row) for row in X_test.to_numpy()])
model_policy=(regulated.predict_proba(X_test)[:,class_index]>POLICY_THRESHOLD).astype(int)
assert np.array_equal(rule_predictions,model_policy)
emit('export_text: default class rules',default_rules)
emit('Business policy if-else rules',policy_rules)
# BLOCK rules END

# BLOCK mdi START
def manual_mdi(model):
    structure=model.tree_
    weighted=structure.weighted_n_node_samples
    importance=np.zeros(model.n_features_in_)
    for node in range(structure.node_count):
        left,right=structure.children_left[node],structure.children_right[node]
        if left==-1:
            continue
        decrease=(weighted[node]*structure.impurity[node]
            -weighted[left]*structure.impurity[left]
            -weighted[right]*structure.impurity[right])/weighted[0]
        importance[structure.feature[node]]+=decrease
    return importance/importance.sum() if importance.sum()>0 else importance
assert np.allclose(manual_mdi(regulated),regulated.feature_importances_)
mdi_tree=pd.DataFrame({'feature':FEATURES,'MDI':regulated.feature_importances_})
mdi_tree=mdi_tree.sort_values('MDI',ascending=False).reset_index(drop=True)
emit('Six-feature tree MDI',mdi_tree)
# BLOCK mdi END

# BLOCK bagging START
def majority_correct(M,p):
    return sum(comb(M,k)*p**k*(1-p)**(M-k) for k in range(M//2+1,M+1))
condorcet=pd.DataFrame([{'M':M,'p':p,'majority_correct':majority_correct(M,p)}
    for p in [0.45,0.55,0.65] for M in [1,5,21,101]])
rng=np.random.default_rng(405)
oob_rows=[]
for n in [20,200,1200]:
    fractions=[]
    for repeat in range(300):
        indices=rng.integers(0,n,size=n)
        fractions.append(1-len(np.unique(indices))/n)
    oob_rows.append({'n':n,'theoretical':(1-1/n)**n,
        'simulation_mean':float(np.mean(fractions)),'limit':float(np.exp(-1))})
oob_table=pd.DataFrame(oob_rows)
forest=RandomForestClassifier(n_estimators=200,max_features='sqrt',
    min_samples_leaf=10,bootstrap=True,oob_score=True,n_jobs=1,random_state=404)
all_features_forest=RandomForestClassifier(n_estimators=200,max_features=None,
    min_samples_leaf=10,bootstrap=True,oob_score=True,n_jobs=1,random_state=404)
forest_diagnostics=[]
for name,model in [('random_forest_sqrt',forest),('bagged_all_features',all_features_forest)]:
    model.fit(X_train,y_train)
    for split,X,y in [('train',X_train,y_train),('validation',X_valid,y_valid),('test',X_test,y_test)]:
        all_metrics.append(metrics(model,X,y,name,split))
    predictions=np.array([est.predict_proba(X_valid.to_numpy())[:,1] for est in model.estimators_])
    usable=predictions.std(axis=1)>0
    correlation=np.corrcoef(predictions[usable])
    avg_correlation=correlation[np.triu_indices_from(correlation,k=1)].mean()
    forest_diagnostics.append({'model':name,'OOB_accuracy_diagnostic':model.oob_score_,
        'mean_validation_prediction_correlation':float(avg_correlation),
        'features_considered':model.estimators_[0].max_features_,
        'mean_tree_depth':float(np.mean([t.get_depth() for t in model.estimators_]))})
forest_diagnostics=pd.DataFrame(forest_diagnostics)
mdi_forest=pd.DataFrame({'feature':FEATURES,'MDI':forest.feature_importances_})
mdi_forest=mdi_forest.sort_values('MDI',ascending=False).reset_index(drop=True)
emit('Condorcet',condorcet)
emit('OOB simulation',oob_table)
emit('Forest diagnostics',forest_diagnostics)
emit('Six-feature forest MDI',mdi_forest)
# BLOCK bagging END

# BLOCK adaboost START
def classic_adaboost_steps(X,y,n_rounds=8):
    signed_y=2*y-1
    weights=np.full(len(y),1/len(y))
    margin=np.zeros(len(y))
    rows=[]
    for iteration in range(n_rounds):
        stump=DecisionTreeClassifier(max_depth=1,random_state=600+iteration)
        stump.fit(X,signed_y,sample_weight=weights)
        predicted=stump.predict(X)
        missed=predicted!=signed_y
        error=float(np.dot(weights,missed))
        if error>=0.5:
            break
        clipped=np.clip(error,1e-12,1-1e-12)
        alpha=0.5*np.log((1-clipped)/clipped)
        unnormalized=weights*np.exp(-alpha*signed_y*predicted)
        normalizer=unnormalized.sum()
        weights=unnormalized/normalizer
        margin+=alpha*predicted
        rows.append({'round':iteration+1,'weighted_error':error,'alpha_classic':alpha,
            'Z':float(normalizer),'exp_loss':float(np.exp(-signed_y*margin).mean()),
            'effective_weight_samples':float(1/np.sum(weights**2))})
        assert np.isclose(weights.sum(),1)
        if error==0:
            break
    return pd.DataFrame(rows)
ada_steps=classic_adaboost_steps(X_train,y_train)
ada=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1,min_samples_leaf=20),
    n_estimators=120,learning_rate=0.3,random_state=404)
ada.fit(X_train,y_train)
for split,X,y in [('train',X_train,y_train),('validation',X_valid,y_valid),('test',X_test,y_test)]:
    all_metrics.append(metrics(ada,X,y,'AdaBoost',split))
emit('Classic binary AdaBoost derivation check',ada_steps)
# BLOCK adaboost END

# BLOCK gbm START
initial_p=float(y_train.mean())
initial_F=np.log(initial_p/(1-initial_p))
residual=y_train-initial_p
gradient_tree=DecisionTreeRegressor(max_depth=2,min_samples_leaf=20,random_state=404)
gradient_tree.fit(X_train,residual)
leaf_index=gradient_tree.apply(X_train)
increment=np.zeros(len(y_train))
leaf_updates=[]
for leaf in np.unique(leaf_index):
    selected=leaf_index==leaf
    numerator=float(residual[selected].sum())
    denominator=float(selected.sum()*initial_p*(1-initial_p))
    gamma=numerator/denominator
    increment[selected]=gamma
    leaf_updates.append({'leaf':int(leaf),'samples':int(selected.sum()),
        'sum_negative_gradient':numerator,'sum_hessian':denominator,'newton_step':gamma})
learning_rate=0.05
one_step_F=initial_F+learning_rate*increment
one_step_p=1/(1+np.exp(-one_step_F))
gbm_step={'initial_log_loss':float(log_loss(y_train,np.full(len(y_train),initial_p))),
    'after_one_step_log_loss':float(log_loss(y_train,one_step_p)),
    'initial_log_odds':float(initial_F)}
gbm=GradientBoostingClassifier(loss='log_loss',n_estimators=120,
    learning_rate=0.05,max_depth=2,min_samples_leaf=20,
    subsample=1.0,n_iter_no_change=None,random_state=404)
gbm.fit(X_train,y_train)
for split,X,y in [('train',X_train,y_train),('validation',X_valid,y_valid),('test',X_test,y_test)]:
    all_metrics.append(metrics(gbm,X,y,'GBM',split))
stage_rows=[]
for stage,(p_train,p_valid) in enumerate(zip(gbm.staged_predict_proba(X_train),
        gbm.staged_predict_proba(X_valid)),start=1):
    stage_rows.append({'stage':stage,'train_log_loss':log_loss(y_train,p_train),
        'validation_log_loss':log_loss(y_valid,p_valid)})
stages=pd.DataFrame(stage_rows)
leaf_updates=pd.DataFrame(leaf_updates)
emit('GBM first-step Newton leaf updates',leaf_updates)
emit('GBM first-step loss',gbm_step)
metric_table=pd.DataFrame(all_metrics)
emit('All models: same split',metric_table)
# BLOCK gbm END

# BLOCK shap START
# A predeclared synthetic applicant, not a real borrower or selected test failure.
borrower_raw=pd.DataFrame([[350.,75.,-3.,-0.5,-2.,-4.]],columns=FEATURES)
borrower=pd.DataFrame(imputer.transform(borrower_raw),columns=FEATURES)
background=X_train.sample(n=100,random_state=404).copy()
explainer=shap.TreeExplainer(forest,data=background,model_output='probability',
    feature_perturbation='interventional')
explanation=explainer(borrower)
positive_index=int(np.flatnonzero(forest.classes_==1)[0])
values=np.asarray(explanation.values)
if values.ndim==3:
    phi=values[0,:,positive_index]
    base=float(np.asarray(explanation.base_values)[0,positive_index])
else:
    raise ValueError(f'Unexpected SHAP shape for the pinned random-forest version: {values.shape}')
borrower_pd=float(forest.predict_proba(borrower)[0,positive_index])
reconstructed=float(base+phi.sum())
assert np.isclose(reconstructed,borrower_pd,atol=1e-6)
contributions=pd.DataFrame({'feature':FEATURES,'input':borrower.iloc[0].to_numpy(),
    'SHAP_probability':phi,'SHAP_percentage_points':100*phi})
contributions['absolute_SHAP']=np.abs(phi)
contributions=contributions.sort_values('absolute_SHAP',ascending=False).reset_index(drop=True)
decision='DECLINE' if borrower_pd>POLICY_THRESHOLD else 'PASS_MODEL_SCREEN'
borrower_result={'borrower':'synthetic applicant 1','background_rows':len(background),
    'base_probability':base,'predicted_probability':borrower_pd,
    'sum_SHAP':float(phi.sum()),'reconstructed_probability':reconstructed,
    'policy_threshold':POLICY_THRESHOLD,'decision':decision}
emit('Applicant 1 probability decomposition',borrower_result)
emit('Applicant 1 feature contributions',contributions)

# Independent exact Shapley audit: six features imply only 64 coalitions.
d=len(FEATURES)
coalition_values={}
for mask in range(1<<d):
    hybrid=background.copy()
    for j in range(d):
        if mask & (1<<j):
            hybrid.iloc[:,j]=float(borrower.iloc[0,j])
    coalition_values[mask]=float(forest.predict_proba(hybrid)[:,positive_index].mean())
exact_phi=np.zeros(d)
for j in range(d):
    for mask in range(1<<d):
        if mask & (1<<j):
            continue
        size=mask.bit_count()
        weight=factorial(size)*factorial(d-size-1)/factorial(d)
        exact_phi[j]+=weight*(coalition_values[mask|(1<<j)]-coalition_values[mask])
assert np.allclose(exact_phi,phi,atol=1e-6)
assert np.isclose(coalition_values[0],base,atol=1e-6)
borrower_result['max_exact_SHAP_difference']=float(np.max(np.abs(exact_phi-phi)))
contributions.to_csv(OUT/'applicant1_shap.csv',index=False,encoding='utf-8-sig')
(OUT/'applicant1_explanation.json').write_text(json.dumps(borrower_result,indent=2),encoding='utf-8')
# BLOCK shap END

# BLOCK figures START
fig,ax=plt.subplots(figsize=(17,8))
plot_tree(regulated,feature_names=FEATURES,class_names=['normal','distress'],
    filled=True,rounded=True,fontsize=9,ax=ax)
ax.set_title('깊이 3, 최소 잎 표본 20: 학습된 규제 트리')
save(fig,'fig_04_01.png')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for name,color in [('unrestricted_tree','#d8794e'),('regulated_tree','#235bb5')]:
    subset=metric_table.loc[metric_table['model']==name]
    axes[0].plot(subset['split'],subset['AP'],'o-',label=name,color=color)
axes[0].set(ylabel='Average Precision',title='학습 성능과 미래 구간 성능');axes[0].legend(fontsize=8)
axes[1].barh(mdi_forest['feature'][::-1],mdi_forest['MDI'][::-1],color='#208980')
axes[1].set(xlabel='정규화 MDI',title='랜덤 포레스트: 전역 불순도 중요도')
save(fig,'fig_04_02.png')
fig,axes=plt.subplots(1,2,figsize=(12,4))
M=np.arange(1,201)
for rho in [0,0.1,0.5]:
    axes[0].plot(M,rho+(1-rho)/M,label=f'rho={rho}')
axes[0].set(xlabel='모형 수 M',ylabel='평균 분산 / 개별 분산',title='상관관계가 남기는 분산 하한');axes[0].legend()
axes[1].plot(stages['stage'],stages['train_log_loss'],label='학습',color='#235bb5')
axes[1].plot(stages['stage'],stages['validation_log_loss'],label='시간 검증',color='#d8794e')
axes[1].set(xlabel='GBM 단계',ylabel='Log loss',title='손실의 음의 그래디언트를 순차 학습');axes[1].legend()
save(fig,'fig_04_03.png')
fig,ax=plt.subplots(figsize=(10,5))
display=contributions.iloc[::-1]
ax.barh(display['feature'],display['SHAP_percentage_points'],
    color=['#d8794e' if value>0 else '#235bb5' for value in display['SHAP_percentage_points']])
ax.axvline(0,color='#999',linewidth=0.8)
ax.set(xlabel='부실 확률 기여도 (%포인트)',title='가상 1번 차주: 기준 확률 대비 여섯 피처의 기여')
save(fig,'fig_04_04.png')
# BLOCK figures END

# BLOCK outputs START
report={'split':split_summary.to_dict('records'),'impurity':split_example.to_dict('records'),
    'tree_structure':tree_structure.to_dict('records'),'metrics':metric_table.to_dict('records'),
    'default_rules':default_rules,'policy_rules':policy_rules,
    'mdi_tree':mdi_tree.to_dict('records'),'mdi_forest':mdi_forest.to_dict('records'),
    'condorcet':condorcet.to_dict('records'),'oob':oob_table.to_dict('records'),
    'forest_diagnostics':forest_diagnostics.to_dict('records'),
    'ada_steps':ada_steps.to_dict('records'),'gbm_step':gbm_step,
    'gbm_leaves':leaf_updates.to_dict('records'),'borrower':borrower_result,
    'shap':contributions.to_dict('records'),
    'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','matplotlib','scikit-learn','shap']},
    'python':platform.python_version(),'checks':'passed'}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'requirements.txt').write_text('\n'.join(k+'=='+v for k,v in report['versions'].items())+'\n',encoding='utf-8')
metric_table.to_csv(OUT/'model_metrics.csv',index=False,encoding='utf-8-sig')
stages.to_csv(OUT/'gbm_stages.csv',index=False,encoding='utf-8-sig')
emit('Checks','passed')
# BLOCK outputs END

