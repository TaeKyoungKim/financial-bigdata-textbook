"""Chapter 01: synthetic, reproducible teaching examples. No market data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

OUT = Path(__file__).resolve().parent
ASSETS = OUT / 'assets'
ASSETS.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                     'figure.facecolor': '#f7f9fc', 'axes.facecolor': '#f7f9fc',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 11, 'savefig.dpi': 170})

def save(fig, name):
    fig.tight_layout(pad=2)
    fig.savefig(ASSETS / name)
    plt.close(fig)

# EXAMPLE 1 START
def regime_example(n=1000, seed=42, cost=0.0003):
    """Known signal flips sign halfway; this is an assumed scenario."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    beta = np.where(np.arange(n) < n // 2, 0.003, -0.003)
    # x[t-1] predicts return[t]; x[t] only becomes available after return[t].
    lag_x = np.r_[0.0, x[:-1]]
    asset_return = beta * lag_x + rng.normal(0, 0.012, n)
    signal = (x > 0).astype(float)
    held = np.r_[0.0, signal[:-1]]
    turnover = np.abs(np.diff(np.r_[0.0, held]))
    net = held * asset_return - cost * turnover
    equity = np.r_[1.0, np.cumprod(1 + net)]
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    return pd.DataFrame({'return': asset_return, 'signal': signal,
                         'held': held, 'net': net}), equity, drawdown

market, equity, drawdown = regime_example()
regime_result = {}
for name, sample in [('전반', market.iloc[:500]), ('후반', market.iloc[500:])]:
    wealth = np.r_[1, np.cumprod(1 + sample['net'])]
    regime_result[name] = {'누적수익률': float(wealth[-1] - 1),
        '최대낙폭': float(-(wealth / np.maximum.accumulate(wealth) - 1).min())}
# EXAMPLE 1 END

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
d = np.arange(1, 16)
ax[0].semilogy(d, 3.0 ** d, 'o-', color='#235bb5')
ax[0].set(xlabel='변수 개수 D', ylabel='가능한 조합 수 (로그 눈금)', title='변수마다 3개 구간을 나누는 경우')
ax[1].plot(equity, color='#235bb5', label='비용 차감 후 자산')
ax[1].axvline(500, color='#dc7350', linestyle='--', label='가정한 관계 변화')
ax[1].set(xlabel='관측일', ylabel='초기자산 = 1', title='같은 신호, 달라진 수익률 관계')
ax[1].legend(fontsize=9)
save(fig, 'fig_01_01.png')

# EXAMPLE 2 START
def edge_example(p=0.508, b=0.0015):
    """Expected P&L per transaction, expressed as a fraction of notional."""
    costs = np.array([0, 0.00001, 0.000024, 0.00003])
    expected = (2 * p - 1) * b - costs
    n = np.array([100, 1000, 10000, 100000])
    # Equicorrelation sensitivity, not a measured market correlation.
    effective = {str(rho): (n / (1 + (n - 1) * rho)).tolist()
                 for rho in [0, 0.001, 0.01]}
    return pd.DataFrame({'비용_bps': costs * 10000,
                         '거래당_기대손익_bps': expected * 10000}), n, effective

edge_table, breadth, effective = edge_example()
# EXAMPLE 2 END
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(edge_table['비용_bps'].map(lambda v: f'{v:.2f}'), edge_table['거래당_기대손익_bps'], color=['#235bb5','#235bb5','#8194a8','#dc7350'])
ax[0].axhline(0, color='#777', linewidth=0.8)
ax[0].set(xlabel='거래당 비용 (bps)', ylabel='거래당 기대손익 (bps)', title='작은 우위는 비용에 민감하다')
for rho, values in effective.items():
    ax[1].loglog(breadth, values, 'o-', label=f'상관계수 {rho}')
ax[1].set(xlabel='명목 거래 수', ylabel='유효 독립 거래 수', title='거래 수와 독립 기회 수의 차이')
ax[1].legend(fontsize=9)
save(fig, 'fig_01_02.png')

# EXAMPLE 3 START
def stationarity_example(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    # Zero expected log increment. Differenced series is iid by construction.
    log_return = rng.normal(0, 0.012, n)
    log_price = np.log(100) + np.cumsum(log_return)
    results = []
    for name, series in [('로그 가격', log_price), ('로그 차분', np.diff(log_price))]:
        result = adfuller(series, regression='c', autolag='AIC')
        results.append({'시계열': name, 'ADF': result[0], 'p값': result[1],
                        '시차': result[2], '5%임계값': result[4]['5%']})
    # True signal is known only because we generate it explicitly.
    true_signal = rng.normal(0, 0.001, n)
    noise = rng.normal(0, 0.01, n)
    theoretical_snr = 0.001**2 / 0.01**2
    return log_price, log_return, pd.DataFrame(results), theoretical_snr

log_price, log_return, adf_table, snr = stationarity_example()
# EXAMPLE 3 END
fig, ax = plt.subplots(2, 1, figsize=(10, 5))
ax[0].plot(log_price, color='#235bb5')
ax[0].set(ylabel='로그 가격', title='합성 랜덤워크와 차분 시계열')
ax[1].plot(log_return, color='#208980', linewidth=0.7)
ax[1].set(xlabel='관측일', ylabel='로그 수익률')
save(fig, 'fig_01_03.png')

# EXAMPLE 4 START
def financial_pipeline(n=500, window=20, horizon=5, seed=42):
    if n <= window + horizon or window < 2 or horizon < 1:
        raise ValueError('n, window, horizon의 범위를 확인하세요.')
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2023-01-02', periods=n)
    increments = rng.normal((0.10 - 0.5 * 0.25**2) / 252, 0.25 / np.sqrt(252), n)
    base = 50000 * np.exp(np.cumsum(increments))
    raw = base.copy()
    split_day = n // 2
    raw[split_day:] *= 0.5
    df = pd.DataFrame({'raw_close': raw, 'split_factor': 1.0}, index=dates)
    df.iloc[split_day, df.columns.get_loc('split_factor')] = 0.5
    # Event-day close is already post-split; apply only strictly future factors.
    factors = df['split_factor'].shift(-1, fill_value=1)
    df['adjustment'] = factors.iloc[::-1].cumprod().iloc[::-1]
    df['adj_close'] = df['raw_close'] * df['adjustment']
    df['log_return'] = np.log(df['adj_close']).diff()
    df['vol_20d'] = df['log_return'].rolling(window).std(ddof=1) * np.sqrt(252)
    df['momentum_20d'] = df['log_return'].rolling(window).sum()
    df['target_fwd5d'] = np.log(df['adj_close'].shift(-horizon) / df['adj_close'])
    # Preserve unavailable labels instead of turning them into class 0.
    df['target_class'] = (df['target_fwd5d'] > 0).astype('Int64')
    df.loc[df['target_fwd5d'].isna(), 'target_class'] = pd.NA
    features = ['vol_20d', 'momentum_20d']
    training = df.dropna(subset=features + ['target_fwd5d']).copy()
    inference = df.dropna(subset=features).copy()
    assert np.allclose(df['adj_close'], base * 0.5)
    assert np.isclose(df['log_return'].iloc[split_day], increments[split_day])
    assert len(training) == n - window - horizon
    assert df['target_class'].tail(horizon).isna().all()
    return df, training, inference, features

raw, dataset, inference, features = financial_pipeline()
X = dataset[features]
y = dataset['target_fwd5d']
# EXAMPLE 4 END
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(raw['raw_close'].values, label='원시 종가', color='#dc7350')
ax[0].plot(raw['adj_close'].values, label='분할 보정 종가', color='#235bb5')
ax[0].axvline(250, linestyle='--', color='#999')
ax[0].set(xlabel='관측일', ylabel='가격', title='1주가 2주가 되는 액면분할')
ax[0].legend(fontsize=9)
ax[1].plot(np.log(raw['raw_close']).diff().values, label='원시 로그수익률', color='#dc7350')
ax[1].plot(raw['log_return'].values, label='보정 로그수익률', color='#235bb5')
ax[1].set(xlim=(240,260), xlabel='관측일', ylabel='로그 수익률', title='분할일의 인위적 급락 제거')
ax[1].legend(fontsize=9)
save(fig, 'fig_01_04.png')

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.set(xlim=(0, 100), ylim=(-1, 3))
for y0, start, width, label, color in [(2,5,40,'피처: 과거 20일','#235bb5'), (1,45,30,'타깃: 미래 5일','#208980'), (0,5,35,'학습 구간','#235bb5'), (0,40,12,'간격','#d7a64a'), (0,52,40,'검증 구간','#208980')]:
    ax.broken_barh([(start,width)], (y0,0.55), facecolors=color)
    ax.text(start+width/2,y0+0.275,label,ha='center',va='center',color='white',fontsize=10)
ax.axvline(45, color='#dc7350',linestyle='--')
ax.text(45,2.8,'예측 시점 t',ha='center',color='#b65132')
ax.set_title('관측 시점과 라벨 구간을 함께 관리한다 (개념도·축척 없음)')
ax.axis('off')
save(fig, 'fig_01_05.png')

fig, ax = plt.subplots(figsize=(11, 4))
labels = [('원천 데이터\n가격·수급·공시','pandas',0),('배열·통계\n수익률·검정','NumPy / statsmodels',1),('학습·검증\n전처리·모델','scikit-learn',2),('해석·운영\n차트·모니터링','Matplotlib',3)]
for title, package, i in labels:
    x = i * 2.7
    ax.text(x,1.2,title,ha='center',va='center',fontsize=12,bbox=dict(boxstyle='round,pad=0.7',fc='#e5edf9',ec='#235bb5'))
    ax.text(x,0.15,package,ha='center',fontsize=10,color='#235bb5')
    if i < 3: ax.annotate('',xy=(x+1.7,1.2),xytext=(x+1,1.2),arrowprops=dict(arrowstyle='->',color='#8194a8',lw=2))
ax.set(xlim=(-1.3,9.5),ylim=(-0.5,2.4),title='금융 머신러닝의 파이썬 생태계')
ax.axis('off')
save(fig, 'fig_01_06.png')

dataset.to_csv(OUT / 'chapter01_dataset.csv', encoding='utf-8-sig')
report = {'regime': regime_result, 'edge': edge_table.to_dict(orient='records'),
          'adf': adf_table.to_dict(orient='records'), 'snr': snr,
          'training_shape': list(dataset.shape), 'X_shape': list(X.shape),
          'inference_rows': len(inference), 'checks': 'passed'}
(OUT / 'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True,indent=2))
