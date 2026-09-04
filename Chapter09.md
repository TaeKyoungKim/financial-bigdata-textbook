# Chapter 09. 금융 시계열 딥러닝: 순환신경망(RNN, LSTM, GRU)과 자산 가격 예측

> 순차 금융 문제 → 쉬운 직관 → 수식 유도 → 완전한 PyTorch 코드 → Walk-Forward 결과 → 적용 한계

## 이 장을 시작하며

앞 장의 MLP는 하나의 관측행을 입력받아 비선형 교차 효과를 학습했다. 그러나 오늘의 시장은 오늘 숫자만으로 만들어지지 않는다. 같은 1% 상승이라도 지난 60일 동안 천천히 회복한 뒤 나타났는지, 급락 다음 날 반등으로 나타났는지에 따라 의미가 다를 수 있다. 시계열 신경망은 과거 관측을 순서대로 읽으며 내부 상태를 갱신한다.

이번 장은 단순 종가 외삽이 아니라 **과거 60영업일의 OHLCV와 후행 변동성으로 5영업일 후 수익률을 예측**한다. 예측 수익률을 현재 종가에 적용해 미래 가격의 조건부 점추정치로 변환한다. 원가격 수준을 직접 맞혀 높은 R²를 얻는 착시를 피하고, MAE·RMSE·방향 적중률로 미래 Walk-Forward 구간을 평가한다.

모든 가격과 거래량은 교육용 합성 자료다. 실제 종목의 가격, 2008년 금융위기 또는 2020년 팬데믹 자료를 학습한 결과가 아니다. 두 역사적 사건은 LSTM의 금융 메모리를 설명하는 개념 사례로만 사용한다. 코드는 PyTorch CPU 환경에서 처음부터 실행할 수 있도록 데이터 생성, Dataset, DataLoader, RNN·LSTM·GRU, 학습, 검증, 모델 저장과 추론을 포함한다.

첨부 원본 교재에는 이 딥러닝 장에 대응하는 슬라이드가 없으므로 슬라이드 번호를 만들어내지 않는다. 이번 요청의 항목은 다음과 같이 배치한다.

| 요청 항목 | 본문 |
|---|---|
| 순차성과 Vanilla RNN | 9.1 |
| BPTT·장기 의존성·기울기 문제 | 9.2 |
| LSTM 5단계 게이트 | 9.3 |
| 금융위기·팬데믹·모멘텀 메모리 해석 | 9.4 |
| GRU와 파라미터 효율 | 9.5 |
| 60일×6피처 PyTorch 텐서 | 9.6–9.7 |
| Walk-Forward 평가 | 9.8–9.10 |

## 9.1 시계열 순차성과 Vanilla RNN

### 9.1.1 표 형태와 순서가 있는 입력의 차이

MLP에 60일×6개 값을 일렬로 펴서 360개 피처로 넣을 수도 있다. 그러면 ‘5일 전 종가’와 ‘30일 전 종가’를 서로 다른 고정 열로 취급한다. RNN은 모든 시점에 같은 전이 가중치를 반복 사용한다. “현재 입력과 이전 기억을 결합한다”라는 규칙을 시점마다 공유하는 구조다.

날짜 t의 입력을 xₜ∈Rᵈ, 은닉상태를 hₜ∈Rᵐ이라고 하자. Vanilla RNN은 다음과 같다.

$$\boldsymbol{a}_t=W_{hh}\boldsymbol{h}_{t-1}+W_{xh}\boldsymbol{x}_t+\boldsymbol{b}_h.$$

$$\boldsymbol{h}_t=\tanh(\boldsymbol{a}_t).$$

사용자가 제시한 식과 같은 형태다.

$$\boldsymbol{h}_t=\tanh(W_{hh}\boldsymbol{h}_{t-1}+W_{xh}\boldsymbol{x}_t+\boldsymbol{b}_h).$$

d가 입력 피처 수, m이 은닉 크기라면 Wₓₕ는 m×d, Wₕₕ는 m×m, b는 m차원이다. 60일 입력에서는 이 계산을 60회 수행한다. 회귀 출력은 마지막 상태를 선형층에 전달한다.

$$\hat{y}=\boldsymbol{w}_{hy}^{T}\boldsymbol{h}_{60}+b_y.$$

이번 목표 y는 5영업일 후 단순 수익률을 %포인트로 표현한 값이다. 은닉상태가 금융적으로 관측되는 ‘시장 상태’인 것은 아니다. 예측 손실을 줄이도록 학습되는 내부 표현이다.

### 9.1.2 은닉상태는 무엇을 기억하는가

hₜ는 과거 모든 입력의 요약이 될 수 있다. 전개하면 hₜ₋₁은 xₜ₋₁과 hₜ₋₂의 함수이고, hₜ₋₂는 더 이전 입력의 함수다.

$$\boldsymbol{h}_t=F(\boldsymbol{x}_t,F(\boldsymbol{x}_{t-1},\ldots,F(\boldsymbol{x}_1,\boldsymbol{h}_0))).$$

그러나 유한한 m차원 벡터가 모든 과거를 손실 없이 저장한다는 뜻은 아니다. 어떤 정보를 유지하고 삭제할지 학습한다. 시퀀스 시작의 h₀는 보통 0으로 둔다.

코드의 각 60일 창은 독립 표본이며 상태를 0에서 시작한다. 연속된 배치 사이에 hidden state를 전달하지 않는다. 따라서 앞 창의 2008년 정보를 다음 창까지 무한히 보존하는 stateful 모형이 아니다. 모델 **가중치**는 여러 과거 학습 창에서 공통 패턴을 배울 수 있지만, **은닉상태**는 현재 60일 창 안의 정보를 운반한다. 이 두 기억을 구분해야 한다.

### 9.1.3 파라미터 공유의 장점과 제약

동일한 W를 매 시점 사용하므로 시퀀스 길이가 60으로 늘어도 파라미터 수가 60배가 되지 않는다. m=16, d=6인 단층 RNN은 입력가중치 16×6, 순환가중치 16×16, 두 편향 벡터 16개씩을 갖고 출력층 16×1과 편향을 더한다.

$$P_{RNN}=md+m^2+2m+(m+1).$$

PyTorch의 RNN은 입력과 은닉 쪽 편향을 각각 보관하므로 편향이 2m개다. 수학식에서는 두 편향을 합쳐 하나의 b로 쓸 수 있다. 파라미터 공유는 자료 효율성을 높일 수 있지만 관계가 시간에 따라 변하는 금융시장에서는 고정된 전이 규칙이라는 가정이 제약이 될 수도 있다.

## 9.2 BPTT와 장기 의존성

### 9.2.1 시간을 펼치면 깊은 신경망이 된다

RNN을 60일에 걸쳐 그리면 같은 셀을 60층 이어 놓은 계산 그래프처럼 보인다. 마지막 손실이 과거 상태에 미친 영향을 연쇄 법칙으로 역산하는 방법이 BPTT(Backpropagation Through Time)다.

시점 t의 상태가 이전 상태에 대한 Jacobian은 다음과 같다. Dₜ는 Tanh 도함수를 대각에 둔 행렬이다.

$$J_t=\frac{\partial\boldsymbol{h}_t}{\partial\boldsymbol{h}_{t-1}}=D_tW_{hh},\qquad D_t=\operatorname{diag}(1-\tanh^2(\boldsymbol{a}_t)).$$

시점 k의 상태가 시점 t>k의 상태에 미치는 미분은 Jacobian들의 곱이다. 행렬곱 순서는 최근 시점부터 적용된다.

$$\frac{\partial\boldsymbol{h}_t}{\partial\boldsymbol{h}_k}=J_tJ_{t-1}\cdots J_{k+1}=\prod_{j=k+1}^{t}D_jW_{hh}.$$

마지막 상태만으로 손실을 계산한다면 과거 상태의 손실 기울기는 다음과 같다.

$$\frac{\partial L}{\partial\boldsymbol{h}_k}=\frac{\partial L}{\partial\boldsymbol{h}_t}\frac{\partial\boldsymbol{h}_t}{\partial\boldsymbol{h}_k}.$$

행 벡터·열 벡터 표기에 따라 Jacobian의 전치 위치는 바뀔 수 있지만, 여러 Jacobian이 곱해진다는 핵심은 같다.

### 9.2.2 기울기 소실의 수학적 상한

행렬 노름의 submultiplicative 성질을 사용하면 곱의 크기를 위에서 제한할 수 있다.

$$\left\|\frac{\partial\boldsymbol{h}_t}{\partial\boldsymbol{h}_k}\right\|\leq\prod_{j=k+1}^{t}\|D_j\|\,\|W_{hh}\|.$$

Tanh 도함수의 절댓값은 1 이하이고, 포화 영역에서는 1보다 훨씬 작다. 모든 단계에서 곱의 상한이 ρ<1이면 다음처럼 지연 τ=t−k에 대해 지수적으로 감소한다.

$$\|D_j\|\,\|W_{hh}\|\leq\rho<1\quad\Longrightarrow\quad \left\|\frac{\partial\boldsymbol{h}_t}{\partial\boldsymbol{h}_{t-\tau}}\right\|\leq\rho^{\tau}\longrightarrow0.$$

60일 전의 정보가 손실을 줄이는 데 중요해도 그 시점 가중치까지 전달되는 학습 신호가 작아진다. 이를 장기 의존성 학습의 어려움이라고 한다. 과거 정보가 순전파에서 완전히 사라졌다는 명제와 기울기를 통해 학습하기 어렵다는 명제는 관련되지만 동일하지 않다.

### 9.2.3 기울기 폭발의 조건

위의 노름 부등식은 상한이므로 ρ>1만으로 모든 입력에서 반드시 폭발한다고 증명할 수 없다. 행렬 방향, 활성화 포화, 상쇄가 영향을 준다. 하지만 특정 고유방향이 반복 보존되고 그 방향의 유효 Jacobian 크기가 1보다 크면 지수적으로 커질 수 있다.

가장 명확한 스칼라 예를 보자. 입력이 0이고 h₀=0이면 Tanh'(0)=1이므로 hₜ=0을 유지하고 미분은 정확히 다음과 같다.

$$h_t=\tanh(wh_{t-1}),\qquad \frac{\partial h_T}{\partial h_0}=w^T.$$

|w|<1이면 소실, |w|>1이면 폭발, |w|=1이면 이 특별한 경로에서 유지된다. 일반적인 RNN에서 이와 같은 방향들이 결합한다. [Pascanu·Mikolov·Bengio의 분석](https://proceedings.mlr.press/v28/pascanu13.html)은 순환신경망의 소실·폭발 문제와 clipping을 다룬다.

### 9.2.4 순환가중치의 BPTT 그래디언트

Wₕₕ는 모든 시점에서 공유되므로 각 시점의 기여를 합한다. δₜ=∂L/∂aₜ로 두면 다음과 같다.

$$\frac{\partial L}{\partial W_{hh}}=\sum_{t=1}^{T}\boldsymbol{\delta}_t\boldsymbol{h}_{t-1}^{T}.$$

$$\boldsymbol{\delta}_t=\left(W_{hh}^{T}\boldsymbol{\delta}_{t+1}+\frac{\partial L_t}{\partial\boldsymbol{h}_t}\right)\odot(1-\boldsymbol{h}_t^2).$$

마지막 시점에만 손실이 있으면 중간 시점의 직접 ∂Lₜ/∂hₜ는 0이고 미래 δ가 계속 전달된다. 여러 기간의 예측 손실을 두면 각 시점의 직접 기여가 추가된다.

이번 학습 코드는 전체 60일을 펼쳐 BPTT한다. truncated BPTT로 중간 상태를 detach하지 않는다. 그래디언트 전체 2-노름이 1을 넘으면 방향은 유지한 채 크기를 줄이는 clipping을 사용한다.

$$\boldsymbol{g}_{clip}=\boldsymbol{g}\min\left(1,\frac{c}{\|\boldsymbol{g}\|_2}\right),\qquad c=1.$$

clipping은 폭발을 완화하지만 소실을 해결하지 않고, 최적의 임계값을 자동으로 보장하지 않는다. PyTorch의 [`clip_grad_norm_`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)를 사용하며 비유한 기울기에는 오류를 내도록 설정한다.

### 9.2.5 스칼라 경로를 코드로 확인하기

다음 코드는 w=0.5, 1.0, 1.2와 길이 10·30·60에서 자동미분과 wᵀ을 대조한다. h₀=0이라는 특별한 경로이므로 Tanh가 포화하지 않는다. 금융 예측 성능 실험이 아니라 장기 곱의 수치 확인이다.

```python
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
```

```python
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
```

| weight | steps | dh_final_dh_initial | analytic |
| --- | --- | --- | --- |
| 0.50000000 | 10 | 0.00097656 | 0.00097656 |
| 0.50000000 | 30 | 9.3132e-10 | 9.3132e-10 |
| 0.50000000 | 60 | 8.6736e-19 | 8.6736e-19 |
| 1.00000000 | 10 | 1.00000000 | 1.00000000 |
| 1.00000000 | 30 | 1.00000000 | 1.00000000 |
| 1.00000000 | 60 | 1.00000000 | 1.00000000 |
| 1.20000000 | 10 | 6.19173642 | 6.19173642 |
| 1.20000000 | 30 | 237.37631380 | 237.37631380 |
| 1.20000000 | 60 | 56347.51435317 | 56347.51435317 |


![Vanilla RNN의 반복 미분과 LSTM 직접 셀 경로](assets/fig01_memory_gradients.png)

## 9.3 LSTM 게이트 메커니즘의 다섯 단계

LSTM은 숨은상태 hₜ 외에 셀 상태 cₜ를 둔다. 아래 식에서는 현재 입력과 이전 숨은상태를 연결한 벡터를 qₜ라고 쓴다.

$$\boldsymbol{q}_t=[\boldsymbol{h}_{t-1};\boldsymbol{x}_t].$$

숨은 크기가 m이면 각 게이트와 후보는 m차원이다. 각 W는 m×(m+d)다. PyTorch는 입력·은닉 가중치를 따로 저장하지만 합치면 아래 식과 같다.

### 9.3.1 1단계: Forget Gate

$$\boldsymbol{f}_t=\sigma(W_f\boldsymbol{q}_t+\boldsymbol{b}_f).$$

각 원소는 0과 1 사이다. cₜ₋₁의 해당 메모리를 얼마나 남길지 조절한다. f가 0에 가까우면 지우고 1에 가까우면 유지한다.

### 9.3.2 2단계: Input Gate

$$\boldsymbol{i}_t=\sigma(W_i\boldsymbol{q}_t+\boldsymbol{b}_i).$$

현재 후보 정보를 각 셀 차원에 얼마나 쓸지 결정한다. 이름이 input gate라고 해서 원래 입력 x를 그대로 통과시키는 단순 스위치는 아니다. x와 이전 h를 함께 보고 계산한다.

### 9.3.3 3단계: Candidate Cell State

$$\tilde{\boldsymbol{c}}_t=\tanh(W_c\boldsymbol{q}_t+\boldsymbol{b}_c).$$

기록할 새 내용의 후보이며 −1과 1 사이 값이다. 후보가 만들어졌다고 모두 셀에 저장되는 것은 아니다. input gate와 원소별로 곱해진다.

### 9.3.4 4단계: Cell State Update

$$\boldsymbol{c}_t=\boldsymbol{f}_t\odot\boldsymbol{c}_{t-1}+\boldsymbol{i}_t\odot\tilde{\boldsymbol{c}}_t.$$

첫 항은 과거에서 남긴 부분, 둘째 항은 새로 기록한 부분이다. 두 경로를 더하는 additive update가 Vanilla RNN의 반복 Tanh 합성보다 긴 미분 경로를 제공한다.

### 9.3.5 5단계: Output Gate와 Hidden State

$$\boldsymbol{o}_t=\sigma(W_o\boldsymbol{q}_t+\boldsymbol{b}_o).$$

$$\boldsymbol{h}_t=\boldsymbol{o}_t\odot\tanh(\boldsymbol{c}_t).$$

셀에 저장된 내용 전부를 외부 출력으로 내보내지 않고 output gate가 노출 정도를 정한다. c는 메모리 경로, h는 다음 셀과 예측층에 보이는 상태다.

### 9.3.6 셀 상태의 직접 미분 경로

게이트가 다른 상태 의존 경로에도 영향을 주므로 LSTM 전체 미분이 단순한 f의 곱만인 것은 아니다. 다만 나머지 경로를 고정해 직접 셀 경로만 보면 다음과 같다.

$$\left.\frac{\partial\boldsymbol{c}_t}{\partial\boldsymbol{c}_{t-1}}\right|_{direct}=\operatorname{diag}(\boldsymbol{f}_t).$$

$$\left.\frac{\partial\boldsymbol{c}_T}{\partial\boldsymbol{c}_k}\right|_{direct}=\prod_{t=k+1}^{T}\operatorname{diag}(\boldsymbol{f}_t).$$

f가 1에 가까우면 이 직접 경로의 기울기가 비교적 오래 유지된다. f=0.99가 60회 반복되면 약 0.547, f=0.9는 약 0.0018이다. 게이트도 Sigmoid 포화와 학습의 영향을 받으므로 LSTM이 모든 장기 의존성을 반드시 해결한다는 뜻은 아니다.

$$\tau_{1/2}=\frac{\log(0.5)}{\log(f)}.$$

f가 일정하다는 단순화에서 셀 기억이 절반으로 줄어드는 시간이다. 실제 f는 셀·날짜마다 다르다.

### 9.3.7 PyTorch LSTMCell과 수작업 식 대조

PyTorch는 게이트를 input, forget, candidate, output 순으로 묶어 계산한다. 입력과 은닉에 대한 두 편향을 더한다. 다음 코드는 가중치로 직접 다섯 단계를 계산하고 `LSTMCell`의 결과와 일치하는지 검사한다. [PyTorch LSTM 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html), [Hochreiter와 Schmidhuber의 LSTM 논문](https://doi.org/10.1162/neco.1997.9.8.1735).

```python
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
```

| unit | forget | input | candidate | old_cell | retained | written | new_cell | output | hidden |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.49115866 | 0.35886543 | -0.42140598 | 0.20000000 | 0.09823173 | -0.15122804 | -0.05299631 | 0.54902503 | -0.02906909 |
| 1 | 0.46995088 | 0.34818254 | -0.13262354 | 0.30000000 | 0.14098526 | -0.04617720 | 0.09480806 | 0.62660472 | 0.05922982 |
| 2 | 0.52188281 | 0.42314028 | -0.19008839 | -0.10000000 | -0.05218828 | -0.08043405 | -0.13262233 | 0.65016324 | -0.08572416 |
| 3 | 0.46811545 | 0.60835352 | -0.73125166 | -0.40000000 | -0.18724618 | -0.44485953 | -0.63210571 | 0.43123820 | -0.24127797 |


| forget | retention_60 | half_life_steps |
| --- | --- | --- |
| 0.90000000 | 0.00179701 | 6.57881348 |
| 0.99000000 | 0.54715664 | 68.96756394 |
| 0.99900000 | 0.94173626 | 692.80054918 |


## 9.4 금융 메모리로 읽는 LSTM

### 9.4.1 위기 정보와 최근 모멘텀의 시간 범위

2008년 리먼 파산이나 2020년 팬데믹 충격은 금융시장 구조가 급격히 바뀐 실제 역사적 사건이다. 장기 금융 시퀀스를 입력하는 가상의 LSTM에서는 일부 셀이 유동성 악화, 변동성 상승, 상관관계 확대와 같은 지속 패턴을 기록하고 forget gate가 높게 유지되는 동안 그 영향을 전달할 수 있다. 시장이 정상화되는 증거가 반복되면 gate가 이전 메모리를 점차 줄일 수 있다.

최근 5일의 가격 모멘텀은 다른 셀 차원에서 빠르게 기록되고, 반전 신호가 들어오면 빨리 잊힐 수 있다. 하나의 LSTM 안에서도 셀별 forget gate가 다르므로 서로 다른 시간척도를 표현할 가능성이 있다.

그러나 게이트 값에 ‘리먼 기억’, ‘팬데믹 기억’이라는 이름이 자동으로 붙는 것은 아니다. 이것은 가능한 메커니즘의 해석이다. 실제로 그런 정보를 사용했는지 확인하려면 해당 역사 자료가 입력에 있어야 하고, 셀 상태·게이트·교란 실험을 분석해야 한다.

### 9.4.2 이번 60일 모델이 기억할 수 없는 것

60영업일 lookback은 약 3개월이다. 2008년과 2020년을 한 입력 창에 동시에 기억할 수 없다. 이번 합성 데이터에는 해당 사건도 없다. 모델 가중치는 과거 훈련 창에서 본 ‘위기와 비슷한 국소 패턴’을 학습할 수 있지만, 15년 전 사건의 hidden state를 현재 창까지 운반하지 않는다.

장기 사건을 직접 조건으로 사용하려면 lookback을 늘리거나 장기 요약 피처·거시 상태를 입력하거나, 상태를 연속해서 운반하는 설계를 검토할 수 있다. 상태를 운반할 경우 배치 경계, 종목 경계, 학습 중 detach, 운영 재시작 시 상태 보관을 엄격히 관리해야 한다.

### 9.4.3 게이트는 인과적 위험관리 규칙이 아니다

forget gate가 높다고 과거 위기가 현재 가격에 인과적으로 영향을 준다는 증거는 아니다. 입력 상관관계로 손실을 줄인 내부 계산이다. 게이트 벡터는 여러 차원이며 단일 숫자로 전체 모델 설명을 대신할 수 없다. 경제적 설명에는 시간별 예측 민감도, 입력 삭제, 대조 시나리오, 구간 외 검증을 함께 사용해야 한다.

## 9.5 GRU의 구조와 LSTM 비교

GRU는 별도의 셀 상태 없이 h만 유지한다. reset gate r과 update gate z를 사용한다. PyTorch가 사용하는 순서에 맞추면 다음과 같다.

$$\boldsymbol{r}_t=\sigma(W_{ir}\boldsymbol{x}_t+\boldsymbol{b}_{ir}+W_{hr}\boldsymbol{h}_{t-1}+\boldsymbol{b}_{hr}).$$

$$\boldsymbol{z}_t=\sigma(W_{iz}\boldsymbol{x}_t+\boldsymbol{b}_{iz}+W_{hz}\boldsymbol{h}_{t-1}+\boldsymbol{b}_{hz}).$$

후보 상태는 reset gate를 은닉 선형변환에 적용한다.

$$\boldsymbol{n}_t=\tanh(W_{in}\boldsymbol{x}_t+\boldsymbol{b}_{in}+\boldsymbol{r}_t\odot(W_{hn}\boldsymbol{h}_{t-1}+\boldsymbol{b}_{hn})).$$

마지막 상태는 과거와 후보의 보간이다.

$$\boldsymbol{h}_t=(1-\boldsymbol{z}_t)\odot\boldsymbol{n}_t+\boldsymbol{z}_t\odot\boldsymbol{h}_{t-1}.$$

z가 1에 가까우면 과거 상태를 유지하고 0에 가까우면 후보로 교체한다. reset gate가 0에 가까우면 후보를 만들 때 과거 은닉변환의 영향을 줄인다. 문헌이나 프레임워크에 따라 update gate 기호나 후보 계산의 배치 위치가 다를 수 있으므로 구현 식과 맞춰야 한다. [PyTorch GRU 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html).

### 9.5.1 파라미터 수

PyTorch 단층 구조에서 게이트 하나는 입력가중치 md, 은닉가중치 m², 입력·은닉 편향 2m을 갖는다. LSTM은 네 묶음, GRU는 세 묶음이다. 출력층은 같다.

$$P_{LSTM}=4(md+m^2+2m)+(m+1).$$

$$P_{GRU}=3(md+m^2+2m)+(m+1).$$

$$P_{RNN}=md+m^2+2m+(m+1).$$

d=6, m=16이면 GRU는 LSTM보다 순환층 파라미터가 25% 적다. 일반적으로 메모리와 계산량도 작을 수 있다. 실제 속도는 커널 구현, 하드웨어, 배치, 시퀀스 길이에 따라 달라지므로 파라미터 비율만으로 wall-clock 시간을 단정하지 않는다.

| 구조 | 기억 상태 | 게이트 묶음 | 장점 후보 | 제한 |
|---|---|---:|---|---|
| Vanilla RNN | h | 없음 | 가장 단순·가벼움 | 장기 기울기 문제 |
| LSTM | h와 c | 4 | 셀의 명시적 가산 메모리 | 파라미터·계산 증가 |
| GRU | h | 3 | LSTM보다 경량 | 별도 셀·출력 게이트 없음 |

어느 구조가 금융 예측에 항상 낫다는 규칙은 없다. 데이터 크기와 기억 길이, 노이즈, 정규화, 학습 절차에 따라 검증한다. 이번 실습은 세 구조를 같은 hidden size·학습률·fold에서 비교하며 시험 결과를 보고 구조를 다시 선택하지 않는다.

## 9.6 예측 표적과 OHLCV 데이터

### 9.6.1 가격보다 수익률을 예측한다

t일 종가 Pₜ에서 H=5영업일 뒤 단순 수익률을 정의한다.

$$y_t^{(5)}=\frac{P_{t+5}}{P_t}-1.$$

모델은 이를 %포인트 단위로 학습한다. 예측 수익률을 가격으로 바꾸면 다음과 같다.

$$\hat{P}_{t+5}=P_t(1+\hat{y}_t^{(5)}).$$

가격 추세를 그대로 맞추는 것이 아니라 현재 가격을 기준점으로 미래 변화를 예측한다. `target_price`는 검증과 그림에 사용할 수 있지만 모델 학습 표적은 수익률이다.

5일 수익률은 겹친다. t와 t+1의 표적은 네 날짜의 가격 정보를 공유하므로 평가 오차들이 독립이라고 보기 어렵다. 600개 예측이 있다고 해서 독립 표본 600개와 같은 표준오차를 갖는 것은 아니다.

### 9.6.2 여섯 입력 피처

각 날짜의 Open, High, Low, Close, Volume, 과거 20일 일별 수익률의 연율화 표준편차를 사용한다.

$$Vol_t=\sqrt{252}\operatorname{SD}(r_{t-19},\ldots,r_t).$$

OHLCV는 t일 장 마감 후 확정된 값이라고 가정한다. 따라서 t일 종가 직후 모델을 실행해 이후 거래 시점에 사용할 수 있다. t일 종가로 즉시 체결했다고 가정하는 백테스트는 하지 않는다.

합성 OHLC는 High가 Open·Close 이상이고 Low가 Open·Close 이하가 되도록 생성한다. 실제 데이터에서는 주식분할·배당·수정주가의 일관성, 거래량 조정, 휴장, 비동시 거래, 상장폐지를 처리해야 한다. 가격 수준과 거래량은 비정상적이고 스케일이 바뀔 수 있다. 이번 실습은 요청된 OHLCV 원열을 그대로 사용해 fold별 표준화를 수행하지만, 실무에서는 log return, 고저 범위, 갭, log volume 변화 같은 정상화 피처와 비교해야 한다.

```python
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
```

![합성 종가와 후행 변동성](assets/fig02_market.png)

## 9.7 3차원 텐서와 Dataset

### 9.7.1 한 표본의 모양

예측 기준일 t의 한 표본은 t−59일부터 t일까지 60행, 여섯 피처를 가진다.

$$X_t=[\boldsymbol{x}_{t-59},\ldots,\boldsymbol{x}_t]\in\mathbb{R}^{60\times6}.$$

배치 B개를 쌓으면 다음 모양이다.

$$\mathcal{X}\in\mathbb{R}^{B\times60\times6}.$$

PyTorch에서 `batch_first=True`를 사용하므로 이 순서다. 기본 설정에서는 시간×배치×피처 순서이므로 옵션과 입력을 맞춰야 한다. 라벨은 B차원이다.

### 9.7.2 창의 끝과 라벨 시점을 구별한다

입력창의 마지막은 t이고 라벨의 마지막은 t+5다. 훈련 기준일의 마지막이 검증 기준일보다 앞선 것만으로 충분하지 않다. 훈련 라벨이 검증 입력 시작 전에 확정되어야 한다.

$$\max(t_{train}+H)<\min(t_{valid}).$$

$$\max(t_{valid}+H)<\min(t_{test}).$$

코드는 두 구간 사이에 H개 기준일을 비워 이 strict inequality를 검사한다. 같아도 시각상 마감·실행 순서에 따라 위험할 수 있으므로 보수적으로 다음 영업일부터 시작한다.

### 9.7.3 스케일러가 볼 수 있는 범위

각 fold의 스케일러는 첫 유효 피처일부터 그 fold의 마지막 **훈련 기준일**까지 원행만 사용한다. 각 훈련 창은 이 범위 안에 있으므로 미래 검증 피처가 평균·표준편차에 들어가지 않는다.

$$x_{t,j}^{scaled}=\frac{x_{t,j}-\mu_{j,train}}{s_{j,train}}.$$

한 fold에서 학습한 스케일을 그 fold의 검증·시험에 그대로 적용한다. 시험 구간의 가격 수준이 훈련 범위를 크게 벗어나면 큰 표준점수가 생길 수 있다. 이를 시험 평균으로 다시 중심화하면 미래 정보를 사용하게 된다.

```python
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
```

## 9.8 PyTorch RNN·LSTM·GRU 모델과 학습

### 9.8.1 공통 모델 인터페이스

`SequenceRegressor`는 문자열에 따라 `nn.RNN`, `nn.LSTM`, `nn.GRU` 중 하나를 만든다. 입력크기 6, 은닉크기 16, 단층, 단방향이다. 각 독립 창의 초기상태는 명시적으로 전달하지 않아 PyTorch가 0으로 만든다.

순환층은 모든 시점의 은닉 출력을 반환한다. 마지막 시점 `sequence[:, -1, :]`만 꺼내 16→1 선형 head로 5일 수익률을 예측한다. bidirectional RNN은 현재 창 안에서는 미래 날짜가 아닌 과거 60일을 양방향으로 읽을 수 있지만, 시간 인과 해석이 달라지므로 이번에는 쓰지 않는다.

```python
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
```

| model | parameters |
| --- | --- |
| RNN | 401 |
| LSTM | 1553 |
| GRU | 1169 |


LSTM의 파라미터가 많은 것은 네 게이트 묶음을 사용하기 때문이다. 동일 hidden size는 동일 모델 복잡도를 뜻하지 않는다. 이 비교는 구조를 같게 유지한 교육용 비교이며 파라미터 수까지 같춘 공정성 실험은 아니다.

### 9.8.2 학습 루프

손실은 %포인트 수익률의 MSE다.

$$L=\frac{1}{B}\sum_{i=1}^{B}(y_i-\hat{y}_i)^2.$$

Adam 학습률은 0.002, 최대 epoch는 40, patience는 7이다. 검증 MSE가 10⁻⁵ 이상 개선될 때 checkpoint를 갱신한다. 훈련 배치만 섞고 검증과 시험 구간은 시간 경계를 유지한다. clipping 전 norm이 1보다 큰 배치 비율도 기록한다.

마지막 epoch의 가중치가 아니라 검증 MSE가 가장 낮았던 checkpoint를 복원한다. 각 fold의 시험 구간은 early stopping에 사용하지 않는다. 저장한 LSTM을 다시 불러 시험 예측이 같은지 검사한다.

## 9.9 Walk-Forward 백테스트 설계

### 9.9.1 세 개의 평가 구간

각 fold는 expanding training window를 사용한다. 첫 fold보다 두 번째, 세 번째 fold의 훈련 기간이 길다. 각 fold는 200개 검증 기준일과 200개 시험 기준일을 갖는다. 라벨 확정 gap 때문에 훈련·검증 사이와 검증·시험 사이의 일부 기준일은 사용하지 않는다.

시험 기준일은 fold 사이에서 겹치지 않는다. 그러나 각 시험의 5일 라벨 구간은 인접 기준일끼리 겹친다. 이것은 동일한 예측을 중복 저장한다는 뜻은 아니며 손익과 통계 추론에서 의존성을 고려해야 한다는 뜻이다.

```python
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
```

| fold | train_windows | valid_windows | test_windows | train_last | train_label_last | valid_first | valid_last | test_first | test_last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 712 | 200 | 200 | 2018-01-11 | 2018-01-18 | 2018-01-19 | 2018-10-25 | 2018-11-02 | 2019-08-08 |
| 2 | 912 | 200 | 200 | 2018-10-18 | 2018-10-25 | 2018-10-26 | 2019-08-01 | 2019-08-09 | 2020-05-14 |
| 3 | 1112 | 200 | 200 | 2019-07-25 | 2019-08-01 | 2019-08-02 | 2020-05-07 | 2020-05-15 | 2021-02-18 |


### 9.9.2 기준 모델

신경망만 비교하면 예측력이 있는지 알기 어렵다. 세 기준을 둔다.

- `Zero`: 모든 미래 수익률을 0으로 예측한다.
- `TrainingMean`: 해당 fold 훈련 라벨 평균을 예측한다.
- `Past5DayMomentum`: 직전 5일 수익률을 다음 5일 예측으로 사용한다.

Zero 예측의 부호는 0이고 실제 합성 연속수익률이 정확히 0일 가능성은 거의 없으므로 방향 적중으로 세지 않는다. 훈련 평균과 모멘텀은 시험 미래를 사용하지 않는다. 모멘텀은 학습된 모델이 아니지만 강한 단순 기준이다.

### 9.9.3 MAE·RMSE·Hit Ratio

수익률은 내부적으로 %포인트이고 결과표에서 MAE와 RMSE를 다시 100배해 bp로 표시한다.

$$MAE=\frac{1}{M}\sum_i|y_i-\hat{y}_i|.$$

$$RMSE=\sqrt{\frac{1}{M}\sum_i(y_i-\hat{y}_i)^2}.$$

$$HitRatio=\frac{1}{M}\sum_i1[\operatorname{sign}(y_i)=\operatorname{sign}(\hat{y}_i)].$$

RMSE는 큰 오류를 제곱하여 더 민감하고 MAE는 절대오차의 평균이다. Hit Ratio는 크기를 무시한다. +0.01%를 맞힌 것과 +5%를 맞힌 것이 같은 한 건이고, 큰 손실 방향 오류도 한 건이다. 거래 수익을 계산하는 지표가 아니며 거래비용·포지션 크기를 포함하지 않는다.

| model | MAE_bp | RMSE_bp | Hit_Ratio | fold |
| --- | --- | --- | --- | --- |
| Zero | 158.02738354 | 194.92091749 | 0.00000000 | 1 |
| TrainingMean | 157.47637063 | 194.53817527 | 0.54000000 | 1 |
| Past5DayMomentum | 212.73957897 | 261.11831718 | 0.53500000 | 1 |
| RNN | 162.54562120 | 199.33539909 | 0.46000000 | 1 |
| LSTM | 161.20454589 | 198.04370789 | 0.46000000 | 1 |
| GRU | 160.37043795 | 196.63822681 | 0.46000000 | 1 |
| Zero | 137.55550603 | 168.46802456 | 0.00000000 | 2 |
| TrainingMean | 139.28635468 | 169.80113875 | 0.47500000 | 2 |
| Past5DayMomentum | 209.77442620 | 255.35900362 | 0.41500000 | 2 |
| RNN | 138.56347020 | 173.47686477 | 0.55000000 | 2 |
| LSTM | 148.11631414 | 185.51213017 | 0.52500000 | 2 |
| GRU | 137.08602588 | 166.57599302 | 0.47500000 | 2 |
| Zero | 178.77718333 | 220.72354815 | 0.00000000 | 3 |
| TrainingMean | 178.62435503 | 220.39530831 | 0.53000000 | 3 |
| Past5DayMomentum | 260.92865515 | 320.91647855 | 0.48500000 | 3 |
| RNN | 179.64062043 | 222.34035408 | 0.48000000 | 3 |
| LSTM | 179.61649688 | 222.28134084 | 0.47000000 | 3 |
| GRU | 180.97289592 | 223.17019001 | 0.47000000 | 3 |


| model | MAE_bp | RMSE_bp | Hit_Ratio |
| --- | --- | --- | --- |
| Zero | 158.12002430 | 195.86944979 | 0.00000000 |
| TrainingMean | 158.46236011 | 196.00307845 | 0.51500000 |
| Past5DayMomentum | 227.81422011 | 280.70054180 | 0.47833333 |
| RNN | 160.24990394 | 199.38577097 | 0.49666667 |
| LSTM | 162.97911897 | 202.52164918 | 0.48500000 |
| GRU | 159.47645325 | 196.82402242 | 0.46833333 |


세 Walk-Forward 시험 구간을 연결한 결과, Zero 기준의 RMSE는 **195.8694bp**였다. RNN은 **199.3858bp**, LSTM은 **202.5216bp**, GRU는 **196.8240bp**로 모두 더 컸다. 방향 적중률도 각각 **49.67%**, **48.50%**, **46.83%**였다. 합성 생성 과정에 약한 순차 구조가 있어도 5일 수익률 잡음이 크고 원 OHLCV 입력이 비정상적이어서, 이 설계의 순환모형은 유용한 미래 예측력을 입증하지 못했다.

![각 fold의 LSTM 학습·검증 손실](assets/fig03_training.png)

훈련 손실과 검증 손실의 차이는 과적합의 단서지만 한 곡선만으로 원인을 확정하지 않는다. 각 fold에서 early stopping epoch가 다른 것은 데이터 길이와 시장 구간이 바뀌기 때문이다.

### 9.9.4 실행시간을 해석하는 법

| fold | model | best_epoch | epochs_run | seconds |
| --- | --- | --- | --- | --- |
| 1 | RNN | 3 | 10 | 1.15966830 |
| 1 | LSTM | 1 | 8 | 0.96161140 |
| 1 | GRU | 6 | 13 | 2.99117880 |
| 2 | RNN | 35 | 40 | 5.69320920 |
| 2 | LSTM | 21 | 28 | 3.98788950 |
| 2 | GRU | 14 | 21 | 5.74832440 |
| 3 | RNN | 13 | 20 | 3.49163190 |
| 3 | LSTM | 6 | 13 | 2.26648280 |
| 3 | GRU | 15 | 22 | 7.18710190 |


`seconds`는 현재 CPU 환경의 한 번 실행시간이다. 라이브러리·하드웨어·스레드·운영체제에 따라 달라진다. 파라미터가 적은 GRU가 반드시 모든 환경에서 LSTM보다 빠르다고 결론내리지 않는다. 예측 성능 역시 이 합성 한 번의 결과일 뿐이다.

## 9.10 수익률 예측과 가격 예측을 함께 보기

Walk-Forward의 모든 시험 예측을 날짜순으로 연결한다. 위 패널은 실제 5일 수익률과 LSTM 예측이다. 아래 패널은 각 기준일의 현재 종가에 실제·예측 수익률을 각각 적용한 t+5 가격이다.

![Walk-Forward 5일 수익률과 조건부 가격 점추정](assets/fig04_forecasts.png)

아래 가격선은 한 날짜에서 시작해 연속 복리로 이어지는 백테스트 가치곡선이 아니다. 매 기준일마다 서로 다른 현재 가격에 5일 예측을 적용한 점들의 연결이다. 겹치는 5일 horizon 때문에 인접 점은 서로 강하게 관련된다.

예측 가격이 실제 가격과 가까워 보이는 현상 중 상당 부분은 둘 다 큰 현재 가격 Pₜ를 공유하기 때문이다. 따라서 모델 비교는 수익률 오차로 수행한다. 가격 그림은 사용자가 수익률 예측의 규모를 해석하는 보조 수단이다.

방향 적중률이 50%보다 높더라도 경제적으로 유용한 전략이 보장되지 않는다. 신호가 작은 날 거래하지 않는 정책, 거래비용, 슬리피지, 리밸런싱 빈도, 포지션 상한, 위험 예산이 필요하다. 이것들은 이번 예측 모델 평가에 포함하지 않는다.

## 9.11 알려지지 않은 미래를 예측하는 추론 코드

평가 구간에서는 라벨이 이미 존재해 오차를 계산했다. 운영 시점의 가장 최근 60일에는 아직 t+5 가격이 없다. 마지막 fold에서 저장한 스케일러와 LSTM을 불러 최근 창을 변환하고 예측한다.

```python
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
```

가장 최근 합성 날짜 **2021-04-29**에서 fold 3 LSTM이 출력한 향후 5영업일 수익률은 **-0.1397%**, 현재 종가 **151.0295**에 적용한 조건부 가격 점추정은 **150.8186**다. 정답이 아직 없는 운영형 예시이며 성능 평가값이 아니다.

이 예측에는 실제 정답이 없으므로 MAE나 적중 여부를 계산하지 않는다. 또한 fold 3 모델을 최신 전체 확정 라벨까지 재학습하지 않은 상태로 그대로 사용한다. 운영 모델을 만들려면 구조·학습 epoch 등 개발 선택을 고정한 뒤 현재 이용 가능한 모든 **라벨 확정 자료**로 다시 학습하고 새 버전으로 저장하는 절차가 필요하다.

저장 항목은 모델 가중치, 피처 순서, lookback, horizon, hidden size, 표적 단위, fold별 스케일 평균과 표준편차다. `lstm_fold3.pt`만 가져오고 스케일러를 잃으면 같은 예측을 재현할 수 없다.

## 9.12 금융 시계열 모델의 흔한 실패

### 원가격 스케일 누수

전체 기간의 평균·표준편차로 OHLCV를 변환하면 미래 가격 수준이 과거 학습에 반영된다. 각 fold의 훈련 원행만 사용해야 한다. 종가를 수정주가로 쓰면서 Open·High·Low는 미수정값으로 두는 것도 내부 일관성을 깨뜨린다.

### 라벨 종료일 무시

t+5 수익률은 t+5가 지나야 안다. 훈련 기준일 t가 검증 시작보다 앞섰다는 이유만으로 안전하지 않다. 라벨 종료 시점을 비교해야 한다.

### 창을 먼저 만든 뒤 무작위 분할

겹치는 60일 창을 무작위로 나누면 훈련과 시험 창이 59일을 공유할 수 있다. 매우 비슷한 입력이 양쪽에 들어간다. 원시 시간축에서 구간을 정한 뒤 창과 라벨의 경계를 검사한다.

### 독립 표본처럼 유의성 계산

5일 중첩 수익률과 60일 중첩 입력은 의존적이다. 단순한 독립 t-test나 √M 표준오차는 불확실성을 과소평가할 수 있다. 비중첩 기준일 평가나 적절한 블록 재표본화를 고려한다.

### 양방향 RNN의 미래 누수 오해

과거 60일만 입력한 bidirectional RNN이 그 창의 앞뒤 방향을 읽는다고 자동으로 t 이후 미래를 본 것은 아니다. 그러나 sequence-to-sequence로 창 안 각 날짜를 예측할 때 backward state가 해당 날짜보다 뒤의 관측을 사용하면 목표 정의에 따라 누수가 될 수 있다. 데이터 창과 예측 시점을 구체적으로 확인해야 한다.

### 재현성만으로 타당성을 주장

시드를 고정하고 결과를 다시 만들 수 있다는 것은 중요한 공학 조건이다. 합성 구조가 현실적이거나 미래 시장에서도 성능이 유지된다는 증거는 아니다. 여러 시장 국면과 실제 point-in-time 데이터에서 검증해야 한다.

## 9.13 확인 문제와 해설

**문제 1. RNN에서 Wₕₕ의 스펙트럴 노름이 0.8이면 60일 전 기울기가 반드시 0.8⁶⁰인가?**

해설: 아니다. 일반 RNN은 시점별 Tanh Jacobian과 방향을 함께 곱한다. 0.8⁶⁰은 매우 단순화한 상한 또는 특별한 스칼라 경로의 직관이다.

**문제 2. LSTM forget gate가 1이면 과거 정보가 완벽히 보존되는가?**

해설: 직접 셀 경로의 이전 c 항은 보존된다. 그러나 input gate가 새 값을 더하고 output gate가 노출을 조절하며 전체 네트워크의 다른 미분 경로도 존재한다. 학습·수치 오차와 다른 셀 차원까지 포함한 완전한 보존을 뜻하지 않는다.

**문제 3. GRU update gate z가 1에 가까우면 어떻게 되는가?**

해설: hₜ≈hₜ₋₁이 되어 과거 상태를 유지한다. 이 장이 사용한 PyTorch 식 기준이다. 다른 자료의 기호 정의가 반대일 수 있으므로 수식을 확인한다.

**문제 4. 60일 창의 마지막이 검증 시작 전날이면 안전한가?**

해설: 5일 미래 수익률 라벨은 검증 기간 안에서 끝날 수 있다. 훈련 기준일+5가 검증 시작보다 앞서야 한다.

**문제 5. 가격 MAE가 작으면 수익률 예측력이 높은가?**

해설: 예측가격과 실제가격이 현재 가격을 공유하면 가격 수준 차이가 작아 보일 수 있다. 수익률 오차와 방향을 기준 모델과 비교한다.

**코드 실습.** 원 OHLCV 대신 전일 대비 log price 변화, 고저 범위, 시가 갭, log volume 변화를 만들고 같은 fold로 평가해 보라. 시험 결과를 보고 변환을 반복 선택하지 말고 검증 구간에서 결정한다. 다음으로 lookback 20·60·120을 비교하되, 가장 긴 창 때문에 각 후보의 첫 학습일이 달라지는 문제를 통제하라.

## 9.14 실행 환경과 산출물

Python 3.12.14에서 실행했다.

| package | version |
| --- | --- |
| numpy | 2.3.5 |
| pandas | 3.0.1 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |
| torch | 2.14.0+cpu |


```text
python -m pip install -r requirements.txt
python chapter09_examples.py
```

`synthetic_ohlcv.csv`에는 OHLCV·후행 변동성·5일 표적이 있다. `walkforward_predictions.csv`에는 겹치지 않는 600개 시험 기준일의 실제 수익률, 기준 모델, RNN·LSTM·GRU 예측, 가격 변환이 있다. `lstm_fold1.pt`부터 `lstm_fold3.pt`, 각 fold 스케일러, `model_config.json`을 함께 보관한다.

코드는 스칼라 BPTT 미분, LSTMCell 5단계, GRUCell, OHLC 일관성, `[Batch,60,6]` 모양, 라벨 확정 gap, 시험 기준일 비중복, 모델 재로딩을 검사한다. HTML은 그림과 수식을 내장한 오프라인 열람본이고 Markdown은 수정용 원고다.

실제 자료로 교체하거나 파라미터를 바꾸어 스크립트를 실행하면 CSV·JSON·그림은 갱신된다. 이미 작성된 결과표와 해석은 자동으로 다시 집필되지 않으므로 새 결과에 맞춰 함께 수정해야 한다.

