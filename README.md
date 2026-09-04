# Chapter 10

- Chapter10.html: 수식과 그림이 내장된 오프라인 교재
- Chapter10.md: 수정용 원고
- chapter10_examples.py: 합성 시장 생성부터 Compact TFT 학습·평가까지 독립 실행 코드
- requirements-finbert.txt: 실제 FinBERT 경로를 사용할 때의 추가 패키지

```powershell
python -m pip install -r requirements.txt
python chapter10_examples.py
$env:FINBERT_INPUT="news.csv"
python chapter10_examples.py
```

기본 실행은 합성 가격과 합성 감성 점수를 사용하며 FinBERT 가중치를 다운로드하지 않습니다. 실제 FinBERT 입력 CSV에는 timestamp,headline 열이 필요합니다.
