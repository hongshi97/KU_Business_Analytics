# Dimensionality Reduction

> Curse of dimensionality
: Feature의 수가 증가 → 동일한 설명 능력을 갖기 위해 필요한 Instance의 수가 급격히 증가
> 

<aside>
💡 “If there are various logical ways to explain a certain phenomenon, the simplest is the best” - Occam’s Razor

</aside>

- 때로 Data는 원래 차원에 비해 상대적으로 낮은 Intrinsic (= Latent/Embedded) 차원을 가짐

## 고차원으로 인해 발생 가능한 문제점

- Data 안에 Noise가 있을 확률 증가 → 예측 성능 저하
- 모델 훈련 또는 적용 시 필요한 계산량 증가
- 모델의 일반화 성능을 확보하기 위해 더 많은 데이터 필요

## ⇒ Curse of Dimensionality 해결 방법

- Domain Knowledge 사용
- Objective Funciton에서 Regularization term 사용 (e.g. 회귀 분석에서 Ridge, Lasso 등)
- **정량적인 축소 기법 사용 → 이것에 대해 알아보자.**

## Purpose of Dimensionality Reduction

- 모델의 성능을 저하시키지 않는 또는 덜 저하시키는 Feature subset을 찾아내자

## Effect of Dimensionality Reduction

- 변수 간의 상관 관계 제거
- 단순한 후처리
- 유용한 정보는 유지한 채 중복되거나 필요없는 변수들은 제거
- 시각화가 가능

# Dimensionality Reduction 방법론 분류

### 분류 기준 1) Supervised vs Unsupervised

- Supervised
    - 축소된 차원을 검증하기 위해 Data Mining 모델 사용
    - 사용되는 Data Mining 알고리즘에 따라 차원 축소 결과가 달라질 수 있음
- Unsupervised
    - 원래 Input 공간(고차원)에서의 분산, 거리 등과 같은 정보를 보존하는 저차원에서의 좌표계를 찾음
    - 위 프로세스 동안 Data Mining 모델을 사용하지 않음
    - 만약 데이터와 방법이 같다면 차원 축소 결과는 동일함

### 분류 기준 2) Feature Selection vs Feature Extraction

- Feature Selection
    - 원래 Feature Set으로부터 Subset 추출
    - Filter: 변수 선택과 모델 훈련이 독립
    - Wrapper: 사용 모델의 결과를 최적화하기 위해 변수 선택이 수행됨
- Feature Extraction
    - 원래 데이터의 특징을 보존하는 새로운 작은 Feature Set을 추출함(extract)
    - 사용되는 Data Mining Model과 독립적인 성능 측정 지표가 사용됨