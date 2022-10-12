# LLE

**L**ocally **L**inear **E**mbedding

![이미지 출처: 고려대학교 산업경영공학과 대학원 강필성 교수님 Business-Analytics 강의 01_4_Dimensionality Reduction_ISOMAP, LLE, tSNE 강의 자료](LLE%205f6e9afe8b2542f98cf2f84bcf176880/Untitled.png)

이미지 출처: 고려대학교 산업경영공학과 대학원 강필성 교수님 Business-Analytics 강의 01_4_Dimensionality Reduction_ISOMAP, LLE, tSNE 강의 자료

- LLE는 ISOMAP과 동일하게 **Non-linear 차원 축소 방법**임
- 고차원 공간에 존재하는 인접 Data Point들 간의 **선형적 구조**를 **보존**하며 저차원으로 Embedding함
- 고차원 공간에 존재하는 Manifold를 표현하기 위해 위 그림 중앙(B)의 검은색 원 부분과 같은 좁은 범위에서 산출한 선형 모델을 활용하는 알고리즘

## LLE 요약

- **현재 차원**에서 “나”를 잘 표현하는 이웃들과 각 이웃과의 가중치를 이용해서 **더 낮은 차원**에서도 동일한 가중치를 이용해서 “나”와 “이웃”들을 표현하자

## LLE의 장점

- 사용하기 간단함
- 최적화가 Local Minima를 포함하지 않음
- Non-linear Embedding을 생성할 수 있음
- 고차원 데이터를 저차원의 단일 전역 좌표계(Single Global Coordinate System)로 Mapping함

## LLE Procedure

- Step 1. 각 Data Point의 Neighbors 계산 및 Neighborhood Graph 생성
- Step 2. 가중치 행렬 $\bold{W}$ 산출
- Step 3. 저차원으로 Embedding

![이미지 출처: 고려대학교 산업경영공학과 대학원 강필성 교수님 Business-Analytics 강의 01_4_Dimensionality Reduction_ISOMAP, LLE, tSNE 강의 자료](LLE%205f6e9afe8b2542f98cf2f84bcf176880/Untitled%201.png)

이미지 출처: 고려대학교 산업경영공학과 대학원 강필성 교수님 Business-Analytics 강의 01_4_Dimensionality Reduction_ISOMAP, LLE, tSNE 강의 자료

### Step1. 각 Data Point의 Neighbors 계산 및 Neighborhood Graph 생성

- ISOMAP에서의 Neighborhood Graph 생성 방법과 동일
    - $\epsilon$-기준
        
        두 Data Point 간의 거리가 $\epsilon$ 보다 작을 경우 연결
        
    - $k$-기준
        
        만약 데이터 포인트 두 개를 i,j라고 할 때, i가 j의 $k$-Nearest Neighbor라면 i와 j를 연결
        
        ! k는 사용자가 지정하는 하이퍼 파라미터
        

### Step2. 가중치 행렬 $\bold{W}$ 산출

- 각 Data Point는 자신의 Neighbor들의 Weighted Sum으로 Reconstruction 됨
    
    : 각 Data Point의 Neighbor로부터 선형적으로 각 Data Point를 가장 잘 Reconstruction하는 가중치 행렬을 구함
    
- 전체 Data Point가 아닌 각 Data Point의 **Neighbors**의 Weight만 사용하고 → “Locally”
Neighbor들을 이용해 **선형 결합**으로 Data Point를 나타내기에 → “Linear”
“**Locally**” “**Linear**” Embedding임
    
    ---
    
    $E(\bold{W}) = \sum_i|\bold{x}_i - \sum\bold{W}_{ij}\bold{x}_j|^2$    → Eq(1)
    
    s.t. $\bold{W}_{ij} = 0$ 만약 $\bold{x}_j$가 $\bold{x}_i$의 Neighbor가 아니라면
    
    $\sum_i\bold{W}_{ij} = 1$ for all $i$
    
    ---
    
    위의 Eq(1) ($E(\bold{W})$)를 Reconstruction Error라고 하고, 이 식을 최소화하는 가중치 행렬의 원소 $\bold{W}_{ij}$들을 찾는 것이 목적이다. 
    

### Step3. 저차원으로 Embedding

- Step2를 통해 구한 가중치 행렬 $\bold{W}$을 이용해 Data Point들을 저차원으로 Embedding
    - Step2에서 구한 가중치 $\bold{W}_{ij}$는 Data Point i와 Neighbor들 간의 Locally Linear Relationship을 나타냄.
        
        ⇒ Step3에서는 이 Locally Linear Relationship을 **최대한 보존하면서** 데이터를 저차원 공간으로 Mapping함
        
- $y_i$를 저차원 공간에 Embedding된 Data Point i라고 한다면 $y_i$와 저차원 공간 상에서 $y_i$의 Neighbor들에 대해 Reconstruction(재구성)된 $\sum_{j=1}\bold{W}_{ij}\bold{y}_j$ 간의 거리를 최소화하는 $\bold{W}$를 찾는 Minimize 문제가 된다.

---

$$min$ $\Phi{(\bold{W})} = \sum_i|\bold{y}_i - \sum_{j=1}\bold{W}_{ij}\bold{y}_j|^2$  ⇒  $min$ $\Phi(\bold{W}) = \sum_{i,j}\bold{M}_{ij}(\bold{y}_i\centerdot\bold{y}_j)$

where, $$\bold{M}_{ij} = \delta_{ij} - \bold{W}_{ij} - \bold{W}_{ji} + \sum_k\bold{W}_{ki}\bold{W}_{kj}, \delta_{ij} = 1$ if $i = j$, 0 otherwise

s.t. $$\sum_i\bold{y}_i = 0$, ${1\over{n}}\sum_i\bold{y}\bold{y}^T = \bold{I}$

---

> 위 제약 조건의 의미
> 
> - $$\sum_i\bold{y}_i = 0$    → Embedding된 저차원 공간 상에 각 변수의 평균 = 0
> - $${1\over{n}}\sum_i\bold{y}\bold{y}^T = \bold{I}$    →  Embedding된 저차원 공간 상에서의 각 변수들 서로 직교함

---

$$min$ $\Phi{(\bold{W})} = \sum_i|\bold{y}_i - \sum_{j=1}\bold{W}_{ij}\bold{y}_j|^2$

                      $$= [(\bold{I} - \bold{W})\bold{y}]^T (\bold{I} - \bold{W})\bold{y}$

                      $$= \bold{y}^T(\bold{I}-\bold{W})^T(\bold{I}-\bold{W})\bold{y}$

                      $$= \bold{y}^T\bold{M}\bold{y}$

⇒ $$min$ $$\bold{y}^T\bold{M}\bold{y}$  → Eq(2)

    s.t. $$\sum_i\bold{y}_i = 0$, ${1\over{n}}\sum_i\bold{y}\bold{y}^T = \bold{I}$

- 식 2를 라그랑지안 함수 $L$로 나타낸 후, 해당 함수 $L$을 $\bold{Y}$에 대해 편미분 즉, $\partial{L} \over \partial{Y}$ = 0을 하면 $\bold{y}$의 Eigen Vector 및 Eigen Value를 구할 수 있음

<aside>
💡 PCA는 목적이 “분산 최대화”였기에, Eigen Value가 큰 순서대로 해당하는 Eigen Vector를 사용했음.

</aside>

- LLE는 위 Eq(2)와 같이 Minimize 문제이기에 Eigen Value가 작은 Eigen Vector부터 사용함
