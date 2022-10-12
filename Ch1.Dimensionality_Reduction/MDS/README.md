# MDS

Multidimensional Scaling

- 두 Object 간의 거리가 최대한 잘 보존이 되도록하는 저차원 공간으로 각 Object를 차원 축소
- Input: n개의 Object간 근접 행렬(Proximity Matrix) (size: n by n)을 사용
- Output: 저차원 d-dimension에서 각 Object의 좌표

# MDS Procedure

## Step1. Proximity/Distance Matrix 생성

- 각 Object에 대한 좌표가 존재한다면  Object 간의 유사도 또는 거리를 계산
    - Distance: Euclidean, Manhattan, …
    - Similarity: Correlation, Jaccard, …

## Step2. Distance 정보를 보존하는 좌표 추출

- Distance Matrix $D$의 각 Element는
    
    $d_{rs}^2 = (x_r - x_s)^T(x_r - x_s)$ 로 표현됨
    
- 다음과 같이 내적 행렬 $B$는 Distance Matrix $D$로부터 얻을 수 있음
    
    ! 원래는 $D$ → $X$를 바로 구하고 싶은데, 이것이 안 되니까 $D$  → $B$ → $X$ 과정을 수행함
    
    $[B]_{rs} = b_{rs} = x_r^Tx_s$
    
    - 모든 p 변수들의 평균은 0이라고 가정 → 계산 용이성을 위해
        
        $\sum_{r=1}^nx_{ri} = 0$, $(i = 1, 2, ..., p)$
        
        $d_{rs}^2 = x_r^Tx_r + x_s^Tx_s - 2x_r^Tx_s$
        

![Untitled](https://user-images.githubusercontent.com/56019094/195246883-f16ab02c-9197-4de8-90d8-054bad0da310.png)

![Untitled 1](https://user-images.githubusercontent.com/56019094/195246889-0e884820-d340-462f-a2d6-be9825cfc1a4.png)

![Untitled 2](https://user-images.githubusercontent.com/56019094/195246890-f89ef877-a54b-4915-b012-6546d0dda98c.png)

![Untitled 3](https://user-images.githubusercontent.com/56019094/195246893-64317362-13bb-469b-8472-3661bdd85f70.png)

- 행렬 $B$로부터 $X$의 좌표를 얻음 ($X$: n by p, p < n)
    - $B = XX^T$      $rank(B) = rank(XX^T)= rank(X) = p$
    
    - 행렬 $B$는 대칭이며, Positive semi-definite 행렬이고 rank는 p임 → $B$라는 행렬이 가지고 있는 intrinsic dim이 p겠구나!
        
        → p개의 non-negative 고유값과 (n-p)개의 zero 고유값을 가짐 
        
    
    $B = V\boldsymbol{\Lambda}V^T, \boldsymbol{\Lambda} = diag(\lambda_1, \lambda_2, ..., \lambda_n), V = [v_1, v_2, ...,v_n]$
    
    — (n-p)개의 Zero 고유값이 존재하기 때문에 →
    
    $B = V_1\boldsymbol{\Lambda}_1V_1^T, \boldsymbol{\Lambda}_1 = diag(\lambda_1, \lambda_2, ..., \lambda_p), V = [v_1, v_2, ...,v_p]$
    
    ⇒ 좌표 행렬 $X$는 $X = V_1\boldsymbol{\Lambda}_1^{1/2}$로 구할 수 있음
