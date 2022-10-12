# ISOMAP

**ISO**metric **MAP**ping

![Untitled](https://user-images.githubusercontent.com/56019094/195226123-5afd2c28-a5a3-4da4-a421-d1aa7ef5c9ee.png)

- **ISOMAP**은 PCA(주성분 분석)와 MDS(다차원 척도법)를 결합해 만들어낸 **Non-Linear 차원 축소 방법**임
- MDS가 Object 간의 Distance Information을 최대한 보존하는 것을 목적으로 만들었다면, ISOMAP은 **Manifold 상에서 Object 간의 Distance**를 이용해 고차원의 데이터를 저차원으로 축소시킴
- Swiss Roll Dataset
    
    ![Untitled 1](https://user-images.githubusercontent.com/56019094/195226126-2068d220-d429-4fd6-9361-a4a881677f6a.png)
    
    이미지 출처: 고려대학교 산업경영공학과 대학원 강필성 교수님 Business-Analytics 강의 01_4_Dimensionality Reduction_ISOMAP, LLE, tSNE 강의 자료
    
    - PCA와 MDS의 경우 위 그림 좌측에서와 같이 두 점 사이의 거리를 직선으로 산출하지만, 이는 실제 데이터 간의 거리라고 보기 어려움
    - 반면 ISOMAP을 이용하여 위 그림 우측과 같이 산출하면, PCA와 MDS에 비해 두 점 사이의 거리를 보다 실제에 가깝게 반영할 수 있음
    

## ISOMAP 알고리즘 Procedure

1. Neighborhood Graph 생성
2. Graph 상에서 두 Data Point를 잇는 최단 경로(Shortest Path) 계산
3. MDS를 활용해 d-dimensional (저차원) embedding 생성

각 단계에 대해 조금 더 자세히 살펴보면,

### Step1. Neighborhood Graph 생성

- Neighborhood Graph 생성 방법을 기준으로 ($\epsilon$-ISOMAP, $k$-ISOMAP) 두 가지 방법이 존재
    - $\epsilon$-ISOMAP
        
        두 Data Point 간의 거리가 $\epsilon$ 보다 작을 경우 연결
        
    - $k$-ISOMAP
        
        만약 데이터 포인트 두 개를 i,j라고 할 때, i가 j의 $k$-Nearest Neighbor라면 i와 j를 연결
        
        ![Untitled 2](https://user-images.githubusercontent.com/56019094/195226127-460e873e-4bae-4ea2-95f8-0e314bb9e844.png)

        
        이미지 출처: [https://jeheonpark93.medium.com/vc-isomap-manifolds-learning-965e758316eb](https://jeheonpark93.medium.com/vc-isomap-manifolds-learning-965e758316eb)
        
- Neighborhood Graph 생성 결과 예시
    
    ![Untitled 3](https://user-images.githubusercontent.com/56019094/195226130-2c72823d-87ed-4196-9b8a-a600b575bb9b.png)

    
    이미지 출처: [https://woosikyang.github.io/first-post.html](https://woosikyang.github.io/first-post.html)
    

### Step2. Graph 상에서 두 Data Point를 잇는 최단 경로(Shortest Path) 계산 → Distance Matrix 생성

- Step 2-1. 두 Data Point i, j의 연결 유무에 따라 $d_G(i,j)$ 다르게 초기화 
               ($d_G(i,j)$는 Data Point $i$와 $j$의 최단 거리를 의미)
    - 두 Data Point i, j이 서로 연결되어 있는 경우 → $d_G(i,j) = d_X(i,j)$ 으로 초기화
    - 두 Data Point i,j이 서로 연결되어 있지 않은 경우 → $d_G(i,j) = \infin$ 으로 초기화
- Step 2-2.  $k = 1, 2, ..., N$에 대하여 $d_G(i,j) = min(d_G(i,j), d_g(i,k) + d_G(k,i))$로 값 대체

### Step3. MDS를 활용해 d-dimensional (저차원) embedding 생성

- 아래 그림은 ISOMAP을 통해 “Swiss roll” 데이터셋을 2차원 Embedding으로 축소한 뒤 시각화 한 것
- 빨간 선은 Step2까지의 과정을 통해 얻은 두 Data Point 간의 Shortest Path
- 파란 직선은 2차원 Embedding 공간에서의 두 Data Point 간의 Shortest Path를 빨간 선보다 더 단순하고 깔끔하게 근사(approximation)함

![Untitled 4](https://user-images.githubusercontent.com/56019094/195226132-59dec371-1f39-4dd7-8cd8-51f1c538ab68.png)


이미지 출처: 고려대학교 산업경영공학과 대학원 강필성 교수님 Business-Analytics 강의 01_4_Dimensionality Reduction_ISOMAP, LLE, tSNE 강의 자료
