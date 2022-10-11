# Genetic Algorithm

- 본 Readme 파일은 고려대학교 산업경영공학과 강필성 교수님의 Business Analytics 강의 자료를 기반으로 작성되었습니다.

![Untitled](https://user-images.githubusercontent.com/56019094/194996274-be67dc17-d500-4eb6-a1a6-aaf87b056af3.png)

## Heuristic-based Feature Selection 기법의 한계

- 전역 탐색: 최적 변수 집합 선정을 보장하지만 많은 시간이 소요됨
- 전진 선택/후진 소거/단계적 선택 → 전역 탐색에 비해서는 시간 효율적이나 최적 변수 집합을 찾을 가능성이 낮아짐

## ⇒ Then, What about GA(Genetic Algorithm)?

- 아이디어: 자연선택설에서 모티브를 얻음
    
    → 우수한 유전자(문제의 Solution)는 생식을 통해 다음 세대에서도 잘 발현될 수 있도록 함
    

## Genetic Algorithm의 Procedure

1. 선택(Selection): 현재 가능 해집합에서 우수한 해들을 선택하여 다음 세대를 생성하기 위한 부모 세대로 지정
2. 교배(Crossover): 선택된 부모 세대들의 유전자를 교환하여 새로운 세대를 생성
3. 돌연변이(Mutation): 낮은 확률로 변이를 발생시켜 Local Optimum에서 탈출할 수 있는 기회 제공

![Untitled 1](https://user-images.githubusercontent.com/56019094/194996275-9592ba3a-fb79-46ec-a6f1-0f7347e90d43.png)

⇒ Feature Selection 용도로 Genetic Algoritm은 어떻게 사용될까?

![Untitled 2](https://user-images.githubusercontent.com/56019094/194996276-9e2577de-37bd-4465-91c6-e235485b7870.png)

## Step1. Initiation

- 염색체(Chromosome) 인코딩
    - 염색체 인코딩 방식은 Task에 따라 다름
    - Feature Selection을 위한 GA에서는 Binary Encoding 사용
        
        ![Untitled 3](https://user-images.githubusercontent.com/56019094/194996278-901affb5-22aa-4b08-992b-2c5b8e666ce3.png)

        
- 세대 초기화
    - 염색체의 각 유전자(Gene)마다 난수 생성
    - 생성된 난수와 기준값(Cut-off) 비교 → Binary(0/1) 값으로 변환
    - Population Size = 8, Cut-off = 0.5 인 상황
        ![Untitled 4](https://user-images.githubusercontent.com/56019094/194996249-a38ce10b-49d6-4a06-a837-da3c9ab87823.png)

        

## Step2. 모델 학습

- 각 염색체에 담긴 정보(해당 변수의 모델링 사용 유무)를 활용하여 염색체 수만큼 모델을 학습
    - e.g. 다중 선형 회귀분석
        
        ![Untitled 5](https://user-images.githubusercontent.com/56019094/194996255-9b6e5d46-1623-4671-b305-2c63b167c658.png)

        
        ![Untitled 6](https://user-images.githubusercontent.com/56019094/194996256-c4d29fad-d201-4e50-8bd9-9474e5ab263d.png)

        
    

## Step3. 적합도 평가

- 각 염색체의 정보를 사용하여 학습된 모형(Step2를 통해 학습된 모델)의 적합도 평가
    - 적합도 함수: 염색체의 우열을 가릴 수 있는 정량적 지표
        - 적합도 함수 값이 높으수록 우수한 염색체 (우수한 변수 조합)
    - 적합도 함수가 가져야 하는 일반적인 조건
        - 두 염색체가 동일한 예측 성능을 나타낼 경우, 적은 수의 변수를 사용한 염색체를 선호
        - 두 염색체가 동일한 변수를 사용했을 경우, 우수한 예측 성능을 나타내는 염색체를 선호
    - 적합도 함수 예시: 선형 회귀 분석의 경우 Adjusted $R^2$, AIC, BIC 등이 사용될 수 있음

- Step2 예시에서 8개의 염색체에 대한 다중 선형 회귀분석의 Adj $R^2$ 산출
    
    ![Untitled 7](https://user-images.githubusercontent.com/56019094/194996257-a3783385-976b-4d70-a42d-4ebcc2e1826e.png)

    
    - Rank는 Adj $R^2$ 기준 순위를 나타냄
    - Weight는 개별 염색체의 Adj $R^2$을 전체 염색체들의 Adj $R^2$ 합으로 나눈 값 → 추후 확률적 염색체 선택 기법에서 사용됨

## Step4. 부모 염색체 선택

- 현재의 세대에서 높은 적합도를 나타내는 염색체들을 부모로 선택하여 다음 세대의 염색체를 생성하는데 사용
    - 확정적 선택(Deterministic Selection)
        - 적합도 기준 상위 N%에 해당하는 염색체만 부모 염색체로 사용 가능
        - 하위 (100-N)%에 해당하는 염색체는 부모 염색체로 사용될 가능성 없음
        - e.g. 확정적 선택 예시 (50%)
            
            ![Untitled 8](https://user-images.githubusercontent.com/56019094/194996259-e2b7fada-1db6-4935-b588-f0b31401748c.png)

            
            → 오직 1번, 2번, 4번, 8번 염색체만 다음 세대를 생성하는 부모 염색체의 역할을 수행할 수 있음
            
    - 확률적 선택(Probabilistic Selection)
        - 적합도 함수에 비례하는 Weight를 사용하여 부모 염색체를 선택
        - 모든 염색체가 부모 염색체로 선택될 가능성이 있으나 적합도 함수가 낮을수록 선택 가능성도 낮아짐
        - e.g. 1
            
            ![Untitled 9](https://user-images.githubusercontent.com/56019094/194996262-582cd02b-50dc-4f3b-9168-f1d1994c6303.png)

            
        - e.g. 2
            
            ![Untitled 10](https://user-images.githubusercontent.com/56019094/194996264-4b44bd54-e9ca-4062-aaec-e130b7f8b89a.png)

            

## Step5. 교배 및 돌연변이

- 교배(Crossover)
    - 앞 단계에서 선택된 한 쌍의 부모 염색체들이 서로 유전자 정보를 일부 교환하여 새로운 자식 염색체들을 생성
    - 교배 지점(Crossover Point)는 최소 1개부터 최대 n(전체 유전자(변수) 수)까지 가능
    - 교배 지점 1개 e.g.
        
        ![Untitled 11](https://user-images.githubusercontent.com/56019094/194996266-dfcdbd96-7a3f-4ad2-a6d2-4ef350578930.png)

        
    
    - 교배 지점 2개 e.g.
        
        ![Untitled 12](https://user-images.githubusercontent.com/56019094/194996267-552b19f3-bac4-4e4a-97b5-572b9203caa6.png)

        
    
    - 교배 지점 10개 e.g.
        
        ![Untitled 13](https://user-images.githubusercontent.com/56019094/194996269-9fe547ee-5ab2-41a4-a98e-d1f6473f5f05.png)

        

- 돌연변이 (Mutation)
    - 세대가 진화해가는 과정에서 다양성(Diversity)을 확보하기 위함
    - 특정 유전자의 정보를 낮은 확률로 반대 값으로 변환하는 과정을 통해 돌연변이 유도
    - 돌연변이를 통해 현재 해가 Local Optima에서 탈출할 수 있는 기회 제공
    - 너무 높은 돌연변이율은 유전 알고리즘의 수렴(Convergence) 속도를 늦춤
        
        → 주로 0.01 이하 값을 사용
        
        ![Untitled 14](https://user-images.githubusercontent.com/56019094/194996272-06a0285a-ccaa-4033-ac1d-7dd98f383107.png)

        

## Step6. 최적 변수 조합 선정

- 유전 알고리즘이 종료된 후 가장 높은 적합도 함수값을 갖는 염색체에 인코딩된 변수 조합을 선택
- 일반적으로 적합도 함수는 세대 초기에는 급격한 향상을 보이며, 세대가 진화할수록 향상률이 둔화됨

## Genetic Algorithm Hyperparameter

- Chronosome의 수 (= Population Size)
    - 한 세대에서 고려하는 변수 집합의 총 수
    - 크게 설정할수록 많은 범위를 탐색할 수 있으나, 그만큼 연산 자원을 많이 필요로 함
- 적합도 함수(Fitness function)
    - 현재의 변수 집합이 얼마나 우수한지를 평가할 수 있는 지표
- 교배 방식(Crossover mechanism)
    - 부모 염색체 간 유전자를 교환하는데 적용되는 방식
- 돌연변이율(Mutation rate)
    - 각 유전자마다 적용되는 돌연변이율
- 종료 조건 (Stopping criteria)
    - 적합도 함수가 일정 수준 이상 개선되지 않을 때
    - 최대 세대 수까지 진행되었을 때
    - …
