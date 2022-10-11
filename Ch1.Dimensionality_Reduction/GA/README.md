# Genetic Algorithm

![Untitled 1](https://user-images.githubusercontent.com/56019094/194995675-d0978f5f-aa85-4663-a2d1-ab6c763207d4.png

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

![Untitled](Genetic%20Algorithm%2000116ebd6d6145e8a5840389f047caaa/Untitled%201.png)

⇒ Feature Selection 용도로 Genetic Algoritm은 어떻게 사용될까?

![Untitled](Genetic%20Algorithm%2000116ebd6d6145e8a5840389f047caaa/Untitled%202.png)

## Step1. Initiation

- 염색체(Chromosome) 인코딩
    - 염색체 인코딩 방식은 Task에 따라 다름
    - Feature Selection을 위한 GA에서는 Binary Encoding 사용
        
        ![Untitled](Genetic%20Algorithm%2000116ebd6d6145e8a5840389f047caaa/Untitled%203.png)
        
- 세대 초기화
    - 염색체의 각 유전자(Gene)마다 난수 생성
    - 생성된 난수와 기준값(Cut-off) 비교 → Binary(0/1) 값으로 변환
    
    ![Population Size = 8, Cut-off = 0.5인 상황](Genetic%20Algorithm%2000116ebd6d6145e8a5840389f047caaa/Untitled%204.png)
    
    Population Size = 8, Cut-off = 0.5인 상황
    

## Step2. 모델 학습

- 각 염색체에 담긴 정보(해당 변수의 모델링 사용 유무)를 활용하여 염색체 수만큼 모델을 학습
    - e.g. 다중 선형 회귀분석
        
        ![Untitled](Genetic%20Algorithm%2000116ebd6d6145e8a5840389f047caaa/Untitled%205.png)
        
        ![Untitled](Genetic%20Algorithm%2000116ebd6d6145e8a5840389f047caaa/Untitled%206.png)
        
    

## Step3. 적합도 평가

- 각 염색체의 정보를 사용하여 학습된 모형(Step2를 통해 학습된 모델)의 적합도 평가
    - 적합도 함수: 염색체의 우열을 가릴 수 있는 정량적 지표
        - 적합도 함수 값이 높으수록 우수한 염색체 (우수한 변수 조합)
    - 적합도 함수가 가져야 하는 일반적인 조건
        - 두 염색체가 동일한 예측 성능을 나타낼 경우, 적은 수의 변수를 사용한 염색체를 선호
        - 두 염색체가 동일한 변수를 사용했을 경우, 우수한 예측 성능을 나타내는 염색체를 선호
    - 적합도 함수 예시: 선형 회귀 분석의 경우 Adjusted $R^2$, AIC, BIC 등이 사용될 수 있음

- Step2 예시에서 8개의 염색체에 대한 다중 선형 회귀분석의 Adj $R^2$ 산출
    
    ![Untitled](Genetic%20Algorithm%2000116ebd6d6145e8a5840389f047caaa/Untitled%207.png)
    
    - Rank는 Adj $R^2$ 기준 순위를 나타냄
    - Weight는 개별 염색체의 Adj $R^2$을 전체 염색체들의 Adj $R^2$ 합으로 나눈 값 → 추후 확률적 염색체 선택 기법에서 사용됨

## Step4. 부모 염색체 선택

- 현재의 세대에서 높은 적합도를 나타내는 염색체들을 부모로 선택하여 다음 세대의 염색체를 생성하는데 사용
    - 확정적 선택(Deterministic Selection)
        - 적합도 기준 상위 N%에 해당하는 염색체만 부모 염색체로 사용 가능
        - 하위 (100-N)%에 해당하는 염색체는 부모 염색체로 사용될 가능성 없음
        - e.g. 확정적 선택 예시 (50%)
            
            ![Untitled](Genetic%20Algorithm%2000116ebd6d6145e8a5840389f047caaa/Untitled%208.png)
            
            → 오직 1번, 2번, 4번, 8번 염색체만 다음 세대를 생성하는 부모 염색체의 역할을 수행할 수 있음
            
    - 확률적 선택(Probabilistic Selection)
        - 적합도 함수에 비례하는 Weight를 사용하여 부모 염색체를 선택
        - 모든 염색체가 부모 염색체로 선택될 가능성이 있으나 적합도 함수가 낮을수록 선택 가능성도 낮아짐
        - e.g. 1
            
            ![Untitled](Genetic%20Algorithm%2000116ebd6d6145e8a5840389f047caaa/Untitled%209.png)
            
        - e.g. 2
            
            ![Untitled](Genetic%20Algorithm%2000116ebd6d6145e8a5840389f047caaa/Untitled%2010.png)
            

## Step5. 교배 및 돌연변이

- 교배(Crossover)
    - 앞 단계에서 선택된 한 쌍의 부모 염색체들이 서로 유전자 정보를 일부 교환하여 새로운 자식 염색체들을 생성
    - 교배 지점(Crossover Point)는 최소 1개부터 최대 n(전체 유전자(변수) 수)까지 가능
    - 교배 지점 1개 e.g.
        
        ![Untitled](Genetic%20Algorithm%2000116ebd6d6145e8a5840389f047caaa/Untitled%2011.png)
        
    
    - 교배 지점 2개 e.g.
        
        ![Untitled](Genetic%20Algorithm%2000116ebd6d6145e8a5840389f047caaa/Untitled%2012.png)
        
    
    - 교배 지점 10개 e.g.
        
        ![Untitled](Genetic%20Algorithm%2000116ebd6d6145e8a5840389f047caaa/Untitled%2013.png)
        

- 돌연변이 (Mutation)
    - 세대가 진화해가는 과정에서 다양성(Diversity)을 확보하기 위함
    - 특정 유전자의 정보를 낮은 확률로 반대 값으로 변환하는 과정을 통해 돌연변이 유도
    - 돌연변이를 통해 현재 해가 Local Optima에서 탈출할 수 있는 기회 제공
    - 너무 높은 돌연변이율은 유전 알고리즘의 수렴(Convergence) 속도를 늦춤
        
        → 주로 0.01 이하 값을 사용
        
        ![Untitled](Genetic%20Algorithm%2000116ebd6d6145e8a5840389f047caaa/Untitled%2014.png)
        

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
