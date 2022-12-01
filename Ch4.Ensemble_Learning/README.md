# **Ensemble Learning**
- 본 Tutorial은 고려대학교 산업경영공학부 대학원 Business Analytics 강의 자료 및 외부 자료를 기반으로 작성되었습니다.
- 본 Tutorial에서는 Ensemble 기법 중 Bagging에 대한 설명과 코드를 함께 설명하였습니다.

---
## Overview
<p align = 'center'>
<img src = 'https://user-images.githubusercontent.com/56019094/205027310-d6295e3b-441c-42b9-8c61-840448446ec5.png' width = 80%>
</p>

🤔 혹시 구슬 수 실험을 아시나요?  
: 한 교수가 유리병에 유리 구슬 850개를 넣고 학생들에게 보여준 후, 구슬의 총 개수를 맞춰보라고 했습니다. 학생들의 답변의 평균값은 871개였습니다. 그러나 모든 학생들의 개별 답변 중에는 이보다 더 정확한 답변은 없었다고 합니다.  

그리고 오른쪽 그림에 나와있는 위키피디아는 누구나 자유롭게 쓸 수 있는 인터넷 백과사전입니다.  
이 두 개를 아우르는 키워드는 바로 ```집단 지성```이라고 할 수 있습니다. 때로는 다수의 평범한 사람들이 소수의 전문가보다 나은 판단을 하는 것처럼 여러 사람이 모여 협력하거나 경쟁해서 얻은 집단적인 지적 능력이 바로 집단 지성입니다.

머신 러닝에 바로 이 집단 지성이라는 개념을 적용한 것이 ```Ensemble```이라고 생각하시면 이해하기 쉬울 것 같습니다.

<p align = 'center'>
<img src = 'https://user-images.githubusercontent.com/56019094/205031265-5c2e9ccd-ba33-4643-9559-2b55f5545f03.png' width = 80%>
</p>


이미지 출처: https://towardsdatascience.com/four-ways-teams-win-on-kaggle-50e62acb87f4  

Kaggle에서 상위 랭크가 되기 위해서는 Ensemble 기법은 필수라고들 합니다. 비록 위 사진은 2020년 사진이지만, Kaggle의 120개 Competitions에서 Top-5 팀들이 사용한 머신러닝 소프트웨어를 보면 앙상블 기법인 LightGBM과 XGBoost가 상위권을 차지하고 있습니다. 눈에 띄는 것은 Keras나 PyTorch, TensorFlow와 같은 것은 특정 기법이 아닌 프레임워크인데, LightGBM과 XGBoost가 그보다 더 많이 쓰였다고 하네요.

---
이번 Tutorial에서는 Ensemble 기법의 대분류라고 할 수 있는 ```Bagging```과 ```Boosting``` 중 Bagging에 대해서 알아보겠습니다. 

<p align = 'center'>
<img src = 'https://user-images.githubusercontent.com/56019094/205032202-a906c8e0-5004-431e-b735-af2918e399e6.png' width = 80%>
</p>

```Bagging```은 위 그림과 같이 각 모델들이 병렬 처리가 가능하다는 장점이 있습니다. 반면 ```Boosting```은 Classifier1에서 나온 화살표가 오른쪽에 있는 Dataset Manipulator로 연결되어 있는 것처럼 각 모델들을 병렬로 처리하는 것이 불가능합니다. 대신, Bagging은 모델 복잡도가 높은 Artificial Neural Network나 K(Small)-NN을 Base Model로 사용합니다. 반면, Bagging은 Logistic Regression이나 K(Large)-NN을 사용합니다. 이제 그렇다면 Bagging에 대해서 조금 더 자세히 알아보겠습니다.

---
## Bagging
```Bagging```의 풀 네임은 바로 ```B```ootstrapp ```Agg```regat```ing```입니다.  
Bagging의 특징은 아래와 같습니다.
- Ensemble의 각 모델은 서로 다른 학습 데이터셋을 이용합니다.
- 각 데이터셋은 **복원 추출**을 통해 **원래** 데이터셋의 수만큼의 **크기**를 갖도록 샘플링해서 생성됩니다.
- 이렇게 생성된 개별 데이터셋을 **Bootstrap**(붓스트랩)이라고 합니다.
- Bagging의 특이한 점은 바로 **원하는 개수**만큼 Bootstrap을 만들 수 있다는 것입니다.

이제 그렇다면 코드 예시와 함께 살펴보겠습니다.  
튜토리얼에서는 머신러닝을 접해보신 분들이라면 다들 한 번은 사용해보셨을 Titanic 데이터셋을 이용하겠습니다.  

#### Bootstrap 생성 함수
---
```python
# Bootstrap Dataset 생성 함수 정의
def make_bs_data_list(train_X, train_y, bootstrap_num:int, verbose = 0) -> list:
    assert len(train_X) == len(train_y), "Your Train_X's length is not equal to Train_y's length"
    
    bootstrap_data_list = []
    
    total_index = set(range(len(train_X)))

    for i in range(bootstrap_num): # 데이터셋을 bootstrap_num만큼 반복 복원 추출 진행
        data_index = [data_index for data_index in range(train_X.shape[0])]
        random_data_index = np.random.choice(data_index, train_X.shape[0]) # 복원 추출
        
        valid_index = list(total_index.difference(set(random_data_index)))

        bs_train_X = train_X.iloc[random_data_index,]
        bs_train_y = train_y.iloc[random_data_index,]
        bs_valid_X = train_X.iloc[valid_index,]
        bs_valid_y = train_y.iloc[valid_index,]

        bootstrap_data_list.append([bs_train_X,bs_train_y, bs_valid_X, bs_valid_y])

        if (valid_index != 0) & (verbose == 1):
            print(f"{i+1}번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.")
            print(f'Validation Set으로 사용된 Instance의 수: {len(valid_index)} ({np.round(len(valid_index)/len(total_index)*100,2)}%) \n')

    return bootstrap_data_list
```
위 코드 결과 예시
```
1번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.
Validation Set으로 사용된 Instance의 수: 193 (36.14%) 

2번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.
Validation Set으로 사용된 Instance의 수: 189 (35.39%) 

3번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.
Validation Set으로 사용된 Instance의 수: 192 (35.96%) 

4번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.
Validation Set으로 사용된 Instance의 수: 197 (36.89%) 

5번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.
Validation Set으로 사용된 Instance의 수: 202 (37.83%) 

6번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.
Validation Set으로 사용된 Instance의 수: 194 (36.33%) 

7번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.
Validation Set으로 사용된 Instance의 수: 198 (37.08%) 

8번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.
Validation Set으로 사용된 Instance의 수: 191 (35.77%) 

9번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.
Validation Set으로 사용된 Instance의 수: 211 (39.51%) 
```

조금씩 코드를 나눠서 살펴보겠습니다.
```python
def make_bs_data_list(train_X, train_y, bootstrap_num:int, verbose = 0) -> list: 
    assert len(train_X) == len(train_y), "Your Train_X's length is not equal to Train_y's length"
    
    bootstrap_data_list = []
    
    total_index = set(range(len(train_X)))

```
- 위 함수에 들어가는 Input을 보니 Training Set의 X와 Training Set의 Y가 들어가는 것을 확인할 수 있습니다. 그리고 ```bootstrap_num```이라는 어떤 ```int(정수)```가 들어가는 것으로 보입니다. 위 설명에서 제가 "Bagging의 특이한 점은 바로 **원하는 개수**만큼 Bootstrap을 만들 수 있다는 것"이라고 했는데 바로 여기서 **원하는 개수**를 ```bootstrap_num```에 입력해주면 됩니다.


아래 코드가 바로 Bagging의 핵심인 Bootstrap을 만드는 파트입니다. 아래 코드를 다시 조금씩 나눠서 살펴보겠습니다.
```python
    for i in range(bootstrap_num): # 데이터셋을 bootstrap_num만큼 복원 추출 진행
        data_index = [data_index for data_index in range(train_X.shape[0])] # Training Set의 Instance의 index를 0부터 끝까지 리스트 형태로 저장
        random_data_index = np.random.choice(data_index, train_X.shape[0]) # 복원 추출
        
        valid_index = list(total_index.difference(set(random_data_index)))

        bs_train_X = train_X.iloc[random_data_index,]
        bs_train_y = train_y.iloc[random_data_index,]
        bs_valid_X = train_X.iloc[valid_index,]
        bs_valid_y = train_y.iloc[valid_index,]

        bootstrap_data_list.append([bs_train_X,bs_train_y, bs_valid_X, bs_valid_y])
```
<p align = 'center'>
<img src = "https://user-images.githubusercontent.com/56019094/205035487-a164b133-26fb-4bf9-a377-c5d9d3c6441b.png">
</p>

Original Dataset에는 원래 Index가 1부터 순서대로 존재합니다. 이 Index 값들을 저장하는 것이 바로 아래 Line입니다.
```python 
data_index = [data_index for data_index in range(train_X.shape[0])]
```
그리고 그림에 나와있듯이 각 Bootstrap의 X와 y를 보면 Original Dataset과 Index가 다른 것을 확인할 수 있습니다. 아래 Line을 통해 원래 Index에 Random Permutation을 적용해서 Index를 무작위로 섞어줍니다.
```python
random_data_index = np.random.choice(data_index, train_X.shape[0])
```
이제 무작위로 섞인 index를 통해 개별 Bootstrap을 생성하게 됩니다.
```python
valid_index = list(total_index.difference(set(random_data_index)))

bs_train_X = train_X.iloc[random_data_index,]
bs_train_y = train_y.iloc[random_data_index,]
bs_valid_X = train_X.iloc[valid_index,]
bs_valid_y = train_y.iloc[valid_index,]

bootstrap_data_list.append([bs_train_X,bs_train_y, bs_valid_X, bs_valid_y])
```
🙄 그런데 여기서 갑자기 Train이 아닌 ```valid```라는 단어가 등장합니다. 이것은 바로 Bootstrap을 생성할 때 **복원 추출**을 통해 **원래 데이터셋 크기**와 동일한 크기의 ```Bootstrap을 만드는데 선택되지 못한 Instance```들을 ```Validation Set```으로 사용하기 위함입니다.

<p align = 'center'>
<img src = "https://user-images.githubusercontent.com/56019094/205039243-a161c0f5-9415-49e5-8207-6334007a725d.png" >
</p>

- N: Data Instance의 수  
- 1/N: 각 Instance가 Bootstrap에 선택될 확률  

즉, $p$는 한 Instance가 하나의 Bootstrap에 단 한번도 선택되지 않을 확률입니다. Data Instance의 수 N을 무한대로 보내면 결국 $p$ = 0.368 즉, 한 Instance가 하나의 Bootstrap에 단 한번도 선택되지 않을 확률은 36.8%나 됩니다.
이러한 선택되지 않은 Data Instance들(=OOB Data)는 이후에 Validation Set으로 사용될 수 있습니다.  
이제 저기서 ```valid```가 왜 있는지 이해가 되실 것 같습니다. total_index(Training Set의 모든 Index)에서 random_data_index(Bootstrap으로 뽑힌 Instance들의 Index)의 차집합을 통해 **단 한번도 Bootstrap으로 선택되지 않은 Index**들을 찾았습니다.

그리고 사실 이미 코드 결과를 이전에 보셨겠지만, 실제로 
```
1번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.
Validation Set으로 사용된 Instance의 수: 193 (36.14%) 

2번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.
Validation Set으로 사용된 Instance의 수: 189 (35.39%) 

3번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.
Validation Set으로 사용된 Instance의 수: 192 (35.96%) 

4번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.
Validation Set으로 사용된 Instance의 수: 197 (36.89%) 

5번째 Bootstrap을 생성할 때 한 번도 선택되지 않은 Instance가 존재하여 Validation Set을 생성했습니다.
Validation Set으로 사용된 Instance의 수: 202 (37.83%) 
```
이와 같이 ```Bootstrap에 단 한번도 선택되지 않은 Instance의 수```가 이전 수식에서 보았던 이론적인 확률 36.8%와 유사함을 확인할 수 있습니다. 

이제 제가 원하는 만큼의 Bootstrap을 생성해보겠습니다. 9개의 Bootstrap을 생성해서 하나의 List인 ```bs_data_list```에 담아놓겠습니다.
```python
# bs_data_list = [[Bootstrap1's train_X, Bootstrap1's train_Y, Bootstrap1's valid_X, Bootstrap1's valid_Y], ...]
bs_data_list = make_bs_data_list(train_X, train_y, 9, verbose = 1) 
```

이제 그렇다면 Ensemble 모델에 사용할 Base Model을 정의해보겠습니다.  
이전에 제가 Bagging에는 모델 복잡도가 높은 Artificial Neural Network나 K(Small)-NN을 Base Model로 사용한다고 했습니다. 그런데 과연 **모델 복잡도가 낮은 모델인 Decision Tree나 Logistic Regression을 사용하면 최종 Ensemble 모델의 성능**이 어떻게 될지 궁금해서 직접 실험해보았습니다. 

```python
from sklearn.tree import DecisionTreeClassifier

def make_classifier(bs_data_list:list, bootstrap_num:int ,dt_max_depth:int, max_leaf_nodes:int):
    '''
    input: bs_data_list, bootstrap_num, dt_max_depth
    output: decision_trees

    boostrap_num: Should be same number(int) that used ini "make_bs_data_list" function
    '''
    models_list = []

    for idx, xy in enumerate(range(bootstrap_num)):
        train_x, train_y = bs_data_list[xy][0], bs_data_list[xy][1]
        globals()['{}_th_dt'.format(idx)] = DecisionTreeClassifier(max_depth = dt_max_depth, max_leaf_nodes= max_leaf_nodes).fit(train_x,train_y)
        models_list.append(globals()['{}_th_dt'.format(idx)])
    
    return models_list
```
그리고 한번 ```Base Model의 성능 차이가 존재하는 경우 최종 Ensemble 모델의 성능에도 어느 정도 차이가 있는지``` 확인해보기 위해 다음과 같은 코드를 사용했습니다. 즉, Decision Tree의 max_depth를 1, 2, 4, 8로 설정해서 [max_depth = 1인 Decision Tree가 Bootstrap 수만큼 있는 Group, max_depth = 2인 Decision Tree가 Bootstrap 수만큼 있는 Group, ...]과 같이 모델을 수립해서 Base Model Group"들"이라는 의미로 bm_groups라는 List 형태의 변수에 저장했습니다.
```python
# bm_groups = [max_depth = 1인 Base Model들로 구성된 첫번째 앙상블에 사용될 모델들, max_depth = 2인 Base Model들로 구성된 두번째 앙상블에 사용될 모델들, ... ]
bm_groups = []

for i in [1,2,4,8]:
    globals()['{}_dt_group'.format(i)] = make_classifier(bs_data_list, bootstrap_num = 9, dt_max_depth= i, max_leaf_nodes= 10)
    bm_groups.append(globals()['{}_dt_group'.format(i)])
```

로지스틱 회귀모형의 경우, 하이퍼 파라미터 C값에 따라 모델이 Training Set에 Over-fitting이 될 수도 혹은 Under-fitting이 될 수도 있습니다. 이번 튜토리얼에서는 Base Model의 성능 차이를 유발하기 위해 고의적으로 아래와 같이 C 값의 범위를 일반적으로 사용하는 경우보다 조금 더 넓은 범위로 로지스틱 회귀모형을 수립했습니다. 전체적인 코드 흐름은 Decision Tree와 동일합니다.
```python
from sklearn.linear_model import LogisticRegression

def make_classifier(bs_data_list:list, bootstrap_num:int, C:float):
    models_list = []

    for idx, xy in enumerate(range(bootstrap_num)):
        train_x, train_y = bs_data_list[xy][0], bs_data_list[xy][1]
        globals()['{}_th_dt'.format(idx)] = LogisticRegression(C = C).fit(train_x,train_y)
        models_list.append(globals()['{}_th_dt'.format(idx)])
    
    return models_list
```
```python
# 로지스틱 회귀분석 하이퍼 파라미터 중 C 값 변화
bm_groups = []

for c in [0.001, 0.01, 0.1, 1, 5, 10, 100]:
    globals()['{}_dt_group'.format(i)] = make_classifier(bs_data_list, bootstrap_num= 9, C = c)
    bm_groups.append(globals()['{}_dt_group'.format(i)])

```



<p align = 'center'>
<img src = "https://user-images.githubusercontent.com/56019094/205043363-9ec1a308-7f8d-4ba3-af2a-62ad17eceaf7.png" width = 100%>
</p>

위 그림에서 저희는 이제 Step2까지 완료했습니다. 즉,  
1. 원래 Training Set을 이용해서 원하는 개수만큼의 Bootstrap을 생성하고,
2. 각 Bootstrap을 이용해 Base Model을 각각 수립했습니다.  

이제 그렇다면 Step 3:Model forecasting(개별 Model의 예측값 구하기)와 Step 4: Result Aggregating(개별 Model의 예측값 합치기)가 남았습니다. 

Ensemble Model로 풀고자 하는 문제가 Classification(분류)인지, Regression(회귀)인지에 따라 개별 Model의 예측값을 합치는 방법은 Voting을 사용하거나 Averaging을 사용할 수 있습니다. 본 Tutorial에서는 **Titanic** Dataset을 이용해서 어떤 Instance(사람)이 ```Survived(생존) 여부```를 예측하고자 하기 때문에 ```Voting```을 이용하겠습니다.

<p align = 'center'>
<img src = "https://user-images.githubusercontent.com/56019094/205046452-30780e2c-eefa-4d1a-8040-bc82715cf0b3.png">
</p>

Voting에는 Majority Voting, Weighted Majority Voting 등의 방법이 있습니다. 본 Tutorial에서는 가장 직관적인 방법인 Majority Voting을 이용해서 개별 Model의 결과값을 합쳐 최종 예측값을 반환하겠습니다.  

Majority Voting 구현 코드는 매우 간단합니다.
```python
def majority_voting(bm_group:list, test_X):
    preds_per_bm_group = [bm.predict(test_X) for bm in bm_group]
    mv_pred = pd.DataFrame(preds_per_bm_group).T.mode(axis = 1)

    return mv_pred
```
- Input으로는 Base Model Group 즉, [Base Model 1, Base Model 2, ..., Base Model B]와 같이 Ensemble에 사용하게 될 Base Model들의 List를 받습니다.
- preds_per_bm_group은 Base Model Group 내의 각 Base Model을 이용해 Test Set에 대한 Inference 결과들을 합친 List입니다.
- 그리고 .mode() 메서드를 이용해 **최빈값**을 구해주면 이것이 바로 Majority Voting을 한 결과입니다.

이제 그렇다면 Decision Tree의 max_depth를 변경해가며 Base Model Group을 수립한 결과(F1 Score)를 보겠습니다. Red Line은 Base Model들을 합쳐 Majority Voting을 통해 얻은 최종 예측값을 이용한 즉, Ensemble의 성능입니다.

- Max_Depth = 2
<p align = 'center'>
<img src = "https://user-images.githubusercontent.com/56019094/205061175-cfe736ed-ac4c-4b2e-ab7e-67aaded023db.png" width = 80%>
</p>

- Max_Depth = 4
<p align = 'center'>
<img src = "https://user-images.githubusercontent.com/56019094/205061575-8c7d0d94-7672-4a17-8255-54b1e5e82323.png" width = 80%>
</p>

- Max_Depth = 8
<p align = 'center'>
<img src = "https://user-images.githubusercontent.com/56019094/205061702-f06f608a-86a7-46e1-840c-9aa8891527d2.png" width = 80%>
</p>

- Max_Depth = 16
<p align = 'center'>
<img src = "https://user-images.githubusercontent.com/56019094/205061803-02d45880-f2e8-4ee8-ba04-2f52656c33b3.png" width = 80%>
</p>


결과

- Max_Depth = 2인 경우 Ensemble 결과 성능이 Best Single Model보다 현저히 낮은 성능을 보이고 있습니다. 
- 반면 **Max_Depth = 4인 경우 Ensemble 결과 성능이 Best Single Model보다 높은 성능**을 보이며 Ensemble을 통해 집단 지성의 힘을 보여주었습니다. 
- 그리고 Max_Depth = 8, 16인 경우는 Ensemble 성능이 각각 Best Single Model보다 약간 낮은 성능을 보이고 있습니다.

결과 해석

- Max_Depth = 2인 경우 9개의 Base Model 중 8개의 성능이 동일한 것으로 유추해보아 ```Base Model의 다양성```이 매우 낮아 Ensemble의 효과가 거의 없었던 것으로 보입니다. 

- Base Model 자체의 성능 차이에 따른 최종 Ensemble 성능의 변화라는 관점에서 보면, Max_Depth = 2의 Base Model 성능이 가장 낮고 실제로 Ensemble 성능이 가장 낮은 것을 확인할 수 있습니다. 또한 Max_Depth = 8, 16은 Base Model의 성능이 거의 유사함에 따라 Ensemble 성능 또한 거의 동일한 것을 확인할 수 있습니다. 

- Random Forest의 경우 Bagging을 이용한 대표적인 앙상블 기법입니다. 그러나, 본 튜토리얼에서 Decision Tree에 단순히 Bagging을 한 것과 차이점은 바로 **개별 Decision Tree**를 수립하는 데 사용되는 **독립 변수가 다르다**는 것입니다. 즉, 본 튜토리얼에서 Decision Tree에 Bagging을 적용한 효과가 미미한 것은 Ensemble 모델의 효과에 중요한 요인인 **개별 Base Model의 "Diversity"**가 부족했던 것이 가장 큰 요인이라고 판단했습니다.
---
로지스틱 회귀 모형을 기반으로 하이퍼 파라미터 C를 변화시키며 추가 실험을 진행한 결과입니다.

- C = 0.00001
<p align = 'center'>
<img src = "https://user-images.githubusercontent.com/56019094/205069090-4cfcaf22-a55d-4903-914f-ae688c6c4448.png" width = 80%>
</p>

- C = 0.001
<p align = 'center'>
<img src = "https://user-images.githubusercontent.com/56019094/205069314-27d137fb-5fee-48e2-9646-9f10e7a1dc4d.png" width = 80%>
</p>


- C = 1
<p align = 'center'>
<img src = "https://user-images.githubusercontent.com/56019094/205069580-c838edc7-b1ab-4a65-adbe-6e33552d0014.png" width = 80%>
</p>

- C = 100, 10000의 경우 성능이 C = 1인 경우와 거의 동일했기에 결과에서 제외했습니다.  

|Model|C = 1|C=100|C=10000|
|-----|-----|-----|-------|
|BM1|0.7559|0.7619|0.7619|
|BM2|0.768|0.7559|0.7619|
|BM3|0.7656|0.7559|0.7656|
|BM4|0.7538|0.7538|0.7538|
|BM5|0.7559|0.75|0.75|
|BM6|0.7559|0.7442|0.7442|
|BM7|0.75|0.7442|0.7442|
|BM8|0.7559|0.7559|0.7559|
|BM9|0.7805|0.7667|0.7769|
|Ensemble|0.7559|0.75|0.75

결과  

- C = 0.00001인 경우 Base Model의 성능(F1 Score) 자체가 낮아 Ensemble 모델의 성능도 F1 Score 약 0.38로 낮은 성능을 보였습니다.  
- C = 0.001인 경우 Base Model의 성능이 C = 0.00001인 경우보다는 모두 높아 Ensemble 결과도 F1 Score 약 0.42로 조금 더 높은 성능을 보였습니다.
- C = 1인 경우 Base Model 자체의 성능이 약 0.75로 높아 당연히 Ensemble 모델의 F1 Score가 약 0.76으로 이전 두 경우보다 높은 성능을 보였습니다.  

결과 해석
- 해당 실험을 통해 Ensemble을 통해 높은 성능을 얻기 위해서는 우선적으로 Base Model 자체의 성능이 뒷받침 되어야 함을 직접 실험 결과로 다시 한 번 더 확인할 수 있었습니다. 집단 지성이라는 것도 사실은 어느 정도 이상의 지식을 가진 사람들이 모여야 좋은 결과를 야기할 수 있듯이, Ensemble에서도 개별 Base Model의 성능이 어느 정도 준수한 성능을 보여야 이러한 Model들을 Ensemble하는 것이 효과를 낼 수 있습니다. 

결론
- Ensemble을 통해 개별 Base Model보다 더 좋은 성능을 얻기 위해서는 ```개별 Base Model의 다양성```과 ```개별 Base Model의 성능```이 중요함을 본 튜토리얼을 통해 실험적으로 다시 한 번 더 깨달을 수 있었습니다. 또한 본 실험에서는 이 두 가지 요인으로 인해 Bagging의 효과가 저조했던 것인지, 아니면 ```모델 복잡도가 낮은 Base Model```을 사용해 이러한 결과가 야기된 것인지 정확한 비교 실험을 수행하지 못했다는 것이 주요 한계점입니다. 