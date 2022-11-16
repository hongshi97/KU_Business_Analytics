# **Anomaly Detection**
- About Anomaly Detection
- 본 Tutorial은 고려대학교 산업경영공학부 대학원 Business Analytics 강의 자료를 기반으로 작성되었습니다.

## **Contents**
1. [이론](#이론)
    - [Overview](#overview)
    - [Density-based Anomaly Detection](#density-based-anomaly-detection)
    - [Model-based Anomaly Detection](#model-based-anomaly-detection)

2. [코딩 실습](#코딩-실습)
    - [실험 1](#실험-1-shallow-autoencoder-vs-deep-autoencoder)
    - [실험 2](#실험-2-autoencoder의-bottleneck-size-변화에-따른-autoencoder의-성능-변화-및-anomaly-detection-성능-변화-측정)
    - [실험 3](#실험-3-what-if-encoder의-layer-수--decoder의-layer-수--2라면-그리고-반대의-경우에는)
---
# 이론

## Overview
Q1. 🧐 이상치 데이터(Anomaly/Novelty Data)란?  
A1. 원인 관점: 다른 관측치로부터 너무 많이 벗어나서 **다른 메커니즘에 의해 생성된 것**이라 의심을 받는 관측치 (Hawkins, 1980)  
A2. 결과 관점: **실제 확률 밀도가 매우 낮은** Instance들 (Harmeling et al, 2006)

Q2. 😅 이상치 데이터 = 노이즈?  
A. Noise는 측정 과정에서의 무작위성에 기반  
 이상치 데이터는 정상적인 데이터를 생성하는 메커니즘을 위반하여 생성된 데이터  
  -> 데이터의 수는 적지만 중요한 데이터!

<p align="center">
<img src = "https://user-images.githubusercontent.com/56019094/201920223-0af3e041-17af-47e1-94d9-525aa9898628.png" width = 400>
</p>
<p align = "center">이미지 출처: https://link.springer.com/article/10.1007/s10278-020-00413-2 </p>

Q3. 😉 Anomaly Detection이 어디에 적용되나요?  
A.  제조업 공정에서의 불량 탐지, 신용카드 사기 거래 탐지, 아이디 도용 탐지 등 다양한 분야에서 활용  

Q4. 🥲 Anomaly(Novelty) Detection = Classification?  
A.  Anomaly Detection은 일종의 Classification이라고 볼 수 있습니다. Classification 중에서도 주어진 Data가 특정 Class Label에 대한 Data만 많은 경우(e.g., 1%:99%)


<p align = 'center'>
<img src ="https://user-images.githubusercontent.com/56019094/201925033-192ab0cb-32d2-4dab-8798-25316e984643.png">
</p>

위와 같이 Anomaly Detection에서는 양품 즉, 정상의 범주를 어느 정도로 정하느냐에 따라 모델의 예측 결과가 달라지게 됩니다. 이와 관련된 개념으로 **일반화(Generalization)**와 **특수화(Specialization)**이라는 개념이 존재합니다.

- 일반화: 주어진 데이터로부터 정상 범주의 개념을 확장해 가는 것 -> 일반화에 치중할 경우 이상치 데이터 판별이 어려움  
- 특수화: 주어진 데이터로부터 정상 범주의 개념을 좁혀가는 것 -> 특수화에 치중할 경우 과적합의 위험 (빈번한 False Alarm)에 빠질 수 있음

    <p align = 'center'>
    <img src = "https://user-images.githubusercontent.com/56019094/201925558-e50ce14f-dd40-4e22-8f47-4cdf8a3a38eb.png">
    </p>

1. 만약 일반화의 정도가 심해 정상 범주의 개념을 너무 넓게 잡게 되면 정상(사과)이 아닌 "수박"도 정상(사과)라고 판단하게 됩니다. 
2. 반대로, 만약 특수화의 정도가 심해 정상 범주의 개념을 너무 좁게 잡게 되면 정상(사과)인 "청사과"도 이상치(정상이 아니다)라고 판단하게 됩니다.

다시 Q4로 돌아가서 Anomaly Detection과 Classification에 대해 살펴보자면 다음과 같은 그림을 통해 쉽게 이해할 수 있을 것입니다.
<p align = 'center'>
<img src = "https://user-images.githubusercontent.com/56019094/201926047-f15dfd88-65fb-45ec-8bcb-93e24712fd52.png">
</p>
위 그림과 같이 Classification의 경우 Training Data에 정상(X+)인 데이터와 비정상(X-)인 데이터가 같이 사용됩니다. 반면, Anomaly Detection이 사용되는 경우에는 이미 비정상 데이터보다 정상 데이터가 매우 많은 상황이기에 Training에는 정상 데이터만 사용합니다. 이를 통해, 정상 데이터들만의 패턴을 학습해서 새로운 데이터가 들어오면 자신이 학습했던 패턴과 유사한지 아닌지를 판별해 정상/비정상 여부를 판단하게 됩니다.

## Density based Anomaly Detection
- 주어진 데이터를 바탕으로 각 개체들이 생성될 확률을 추정
- 새로운 데이터가 생성될 확률이 낮을 경우 이상치로 판단!

<p align = 'center'>
<img src = "https://user-images.githubusercontent.com/56019094/201927283-a27d0c13-588b-4a98-861e-63a262f4d871.png" width = 90%>
<img src = "https://user-images.githubusercontent.com/56019094/201927462-7f6ea9ba-f3b9-4a46-8b35-ee2a3269ce48.png" width = 90%>
</p>

- Gaussian Density-based Anomaly Detection
    - 가정: 모든 데이터가 하나의 가우시안(정규) 분포로부터 생성됨
    - 학습: 주어진 정상(Normal) 데이터들을 통해 가우시안 분포의 평균 벡터와 공분산 행렬을 추정
    - 추론: 새로운 데이터에 대하여 생성 확률을 구하고 해당 확률이 낮을수록 이상치에 가까운 것으로 판정
    - 장점 1. 추정이 간단하며 학습 시간이 짧음  
    - 장점 2. 적절한 기준치(Cut-off)를 분포로부터 정할 수 있음
    - 장점 3. 각 변수의 측정 단위에 영향을 받지 않음   
    <p align = 'center'>
    <img src = "https://user-images.githubusercontent.com/56019094/201928001-632bdf9f-7f87-42c3-b8e4-0e78189feb14.png">
    </p>

<br/>  

- Mixture of Gaussian(MoG) Density Estimation
    - 데이터는 여러 개의 가우시안 분포의 혼합으로 이루어져 있음을 허용
    - 아래 가우시안 분포(빨간색 함수 제외한 것)들의 선형 결합으로 전체 데이터의 분포를 표현
    <p align = 'center'>
    <img src = 'https://user-images.githubusercontent.com/56019094/201928561-e7f3bffc-7f34-4b6f-91e8-4f0888391de2.png' width = 80%>
    </p>

    - 일반 Gaussian Density Estimation에서는 모든 데이터가 하나의 가우시안 분포로부터 생성되었다고 가정하기에 평균과 공분산이라는 단 두 개의 미지수만 구하면 됐습니다.
    - 그러나, MoG에서는 위 그림과 같이 f(x)라는 우리가 알고자 하는 함수가 k개의 가우시간 분포의 선형 결합으로 이루어져 있다고 가정하기에, k개의 가우시안 분포에 대해서 평균, 공분산, 그리고 각 가우시안 분포에 대한 가중치 w라는 세 개의 미지수를 구해야 합니다.
    - 따라서, MoG에서는 3*k개의 미지수를 구해야 합니다.
    - 여기서 끝이면 좋겠지만, 사실은 이 k가 몇이면 좋을지도 Train Dataset을 통해서 탐색해야 합니다.

<br/>

- Parzen Window Density Estimation
    - Kernel-density Estimation
        - 이전에 봤던 Gaussian Density Estimation과 MoG는 데이터가 특정 분포로부터 생성되었다는 가정을 하고 접근하는 방식이었습니다.
        - 여기서 보게 될 Kernel-density Estimation은 데이터가 특정 분포로부터 생성되었다는 가정을 하지 않고, 개별 데이터 자체로부터 확률들을 추정하겠다는 접근 방식을 취합니다.

        <p align = 'center'>
        <img src = "https://user-images.githubusercontent.com/56019094/201929688-9db6edfc-93fe-4857-a414-c8be15d3342c.png" width = 80%>
        </p>

        
        이미지 출처:  https://sebastianraschka.com/Articles/2014_kernel_density_est.html 

    - Parzen Window Density Estimation
        - k개의 객체를 포함하는 영역 x를 (무게) 중심으로 하며, 각 면의 길이가 h인 Hypercube로 정의: Hypercube의 볼륨 V는 $h^d$로 정의됨(d = 차원 수) -> V = $h^d$
        - Kernel Function을 다음과 같이 정의
        <p align = 'center'>
        <img src = "https://user-images.githubusercontent.com/56019094/201930589-1a674945-c736-42a8-a8da-d4eeaa289e63.png" width = 80%>
        </p>

        - 예시
        <p align = 'center'>
        <img src = "https://user-images.githubusercontent.com/56019094/201930784-5197dfc1-7974-4943-9f7f-51e9751a3a66.png" width = 60%>
        </p>

        - 기존 Kernel Function은 불연속적임. 즉, Hypercube내에 있는 객체들에 대해서는 모두 동일한 가중치를 적용한다.  
        -> Continuous한 분포를 사용하자!
        <p align = 'center'>
        <img src = "https://user-images.githubusercontent.com/56019094/201931306-0cae4b74-cfb1-4a00-8857-ec8f314328e5.png" width = 80%>
        </p>
        
        - Smoothing Parameter (Bandwidth)h
            - (남색, 0.3)h가 크면 Density Distribution이 과도하게 Smoothing됨
            - (주황색, 0.05)h가 작으면 Density Distribution의 형태가 Spiky해짐
            <p align = 'center'>
            <img src = "https://user-images.githubusercontent.com/56019094/201943177-66390a06-efb8-4241-8be7-f63faf8d986c.png">
            </p>
            <p align = 'center'>
            이미지 출처: https://en.wikipedia.org/wiki/Kernel_density_estimation
            </p>
            - Smoothing Parameter h는 EM 알고리즘을 통해 최적화 가능

<br/>

- Local Outlier Factor (LOF)
    - 목적: 이상치 스코어를 산출할 때, 주변부 데이터의 밀도를 고려하자
    <p align = 'center'>
    <img src = "https://user-images.githubusercontent.com/56019094/201943818-158836f7-72aa-4bde-acee-e9be197f44b2.png" width = 90%>
    </p>
    - LOF 알고리즘에는 5가지 Definition이 사용됨  
    
    <br/>
    
    1. k-distance of an object p: 임의의 양의 정수 k에 대해서 k-distance of object p (=k-distance(p))는 다음 두 조건을 만족하는 데이터셋 D의 두 점 p와 o의 거리 d(p,o)로 정의됨  

        - D에 속하는 개체 중 p를 제외하고 **최소한 k** 개의 개체 o'에 대해서 d(p,o') <= d(p,o)를 만족
        - D에 속하는 개체 중 p를 제외하고 **최대 k-1**개의 개체 o'에 대해서 d(p,o') < d(p,o)를 만족  
        즉, 동렬을 고려한 k번째 이웃까지의 거리로 생각할 수 있음
    2.  k-distance neighborhood of an object p: 개체 p의 k-distance가 Definition 1과 같이 주어질 때, k-distance neighborhood of p는 p에서부터 k-distance보다 멀지 않은 거리에 있는 모든 개체들의 집합을 의미

    3. Reachability Distance
    <p align = 'center'>
    <img src = "https://user-images.githubusercontent.com/56019094/201944909-d207b06f-b6af-4852-898c-8223def082d4.png" width = 90%>
    </p>

    4. local reachability density of an object p
    <p align = 'center'>
    <img src = "https://user-images.githubusercontent.com/56019094/201945584-372314bc-1c2d-4e2a-9858-ab125f34a07f.png" width = 90%>
    </p>
    
    - Case 1: 개체 p의 주변에 높은 밀도로 다른 개체들이 존재하는 경우 위 식의 분모가 작은 값을 갖게 되어 lrd는 큰 값을 갖게 됨
    - Case 2: 개체 p가 두 개의 높은 밀도를 갖는 군집 사이의 밀도가 낮은 공간에 위치하게 되면 위 식의 분모는 커지게 되어 lrd는 작은 값을 갖게 됨
    <p align = 'center'>
    <img src = "https://user-images.githubusercontent.com/56019094/201945935-35866454-c374-4021-9804-369189137a9f.png" width = 80%>
    </p>

    5. local outlier factor of an object p
    <p align = 'center'>
    <img src = "https://user-images.githubusercontent.com/56019094/201946264-b78e2327-59ab-46a2-9d9e-b4335becf80c.png" width = 80%>
    </p>
    
    - LOF의 최종 결과물은 각 개체들의 주변 밀도를 고려한 이상치 스코어
    <p align = 'center'>
    <img src = "https://user-images.githubusercontent.com/56019094/201946474-cf3595ce-ce96-4cf8-9633-f79b92746267.png" width = 85%>
    </p>

    <p align = 'center'>
    이미지 출처: https://godongyoung.github.io/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/2019/03/11/Local-Outlier-Factor(LOF).html
    </p>

</br>

## Model based Anomaly Detection
- Auto-Encoder

    <p align = 'center'>
    <img src = "https://user-images.githubusercontent.com/56019094/201947375-89ba1a2a-64ba-4d94-9d88-965c153e8a45.png" width = 90%>
    </p>

    - Input과 Output이 동일한 Neural Network
    - Latent Space는 원본 데이터보다 작아야 함
    

    - Auto-Encoder의 활용
        - 차원 축소
        1. Auto-Encoder를 학습시킨 후, Latent Vector를 다른 모델의 Input으로 사용:  Auto-Encoder의 Latent Vector가 데이터의 중요한 특징들을 잘 반영하고 있다는 특성으로 인해 가능
        <p align = 'center'>
        <img src = "https://user-images.githubusercontent.com/56019094/201947866-4efc0540-0f25-4fe3-aed2-4dfe5caeea79.png" width = 80%>
        </p>

        - 이상치 탐지
        1. Input과 Output간의 차이를 활용: Input과 Output간의 차이가 크면 Anomaly 데이터, 차이가 작으면 Normal 데이터
        <p align = 'center'>
        <img src = "https://user-images.githubusercontent.com/56019094/201948189-a142af9d-28df-4466-8dda-cb981117d404.png" width = 85%>
        </p>

    - Auto-Encoder의 단점
        - 입력 데이터에 변형이 약간이라도 있으면 모델이 민감하게 반응
        => Input Data에 Noise가 있어도 원본 데이터를 복원할 수 있는 좀 더 Robust한 Auto-Encoder를 만들자!   
        => "Denoising Auto-Encoder"

    - Denosing Auto-Encoder
        - 모델에는 Noise가 추가된 Input을 사용
        - Reconstructioin Error: Output과 **원본** Input 간의 차이를 Loss로 활용
        - 주로 Noise로는 Random Gaussian Noise를 사용
        <p align = 'center'>
        <img src = "https://user-images.githubusercontent.com/56019094/201948897-5a7d1d4b-38f6-4b98-a1b4-b87019bb6df1.png" width = 85%>
        </p>

        <p align = 'center'>
        이미지 출처: https://junstar92.tistory.com/158
        </p>

- Support Vector-based Anomaly Detection
    - 1-SVM: One-class SVM
    - SVDD: Support Vector Data Description
    
    <p align = 'center'>
    <img src = "https://user-images.githubusercontent.com/56019094/201949320-ae13e84c-acad-4c12-a489-9937fbb19c07.png" width = 80%>
    </p>

- Isolation Forests
    - Motivation: Few & Different
    - Few: 이상치 데이터는 Overview에서 봤듯이 정상 데이터에 비해 수가 적음
    - DIfferent: 이상치 데이터는 정상 데이터와는 특징이 다름

    <p align = 'center'>
    <img src = "https://user-images.githubusercontent.com/56019094/201949696-6c04da67-2546-474e-abaf-2ced6beeeb05.png" width = 85%>
    </p>

    <p align = 'center'>
    이미지 출처: https://www.tibco.com/reference-center/what-is-anomaly-detection
    </p>

    - 하나의 개체를 고립시키는 Tree를 생성
    - 정상 데이터는 서로 가까이 위치해 있기 때문에 정상 데이터 개체 하나를 고립시키기 위해서는 많은 Split이 필요
    - 이상치 데이터는 정상 데이터와 특징이 달라 멀리 위치해 있기 때문에 이상치 데이터 개체 하나를 고립시키기 위해서는 상대적으로 적은 Split이 필요
    
    <p align = 'center'>
    <img src = "https://user-images.githubusercontent.com/56019094/201950194-26823da6-6997-4f4c-881d-35f836f3e681.png" width = 90%>
    </p>

    - Isolation Forests는 사실 "학습"을 한다기 보다는 데이터가 들어오면 그 때마다 새롭게 알고리즘이 수행되어 결과를 도출함. 즉, K-NN처럼 **Lazy Learning**에 해당
    - Random으로 데이터를 Split하여 관측치를 고립시킴
    - 변수가 많은 데이터에서도 효율적으로 작동함

    - Isolation Forests 과정
        1. 전체 데이터에서 n개의 데이터 집합 X 샘플링
        2. 랜덤하게 선택된 관측치에 대해 임의의 변수(Spliting Variable)와 분할점(Spliting Point)을 사용하여 아래 조건을 만족할 때까지 이진 분할 진행
            - Tree가 사전에 설정한 최대 깊이에 도달
            - 영역 X 안에 단 하나의 객체만 존재
            - 영역 X 안에 존재하는 객체들이 모두 같은 입력 변수 값을 가짐
        3. 1.2.와 동일한 과정으로 여러 개의 Isolation Tree 생성
        4. Isolation Tree마다 각 관측치의 Path Length 저장
        5. 각 관측치의 평균 Path Length를 기반으로 Anomaly Score 계산 및 이상치 판별

        <p align = 'center'>
        <img src = "https://user-images.githubusercontent.com/56019094/201951689-495f872f-1004-4e1d-aa2c-62ad878380f0.png" width = 80%>
        </p>

</br>  
</br>
</br>

# 코딩 실습
- 주제: AutoEncoder를 이용한 Anomaly Detection
- 사용 데이터셋: MNIST 
- 데이터셋 선정 이유
    - 튜토리얼이라는 취지에 맞추어 AutoEncoder의 복원 결과를 시각화하기 좋은 데이터셋이라 판단함
    - Anomaly Detection에 MNIST 데이터셋을 사용하는 경우 0~9 숫자 중 하나를 Anomaly로 가정하고 수행하는 경우는 존재하나, 본 코딩 실습에서는 "가", "나", "다", "A", "B", "C"와 같은 텍스트를 28*28 이미지로 생성해서 Anomaly로 사용함
    - 즉, 본 코딩 실습에서 학습을 통해 수립된 AutoEncoder는 0~9라는 숫자(Normal)만 학습했기 때문에 한글이나 영어와 같은 텍스트(Anomaly)가 Input으로 들어오면 Anomaly로 판단하게 될 것
- 실험 내용
    1) Shallow AutoEncoder vs Deep AutoEncoder
    2) AutoEncoder의 Bottleneck Size 변화에 따른 AutoEncoder의 성능 변화 및 Anomaly Detection 성능 변화 측정
    3) 🫥 What if? Encoder의 Layer 수 = Decoder의 Layer 수 * 2라면?? 그리고 반대의 경우에는?

```python
# 패키지 및 메소드 불러오기
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

from torchvision import datasets, transforms
from copy import deepcopy
```


</br>

```python
# MNIST 데이터셋을 불러올 때 Training Data로 사용할 것인지, flatten을 할 것인지 여부를 argument로 받는 함수 정의
def load_mnist(is_train=True, flatten=True): # 본 실습에서 구현하게 될 AutoEncoder는 1차원 Vector 형태를 입력으로 받기 때문에 flatten = True

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([ 
            transforms.ToTensor(), # Tensor 형태로 데이터를 받겠다는 의미입니다 :)
        ]),
    )

    # Scaling: MNIST 데이터셋은 기존에 각 Element 값이 0~255라 /255.가 Min-Max Scaling 역할을 수행함
    x = dataset.data.float() / 255. 
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1) # (28,28) -> (28*28,) 형태로 Flatten해서 Neural Network Input으로 넣어주기 위함

    return x, y
```

</br>


```python
# MNIST 데이터 불러오기 및 Train/Validation/Test Set 분할
train_x, train_y = load_mnist(flatten=True) 
test_x, test_y = load_mnist(is_train=False, flatten=True) # Test Set이니까 is_train = False
test_x, test_y = test_x, test_y


train_cnt = int(train_x.size(0) * 0.8)
valid_cnt = train_x.size(0) - train_cnt

# Shuffle dataset to split into train/valid set.
indices = torch.randperm(train_x.size(0))
train_x, valid_x = torch.index_select(
    train_x,
    dim=0,
    index=indices
).split([train_cnt, valid_cnt], dim=0)
train_x, valid_x = train_x, valid_x

train_y, valid_y = torch.index_select(
    train_y,
    dim=0,
    index=indices
).split([train_cnt, valid_cnt], dim=0)
train_y, valid_y = train_y, valid_y

print("Train:", train_x.shape, train_y.shape)
print("Valid:", valid_x.shape, valid_y.shape)
print("Test:", test_x.shape, test_y.shape)
```
**위 코드 결과**  
Train: torch.Size([48000, 784]) torch.Size([48000])  
Valid: torch.Size([12000, 784]) torch.Size([12000])  
Test: torch.Size([10000, 784]) torch.Size([10000])

- 위 코드를 보면 load_mnist()를 통해 각 숫자 이미지의 픽셀값들이 저장되어 있는 x와 해당 이미지의 숫자가 몇인지를 의미하는 Label인 y를 불러왔습니다. 그런데 아래 그림과 같이 AutoEncoder의 모델 구조를 생각해보면, 사실 x를 입력으로 받아서 $\hat{x}$를 반환하기 때문에 y는 받아올 필요가 없습니다.  
(튜토리얼이기 때문에 MNIST 데이터셋을 다운받아서 추후 다른 실험을 해보고 싶으신 분들이 있으실까봐 일단 기재해두었습니다.)
<p align = 'center'>
    <img src = "https://user-images.githubusercontent.com/56019094/201947375-89ba1a2a-64ba-4d94-9d88-965c153e8a45.png" width = 90%>
    </p>

</br>

시각화를 통해 제대로 데이터를 가져왔는지 확인해보겠습니다.
```python
# MNIST Data Instance 시각화 함수 정의 
def show_image(x):
    if x.dim() == 1: # 만약 x의 dim이 1이라면 (x를 flatten() 해서 1차원 벡터 형태로 만든 경우)
        x = x.cpu()
        x = x.view(int(x.size(0) ** .5), -1) # dim = 2로 변환
    
    plt.figure(figsize=(1,1))
    plt.imshow(x, cmap='gray') # Grayscale 이미지
    plt.show()

show_image(train_x[0].flatten()) # train_x 데이터 중 첫번째 데이터를 가져와서 시각화
```
**위 코드 결과**
<p align = 'left'>
<img src = 'https://user-images.githubusercontent.com/56019094/202084631-a65dcfa5-3fbd-433e-9c11-f097c2b449b9.png'>
</p>

시각화는 성공적으로 되었습니다. 숫자 4라는 것도 충분히 알 수 있습니다. 다만, MNIST Dataset은 사람들의 손글씨로 만들어졌기 때문에 이처럼 동일한 숫자라도 예상과 달리(?) 모양이 특이한 경우가 종종 있습니다.

## 실험 1. Shallow AutoEncoder vs Deep AutoEncoder
그렇다면 이제 AutoEncoder를 구현해보겠습니다.
AutoEncoder를 처음 들으시는 분들이라면 모델 구조가 복잡할 것이라 생각할 수도 있지만 위에서 본 모델 구조와 같이 최근에 나온 방법론들에 비해 구조가 매우 간단한 편입니다. 아래 Shallow_AE를 보시면 아마 쉽게 이해가 되실 것입니다.  
추가적으로, 텐서 형태에 대한 설명을 주석으로 달아놨으니 참고해주시면 좋을 거 같습니다! 저 또한 아직도 새로운 방법론을 사용하거나 기존 모델을 변형할 때면 Tensor 형태가 헷갈릴 때가 많았어서 작은 도움이라도 되었으면 하는 마음으로 추가했습니다.
```python
# 비교적 모델의 Depth가 얕은 AutoEncoder 구현
class Shallow_AE(nn.Module):
    
    def __init__(self, btl_size=2):   # 입력 인자로 bottle neck size를 받음
        self.btl_size = btl_size
        
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 100),  # MNIST input이 28*28 형태라서 28*28을 flatten한 784를 입력으로 받음  즉, x는 784차원에 해당
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, btl_size),    # 주의! Encoder와 Decoder의 마지막 Layer는 그냥 Linear Layer임
                                        # 마지막에 Activaiton Function(ReLU 등)이나 BatchNormalization이 없음!
        )
        self.decoder = nn.Sequential(
            nn.Linear(btl_size, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 28 * 28),
        )        
        # 현재 이 코드에서는 encoder와 decoder를 대칭되게 짰는데, 꼭 대칭일 필요는 없음.
        # 예를 들어서 encoder의 레이어를 더 deep하게 만들고, decoder를 더 얕게 만들수도 있음
        
    def forward(self, x):
        z = self.encoder(x)  # |x| = (batch_size, 784), |z| = (batch_size, btl_size)
        y = self.decoder(z)  # |y| = |x|
        
        return y
```
그렇다면 이제 위 AutoEncoder보다 Depth가 깊은 AutoEncoder를 정의해보겠습니다.
```python
class Deep_AE(nn.Module):
    
    def __init__(self, btl_size=2):   
        self.btl_size = btl_size
        
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 300),  
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, btl_size),   
        )
        self.decoder = nn.Sequential(
            nn.Linear(btl_size, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 28 * 28),
        )
        
    def forward(self, x):
        z = self.encoder(x) 
        y = self.decoder(z) 
        
        return y
```
Deep AutoEncoder는 Shallow AutoEncoder보다 Layer의 수가 2배 이상 많은 상황입니다. 그렇다면 이제 이 두 가지 모델을 학습시켜서 성능을 확인해보겠습니다.  
성능 비교 전에 그러면 우선 모델을 학습시킬 클래스나 함수를 만들어 보겠습니다.

```python
# Model 학습에 사용할 Trainer 클래스 정의
class Trainer():

    def __init__(self, model, optimizer, crit, batch_size = 128, n_epochs = 50, verbose = 1):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.crit = crit
        self.batch_size = batch_size
        self.n_epochs = n_epochs 
        self.verbose = verbose

        super().__init__()

    def _train(self, x, y): # 사용자 정의 함수 앞에 _를 붙인 이유: Trainer() 객체를 생성한 후에 이 함수에 접근하는 것을 방지하기 위함. (객체._train() 이라고 코드 입력 시 접근 가능하긴 하지만, 일차적으로 객체. 했을 떄 GUI 상에 표시되지 않음)
        self.model.train() # 모델 Train() 모드 시작

        # Data 셔플해주기
        # 본 튜토리얼에서는 AutoEncoder를 사용하기 때문에 y(Class Label)은 사용되지 않지만 튜토리얼이라는 측면에서 함께 작성함
        indices = torch.randperm(x.size(0), device=x.device)
        x = torch.index_select(x, dim=0, index=indices).split(self.batch_size, dim=0)  # Shuffle 후, batch_size로 기존 x를 나눠줍니다. 즉, x = [batch_1, batch_2, batch_3, ...]와 같이 리스트 안에 batch_x가 Element로 들어가게 됩니다.
        y = torch.index_select(y, dim=0, index=indices).split(self.batch_size, dim=0)

        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            # 모델의 Gradient를 초기화
            self.optimizer.zero_grad()
            loss_i.backward()  # Gradient 계산

            self.optimizer.step() # .backward()를 통해 산출된 Gradient를 이용해 파라미터 업데이트

            if self.verbose >= 2: # Iteration 별 Train Loss를 확인하고자 한다면 verbose에 2 이상의 값을 입력 (훈련 과정을 조금 더 자세히 보고 싶으신 분들을 위해 추가하긴 했지만, Epoch마다 Loss 값을 출력하게 만들어놨기 때문에 verbose를 사용하는 것은 개인적으로 비추천합니다. 정말 Verbose(수다스러운)해집니다.)
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            # OOM 문제를 방지하기 위한 노력의 일환으로 loss를 float() 데이터 타입으로 변경. float()을 하지 않을 경우 loss에 gradient 정보가 남아 있어서 현재 단계에서 필요하지 않은 메모리를 사용하게 됩니다. (동일한 목적으로 종종 loss.detach()를 사용하기도 합니다.)
            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validate(self, x, y):
        self.model.eval() # 모델 Evaluatioin 모드 시작

        # Validation Phase와 Test Phase에서는 Gradient Descent를 통한 파라미터 업데이트가 이루어지지 않기 때문에 gradient를 사용하지 않겠다는 의미로 with torch.no_grad()를 사용합니다.
        with torch.no_grad():
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices).split(self.batch_size, dim=0)
            y = torch.index_select(y, dim=0, index=indices).split(self.batch_size, dim=0)

            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if self.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i)

            return total_loss / len(x)

    def train(self, train_data, valid_data):
        lowest_loss = np.inf   # 초기 최저 Loss 값을 무한대로 설정 -> 추후 모델이 Train을 진행하면서 Loss 값이 낮아지게 되면서 lowset_loss 값이 갱신됩니다.
        best_model = None  # best_model이 아직은 train을 시작하기 전이므로 None으로 선언합니다.

        for epoch_index in range(self.n_epochs): # 각 에포크 별 학습 시작
            train_loss = self._train(train_data[0], train_data[1])
            valid_loss = self._validate(valid_data[0], valid_data[1])

            # Best Model의 파라미터를 저장하기 위해 deepcopy() 사용 
            # 참고: 본 튜토리얼에서는 Early Stopping은 사용하지 않음. 대신 n_epochs 동안 Training을 진행하면서 Valid_Loss가 가장 낮았던 model을 최종 model로 사용함
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            if epoch_index % 20 == 0: # Epoch 5번마다 Loss 출력

                print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                    epoch_index + 1,
                    self.n_epochs,
                    train_loss,
                    valid_loss,
                    lowest_loss,
                ))

        # Restore to best model.
        self.model.load_state_dict(best_model)
```
이제 모델을 훈련시키기 위한 Trainer 클래스 정의를 마쳤으니 AutoEncoder를 학습시키고 성능을 비교해보겠습니다.
```python
# Shallow AutoEncoder Train 
# Bottleneck Size = 5, Batch_size = 512, n_epochs = 100으로 동일하게 맞춰줌
shallow_model = Shallow_AE(btl_size=5)
shallow_model = shallow_model
optimizer = optim.Adam(shallow_model.parameters())
crit = nn.MSELoss() 

trainer = Trainer(shallow_model, optimizer, crit, batch_size = 512, n_epochs = 100, verbose = 1)
trainer.train((train_x.to(device), train_x.to(device)), (valid_x.to(device),valid_x.to(device)))
```
```python
# Deep AutoEncoder Train
Deep_model = Deep_AE(btl_size=5)
Deep_model = Deep_model
optimizer = optim.Adam(Deep_model.parameters())
crit = nn.MSELoss()

trainer = Trainer(Deep_model, optimizer, crit, batch_size = 512, n_epochs = 100, verbose = 1)
trainer.train((train_x.to(device), train_x.to(device)), (valid_x.to(device),valid_x.to(device)))
```
🫣 모델의 성능을 측정하려고 하는데, 생각해보니 모델을 수립하고 훈련시키는 클래스는 정의했는데 성능을 측정하는 함수를 만들지 않았습니다. 모델의 성능을 측정하는 함수를 정의해보겠습니다.
```python
# AutoEncoder Test 성능(MSE Loss) 평가
def test_score(model, test_x): # 성능을 측정할 model과 성능 측정에 사용할 test용 데이터를 입력으로 받음
    preds = model.forward(test_x)  # 모델의 Output(예측값)
    actuals = test_x               # 일반적으로 Actual Label에는 y값을 입력하지만, 현재 AutoEncoder를 사용하기 때문에 이전 그림에서 봤던 것처럼 Target value는 Test Data의 실제 x값입니다.
    loss = 0
    crit = nn.MSELoss() # 모델의 Output과 Test Data의 실제 x값 사이의 MSE를 loss함수로 사용

    for pred, actual in zip(preds, actuals): # 팁! 간혹 이중 for문을 사용하는 대신 for문에서 zip()을 사용하면 for문을 한 번만 사용할 수 있습니다.
        loss += crit(pred,actual)

    test_loss = float(loss)/len(preds)
    
    print(f'Test Score(MSE): {round(test_loss,4)}')

    return round(test_loss, 4)
```
**Shallow AutoEncoder 및 Deep AutoEncoder MSE Loss**
- Shallow: 0.0295
- Deep: 0.022
- MSE Loss를 보았더니 Model의 Depth가 깊은만큼 Deep AutoEncoder가 Shallow AutoEncoder에 비해 복원을 잘한 것으로 보입니다. 그런데 수치상으로만 봤더니 복원을 어느정도 했는지 파악하기가 어렵습니다.

<br/>

- 원본 이미지와 모델이 복원한 이미지를 보여주는 함수를 정의해보겠습니다.
```python
# 원본 이미지와 복원된 이미지 시각화 코드
# 개인적으로 처음 이미지 데이터를 시각화했던 때를 생각하면 데이터의 구조 때문에 많이 헷갈렸습니다. (사실 지금도 그렇습니다...)
# 따라서 튜토리얼을 보시는 분들의 이해를 돕기 위해 주석으로 각 변수의 shape를 기재해두었으니 참고하시면 됩니다.
def visualize_og_and_recon(model1, model2, test_x):
    with torch.no_grad():
        index = int(random.random() * test_x.size(0)) # |test_x| = (10000, 784)  # test_x.size(0) = 10000
        recon_1 = model1(test_x[index].view(1,-1)).squeeze() # |test_x[index]| = (784,)
        recon_2 = model2(test_x[index].view(1,-1)).squeeze() # |test_x[index].view(1,-1)| = (1, 784)  
                                                             # view(1.-1)을 해준 이유는 model을 수립할 때 입력으로 (batch_size, 784)를 입력으로 받게 해놨기 때문.
                                                             # |model(test_x[index].view(1, -1))| = (1, 784)
                                                             # |model(test_x[index].view(1, -1)).squeeze()| = (784, )
        
        print('Original Image')
        show_image(test_x[index])
        print('='*40)
        print(f'First Model\'s Reconstruction Image')
        show_image(recon_1)
        print(f'Second Model\'s Reconstruction Image')
        show_image(recon_2)
```
**시각화 결과**  
</br>
원본 이미지
<p align = 'left'>
<img src = 'https://user-images.githubusercontent.com/56019094/202088809-5eccfc58-f2ad-4ede-ab00-faa91a7a3fbc.png'>
</p> 
</br>

Deep AutoEncoder가 복원한 이미지
<p align = 'left'>
<img src = 'https://user-images.githubusercontent.com/56019094/202088968-033b7150-6dc6-4ba7-b499-5153124f50a1.png'>
</p>
</br>

Shallow AutoEncoder가 복원한 이미지
<p align = 'left'>
<img src = 'https://user-images.githubusercontent.com/56019094/202089094-92ff54c0-f54d-46b0-b78f-8a1a1e2d22c7.png'>
</p>
</br>

🧐 원본 이미지와 복원된 이미지의 결과를 보니까 조금 더 직관적으로 이해가 되는 것 같습니다.
- Deep AutoEncoder가 복원한 이미지가 Shallow AutoEncoder가 복원한 이미지에 비해 조금 더 숫자 7이라는 것이 선명하게 보이고, Shallow AutoEncoder의 경우 복원된 이미지에서 숫자 7의 경계 부분이 더 흐릿한 것을 확인할 수 있습니다.
- 🤨 그런데 두 AutoEncoder가 복원한 이미지는 원본 이미지와 달리 숫자 7 주변에 다른 무엇인가 형체가 보이는 것 같습니다.
- 이것은 아마도 Latent Space 상에서 숫자 7이 다른 숫자들과 완벽히 분리되어 있지 않고 겹쳐져 있기 때문인 것으로 생각됩니다.
- 그렇다면 한 번 Latent Space를 시각화 해보겠습니다.
```python
# Shallow AutoEncoder의 Latent Space 시각화 (Deep AutoEncoder의 경우 아래 shallw_model을 deep_model로 변경해주기만 하면 됩니다.)

color_map = [
    'brown', 'red', 'orange', 'yellow', 'green',
    'blue', 'navy', 'purple', 'gray', 'black',
]  # 10개의 색을 이용: 0~9를 각각 brown, red, ..., black으로 시각화하기 위해.

plt.figure(figsize=(10, 5))

with torch.no_grad(): # 학습시키는 과정이 아니니까 with torch.no_gard(): 사용!
    latents = shallow_model.cpu().encoder(test_x[:1000].cpu())  # 1000개의 test data를 넣어봄
                                            # |latents| = (1000, 2)
    
    for i in range(10):
        target_latents = latents[test_y[:1000] == i].cpu()
        target_y = test_y[:1000][test_y[:1000] == i].cpu()
        plt.scatter(target_latents[:, 0],
                    target_latents[:, 1],
                    marker='o',
                    color=color_map[i],
                    label=i)
    
    plt.legend()
    plt.grid(axis='both')
    plt.show()
```
**Deep AutoEncoder와 Shallow AutoEncoder의 Latent Space 시각화 결과**

Deep AutoEncoder의 Latent Space 시각화 결과
<p align = 'center'>
<img src = 'https://user-images.githubusercontent.com/56019094/202089748-4647ea9d-0c0d-4de0-8291-6f8d9492f831.png'>
</p>  

Shallow AutoEncoder의 Latent Space 시각화 결과
<p align = 'center'>
<img src = 'https://user-images.githubusercontent.com/56019094/202089879-82ca4587-9e68-49a0-af55-7c301dc6a754.png'>
</p>

- 조금 전에 예상했던 것처럼 보라색 점에 해당하는 숫자 7이 혼자 떨어져 있는 것이 아니라 다른 숫자들과 Latent Space 상에서 겹쳐있는 것을 확인할 수 있습니다.
- 또한 눈여겨볼만한 점은 Deep AutoEncoder의 Latent Space 상에서는 각 숫자들이 Shallow AutoEncoder의 경우보다 조금 더 군집화가 잘 되어 있는 것으로 보입니다. 
- 지금까지는 AutoEncoder의 Model Depth를 변경하고 복원 이미지 시각화 및 Latent Space 시각화를 통해 모델의 성능 차이에 대한 간단한 탐구를 진행해보았습니다.
- 위 그림에서 저희가 시각화한 Latent Space는 Bottleneck에 해당하는 벡터가 존재하는 Vector Space입니다. 
- 아래 그림을 보시면 조금 더 직관적으로 이해가 되실 것 같습니다.

<p align = 'center'>
<img src = 'https://user-images.githubusercontent.com/56019094/202090184-f78c44e3-70b8-4a89-a3d1-03aff9bb9db1.png'>
</p>

<p align = 'center'>
이미지 출처: https://www.jeremyjordan.me/autoencoders/
</p>

위에서 그림을 통해 알 수 있듯이, Bottleneck은 일종의 zip파일이라고 생각할 수 있습니다. 즉, Input Data의 특징들을 잘 요약해서 압축시켜놓은 것입니다. 그렇다면 이제 이 Bottleneck의 크기를 변경시키면서 모델의 성능 변화를 측정해보겠습니다. 그리고 **AutoEncoder를 이용한 Anomaly Detection**까지 실험해보겠습니다. (AutoEncoder로는 이전에 사용했던 Deep AutoEncoder를 사용)

## 실험 2. AutoEncoder의 Bottleneck Size 변화에 따른 AutoEncoder의 성능 변화 및 Anomaly Detection 성능 변화 측정
- 먼저 AutoEncoder의 Bottleneck Size를 변화시키면서 MSELoss가 어떻게 변화하는지 살펴보겠습니다.
```python
# Bottleneck Size = [ 2,  4,  6,  8, 10, 12, 14, 16, 18]으로 늘려가면서 모델 훈련시켜보기
model_list = []
Btl_size_list = np.arange(2,20,2) # 실험할 Bottleneck Size를 List 형태로 저장

for i in Btl_size_list: 
    globals()['model_btl_size{}'.format(i)] = Deep_AE(btl_size = i) # 동적 변수를 사용해서 여러 가지 변수를 생성해야할 때 좀 더 편리하게 변수를 선언할 수 있습니다.
    model_list.append(globals()['model_btl_size{}'.format(i)]) # model_list에는 [model_1, model_2, model_3, ...]와 같이 AutoEncoder 모델들이 Element로 들어가 있음

# 각 모델을 불러와서 Train 수행
models = []
for model in model_list:   
    print('Model Training 시작')
    model = model
    optimizer = optim.Adam(model.parameters())
    crit = nn.MSELoss()
    trainer = Trainer(model, optimizer, crit, batch_size = 512, n_epochs = 100, verbose = 1)
    trainer.train((train_x.to(device), train_x.to(device)), (valid_x.to(device), valid_x.to(device)))
    models.append(model) # 수립된 모델은 models라는 새로운 List에 추가
```
<br/>

9가지 크기의 Bottleneck Size를 각각 가진 AutoEncoder를 수립했으니 이제 한 번 MSE Loss를 비교해보겠습니다.
```python
# Test Score 계산
crit = nn.MSELoss()
test_scores = [crit(models[i].cpu().forward(test_x).detach(), test_x.cpu()) for i in range(len(models))] # 수립된 각 Model의 Test Set에 대한 MSE Loss 측정
plt.plot(Btl_size_list,test_scores, label = "MSE Loss according to Bottleneck Size")
plt.legend()
```
**위 코드 결과**
<p align = 'center'>
<img src = 'https://user-images.githubusercontent.com/56019094/202091573-dcd9a41e-87bb-450f-be6c-383b71178bdd.png'>
</p>

😁 꽤나 흥미로운 결과가 나온 것 같습니다.    
- Bottleneck Size가 AutoEncoder의 성능에 영향을 준다는 것은 이미 자명한 사실이고, Bottleneck Size가 2에서 10으로 증가할 때까지는 MSE Loss가 가파르게 감소해서 모델의 성능이 향상됨을 알 수 있습니다. 반면, Bottleneck Size가 10 이상이 되면 성능에는 큰 영향이 없는 것으로 보입니다. - 다만, 14에서 18로 Bottleneck Size가 증가할 때 MSE Loss가 다시 감소하는 것으로 보이는데 이에 대해서 추가 실험을 진행해보겠습니다.

**Bottleneck Size를 10 ~ 30으로 step = 2로 늘려가면서 추가 실험한 결과**
<p align = 'center'>
<img src = 'https://user-images.githubusercontent.com/56019094/202092145-663ba905-e48d-4dd6-bf73-1b1152a64995.png'>
</p>

- 첫 번째 실험에서는 Bottleneck Size를 14에서 16으로, 16에서 18로 증가시키면 꾸준히 MSE Loss가 올라가서 모델의 성능이 하락하는 것처럼 보였습니다.
- 추가 실험을 한 결과, 위 그래프에서 보면 이 구간에서는 일시적으로 증가하는 것처럼 보였으나, 오히려 Bottleneck Size = 20일 때 가장 낮은 MSE Loss를 보이는 경우도 있었습니다.
- 그러나 Bottleneck Size = 20인 구간 이후에는 또 다시 MSE Loss가 커지는 등 Bottleneck Size가 2 -> 10으로 증가할 때처럼 확실히 모델의 성능이 증가한다 혹은 감소한다라는 해석을 하기에는 부족한 것 같습니다.
- 그럼에도 불구하고, 이번 실험을 통해 특정 숫자까지 Bottleneck Size를 늘리는 것이 확실히 모델 성능에 긍정적인 영향을 끼친다는 것은 확인할 수 있었습니다.

</br>

이제 그렇다면 본 챕터의 주제인 **Anomaly Detection**을 AutoEncoder를 이용해 수행해보겠습니다.  
그런데 튜토리얼에서 사용하게 된 데이터셋은 MNIST 데이터셋이라 직관적으로 Anomaly라고 할만한 데이터가 존재하지 않습니다. 간혹, 0~9 중에 한 숫자를 Anomaly라고 가정하고 Anomaly Detection을 수행하기도 하지만, 개인적으로는 조금 더 직관적인 예시를 보이고 싶어서 텍스트를 이미지로 만드는 함수를 정의해 이를 Anomaly로 사용해보겠습니다.
```python
# 인위적으로 Anomaly Instance 생성
# 본 튜토리얼에서 AutoEncoder는 숫자 0~9만을 이용해서 학습을 진행했음
# 따라서 숫자가 아닌 알파벳이나 한글 이미지가 입력되면 Reconstruction Error가 정상(숫자) 데이터에 비해 높아 Anomaly로 판단할 것임

from PIL import Image, ImageDraw, ImageFont

def make_anomaly_image(text: str):
    # 이미지로 출력할 글자 및 폰트 지정
    draw_text = text
    font = ImageFont.truetype('/root/BA/NanumSquareB.ttf')

    # 이미지 사이즈 지정 (MNIST Dataset의 이미지 크기와 동일하게 지정)
    text_width, text_height = 28, 28

    # 이미지 객체 생성 (MNIST 셋과 동일하게 이미지 Channel은 Grayscale, 배경은 검은색으로)
    canvas = Image.new('L', (text_width, text_height), "black")

    # 가운데에 그리기
    draw = ImageDraw.Draw(canvas)
    w, h = font.getsize(draw_text)
    draw.text(((text_width-w)/2.0,(text_height-h)/2.0), draw_text, 'white', font)
    display(canvas)  # 만들어진 텍스트 이미지 보기

    # Tensor 형태로 변환
    canvas = torch.Tensor(np.array(canvas)/ 255.) + 0.2*np.random.normal(0,1, size = (28,28)) # 가우시안 노이즈 추가
    return canvas
```

```python
# 인위적으로 anomaly image 80장을 만들고 원래 test_x에서 920장 가져와서 model 별 성능 측정하기  --> 생각해보니 이것들은 label이 없어서 성능 측정이 불가능하네??? text_x[:920]에 대한 label은 test_y[:920]
korean = "가,나,다,라,마,바,사,아,자,차,카,타,파,하".split(",")
english = "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z".split(",")

artificial_anomalies = [make_anomaly_image(x).flatten() for x in (korean + english)]
noised_anomalies = [image + 0.1*np.random.uniform(0,1,size = image.shape) for image in artificial_anomalies]
artificial_anomalies += noised_anomalies

artificial_anomalies = torch.stack(artificial_anomalies, dim = 0)

# 기존 test_x 에서 Normal 데이터 920개, 새롭게 생성한 이상치 데이터 80개를 합쳐서 새로운 test_x 데이터와 해당하는 Label을 가진 test_y 생성
new_test_x = torch.cat([test_x[:920,].cpu(), artificial_anomalies], dim = 0)
new_test_y = [0]*920 + [1]*80
```
**생성된 이미지 예시**   
(실제로 Anomaly로 사용하게 된 텍스트 이미지는 아래 이미지에 Gaussian Noise도 추가했습니다)

한글 텍스트 이미지
<figure class = 'third'>
    <img src = 'https://user-images.githubusercontent.com/56019094/202094970-ee9046d2-0253-4fcd-8b31-a0eae7da933c.png' width = 32%>
    <img src = 'https://user-images.githubusercontent.com/56019094/202094994-77fa8367-5853-47ba-b38a-949ea187e2ae.png' width = 32%>
    <img src = 'https://user-images.githubusercontent.com/56019094/202095022-d7a5b8fb-f1c1-43f3-b18c-6a1e8279b6cc.png' width = 32%>
</figure>

영어 텍스트 이미지
<figure class = 'third'>
    <img src = 'https://user-images.githubusercontent.com/56019094/202095550-dd4f1366-da95-4a6f-8173-8e8048541b95.png' width = 32%>
    <img src = 'https://user-images.githubusercontent.com/56019094/202095570-e01a89d1-090c-497e-a8b5-b2a3d556a574.png' width = 32%>
    <img src = 'https://user-images.githubusercontent.com/56019094/202095584-18e04eaa-b1e5-45af-b60f-4958ad410b65.png' width = 32%>
</figure>

저희가 수립한 AutoEncoder는 모델 수립 시에 0~9 숫자(Normal) 이미지만을 학습했기 때문에 위와 같은 숫자가 아닌(Not Normal 즉, Anomaly) 데이터를 Input으로 받으면 제대로 복원 해내지 못할 것입니다. 따라서 Reconstruction Loss로 사용할 MSE Loss가 높을 것이며, 이 MSE Loss가 특정 Threshold보다 높을 경우 Anomaly(1)로, Threshold보다 낮다면 Normal(0)으로 판단하게 될 것입니다.  

이제 그렇다면 모델이 Anomaly 여부를 판단하게 될 기준인 Threshold를 정해주는 함수를 정의해보겠습니다. 본 튜토리얼에서는 일차원적인 방법을 이용했지만, 실제로는 Validation Data를 이용해 Threshold 자체도 일종의 하이퍼 파라미터로 튜닝을 해주기도 합니다.
```python
# Train Data를 이용해서 각 모델별 Threshold 구하기

def find_threshold(models:list, train_x) -> dict:
    model_threshold_dict = {}
    for i in range(len(models)):
        model = models[i].cpu()
        train_x = train_x.cpu()
        output = models[i].forward(train_x).detach()
        losses = torch.mean(((train_x - output)**2), dim = -1)
        model_threshold_dict[i] = np.round(np.percentile(losses, 90), 4)  # 90분위수에 해당하는 loss값을 Threshold로 설정

    return model_threshold_dict
```
**위 코드 결과 예시**  
{0: 0.0528,
 1: 0.0384,
 2: 0.0303,
 3: 0.0276,
 4: 0.0249,
 5: 0.0254,
 6: 0.024,
 7: 0.0248,
 8: 0.027}  

 해석: 첫 번째 모델의 Threshold는 0.0528로, 두 번째 모델의 Threshold는 0.0384로 사용

<br/>

이제 각 모델별로 Threshold가 정해졌으니 Anomaly Detection을 수행하는 즉, 새로운 데이터에 대해 Anomaly인지 아닌지를 판단하는 함수를 정의하고 Anomaly Detection까지 수행을 해보겠습니다.
```python
def anomaly_detect(x, model, threshold = 0.06):
    if x.dim() == 1:
        x = x.unsqueeze(dim=0)

    model_outputs = model.forward(x).detach()
    actuals = x
    preds = []
    recon_errors = []

    crit = nn.MSELoss()
    
    for model_output, actual in zip(model_outputs, actuals):
        recon_errors.append(crit(model_output,actual))

    preds = [1 if recon_error > threshold else 0 for recon_error in recon_errors ] # Reconstruction Error가 Threshold보다 크면 1(Anomaly), Threshold보다 작으면 0(Normal)으로 판단
    
    return preds
```
```python
preds = []

for i, model in enumerate(models):
    model_threshold = model_threshold_dict[i] # 모델별로 상이한 threshold를 적용함
    preds.append(anomaly_detect(x = new_test_x.type(torch.float32), model = model, threshold = model_threshold)) # preds[0]: 첫번째 model의 예측 결과, preds[1]: 두번째 model의 에측 결과, ...
```
```python
# Accuracy 및 F1 Score 반환하는 함수 정의
from sklearn.metrics import accuracy_score, f1_score 

def result_metrics(pred, actual):
    assert len(pred) == len(actual)
    f1 = f1_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    return f1, accuracy
```
위 코드를 보면, Anomaly Detection Model의 성능을 측정하는데 Accuracy와 F1 Score를 사용합니다. 의아해 하실수도 있지만, 이전에 'Overview'에서 설명드린 바와 같이 Anomaly Detection 또한 일종의 Classification이기 때문에 Accuracy와 F1 Score를 이용해 성능 측정이 가능합니다.

👍🏼 Remind
<p align = 'center'>
<img src = "https://user-images.githubusercontent.com/56019094/201926047-f15dfd88-65fb-45ec-8bcb-93e24712fd52.png">
</p>

**Bottleneck Size에 따른 Anomaly Detection 성능 결과**

<center>

|Bottleneck Size | Accuracy | F1 Score |
|----------------|----------|----------|
|2 |0.896|0.6061
|4 |0.863|0.5387
|6 |0.854|0.5229
|8 |0.854|0.5228
|10|0.846|0.5096
|12|0.853|0.5212
|14|0.86|0.5333
|16|0.86|0.5333
|18|0.85|0.5161
|20|0.847|0.5112
|22|0.857|0.5281
|24|0.851|0.5178
|26|0.84|0.5000
|28|0.852|0.5195
|30|0.853|0.5212

</center>

- Bottleneck Size가 2인 경우, Accuracy와 F1 Score 둘 다 다른 경우들에 비해 높은 것을 확인할 수 있습니다.
- Bottleneck Size가 2와 4인 경우를 제외하고는 Bottleneck Size가 14와 16인 경우를 중심으로 멀어질수록 조금씩 Accuracy와 F1 Score가 감소하는 경향을 보이고 있습니다.
- Bottleneck Size가 작을수록 Encoder를 거친 vector가 Input에 대한 정보 손실량이 많아 Anomaly Detection 성능 또한 저조할 것이라 예상했는데, 실험 결과 오히려 Bottleneck Size가 2인 경우가 가장 높은 성능을, 그리고 그 다음으로 Bottleneck Size가 작은 4인 경우가 그 다음으로 높은 성능을 보인 것이 매우 흥미로웠습니다. 
- 실험 한계1: Bottleneck Size를 여러 개로 설정하고 모델을 수립했으나, Epoch 수를 모두 동일하게 지정했기 때문에, 이 성능이 각 모델의 최대 성능이 아니라 정확한 비교가 아니라는 점.
- 실험 한계2: Threshold를 설정할 때 90분위수라는 일차원적인 방법론을 적용했기에 이 또한 각 모델의 Anomaly Detection 측면에서의 최대 성능이 아닐 수 있기에 정확한 비교가 아니라는 점.

<br/>

## 실험 3. What if? Encoder의 Layer 수 = Decoder의 Layer 수 * 2라면?? 그리고 반대의 경우에는?
    - Base AutoEncoder vs Encoder Deeper vs Decoder Deeper

```python
# 아래 코드에서 Enc_Deeper_AE, Dec_Deeper_AE는 각각 약 13만 개의 파라미터 사용, 9개의 Layer 사용 
# 아래 비교 대상 Baseline_AE도 이와 유사한 환경을 맞춰주기 위해 파라미터 수(약 12만 5천개) 및 Layer 수 (총 8개의 Layer) 설정함
class Baseline_AE(nn.Module):
    
    def __init__(self, btl_size=2):   
        self.btl_size = btl_size
        
        super().__init__()
        
        # Encoder에는 6개의 Layer를, Decoder에는 3개의 Layer를 사용하여 테스트
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28,75),  
            nn.ReLU(),
            nn.Linear(75, 45),
            nn.ReLU(),
            nn.Linear(45, btl_size),    
        )
        self.decoder = nn.Sequential(
            nn.Linear(btl_size, 45),
            nn.ReLU(),
            nn.Linear(45,75),
            nn.ReLU(),
            nn.Linear(75, 28 * 28),
        )        
        
    def forward(self, x):
        z = self.encoder(x)  # |x| = (batch_size, 784), |z| = (batch_size, btl_size)
        y = self.decoder(z)  # |y| = |x|
        
        return y
```
```python
class Enc_Deeper_AE(nn.Module):
    
    def __init__(self, btl_size=2):   
        self.btl_size = btl_size
        
        super().__init__()
        
        # Encoder에는 6개의 Layer를, Decoder에는 3개의 Layer를 사용하여 테스트
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 100),  
            nn.ReLU(),
            nn.Linear(100, 70),
            nn.ReLU(),
            nn.Linear(70,50),
            nn.ReLU(),
            nn.Linear(50,30),
            nn.ReLU(),
            nn.Linear(30, btl_size),    
        )
        self.decoder = nn.Sequential(
            nn.Linear(btl_size, 50),
            nn.ReLU(),
            nn.Linear(50, 28 * 28),
        )        
        
    def forward(self, x):
        z = self.encoder(x)  # |x| = (batch_size, 784), |z| = (batch_size, btl_size)
        y = self.decoder(z)  # |y| = |x|
        
        return y
```
```python
class Dec_Deeper_AE(nn.Module):
    
    def __init__(self, btl_size=2):   
        self.btl_size = btl_size
        
        super().__init__()
        
        # Encoder에는 6개의 Layer를, Decoder에는 3개의 Layer를 사용하여 테스트
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 50),  
            nn.ReLU(),
            nn.Linear(50, btl_size),    
        )
        self.decoder = nn.Sequential(
            nn.Linear(btl_size, 30),
            nn.ReLU(),
            nn.Linear(30,50),
            nn.ReLU(),
            nn.Linear(50,70),
            nn.ReLU(),
            nn.Linear(70,100),
            nn.ReLU(),
            nn.Linear(100, 28 * 28),
        )        
        
    def forward(self, x):
        z = self.encoder(x)  # |x| = (batch_size, 784), |z| = (batch_size, btl_size)
        y = self.decoder(z)  # |y| = |x|
        
        return y
```
- 실험 전 결과 예상  
    1) Encoder Layer를 더 깊게 했을 때 예상 결과: Encoder의 중요 역할은 Input Data의 중요한 정보를 압축하는 역할인데, Decoder에 비해 Encoder의 깊이가 깊다보니 Decoder가 충분히 복원해내지 못할만큼 압축이 되어서 성능 저하로 이어질 것으로 예상
    2) Decoder Layer를 더 깊게 했을 때 예상 결과: Decoder의 역할은 Encoder를 통해 압축된 Input Data를 통해 원본 데이터를 복원하는 것인데, 이러한 측면에서 예상하기로는 Decoder의 깊이가 깊어진 것은 성능 저하로 이어지지 않을 것으로 예상됨

<br/>

**실험 결과**
- Base AutoEncoder: Accuracy = 0.977, F1-Score = 0.8743
- Encoder Deeper AutoEncoder: Accuracy = 0.98, F1-Score = 0.8876
- Decoder Deeper AutoEncoder: Accuracy = 0.977, F1-Score = 0.8743

<br/>

  

- 본 튜토리얼에 사용된 데이터를 대상으로 Encoder 및 Decoder 레이어 수의 변화에 따른 실험 결과 성능 차이는 생각보다 매우 미미했음.  

- 🙄 예상 외의 결과로, 해당 실험 주제와는 별개이지만 이전 실험 2.(Deep AutoEncoder 사용했음)에서 Bottleneck_size = 2일 때가 성능이 가장 좋았는데(F1 score = 0.6061) 이번 실험에서는 AutoEncoder의 Bottleneck_size = 2로 하고 모델의 레이어를 변화시켰더니 F1-Score와 Accuracy가 매우 높아졌음  
- 실험 3. 에서 사용된 세 AutoEncoder는 Layer 수가 Baseline의 경우 8개, Encoder Deeper와 Decoder Deeper는 9개의 레이어를 사용한 반면, 실험 2.에서 사용된 Deep AutoEncoder의 Layer 수는 14개였음. 즉, Deep AutoEncoder의 Model Depth가 약 1.5배 더 깊지만 오히려 실험 3.에서 사용한 AutoEncoder에 비해 F1-Score가 0.25 이상 차이가 났음.
- 이러한 결과가 나온 이유로 예상되는 내용은 (1) Deep AutoEncoder의 Anomaly Detection에 사용된 Threshold가 부적절, (2) Deep AutoEncoder의 모델 Training Epoch 수가 부족 (3) Deep AutoEncoder의 Test Set에 대한 MSE Loss가 실험 2.에서 사용한 세 AutoEncoder의 MSE Loss보다 모두 낮았다는 점에서 Deep AutoEncoder가 '예상과 달리 AutoEncoder가 원본 데이터만 잘 복구하는 것이 아니라 Anomaly마저 잘 복구할 정도로 Generalization이 되는 문제'에 해당한다는 것입니다.
    - (1) Deep AutoEncoder와 실험 2.에서 사용한 세 AutoEncoder 모두 동일하게 Training Set에 대한 MSE Loss값들 중 90분위수를 Threshold로 사용했는데, 이와 같은 일차원적인 방법론을 사용했기에 Deep AutoEncoder의 최대 성능을 제대로 발휘하지 못한 것으로 추측됩니다.
    - (2) Deep AutoEncoder의 모델 Depth가 실험 2.에서 사용된 모델들에 비해 약 1.5배 깊은 반면에, 모두 동일하게 100 Epoch 동안 Train했습니다. Model Size가 클수록 Model의 학습에 많은 Epoch가 소요된다는 측면에서 실험 2.에서의 세 AutoEncoder들보다 Deep AutoEncoder가 덜 학습된 상태였다고 추측됩니다.
    - (3) 해당 문제가 발생한 것은 아닌가라는 생각에 실험 외에 별도로 시각화를 통해 인위적으로 추가한 Anomaly Instance를 복원해서 시각화한 결과, 학습에 사용한 Normal 데이터만큼 잘 복원하지는 않음을 확인할 수 있었습니다.