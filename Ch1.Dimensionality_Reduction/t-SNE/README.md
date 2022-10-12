# t-SNE

**t**-distributed **S**tochastic **N**eighbor **E**mbedding

- t-SNE와 SNE (Stochastic Neighbor Embedding)의 차이는 저차원 공간에 대해서 가우시안 분포를 사용하느냐 혹은 t-분포를 사용하느냐
- 따라서 t-SNE에 대해 알아보기 이전에 SNE를 알아보자

## SNE (Stochastic Neighbor Embedding)

- 가까운 이웃 객체들과의 거리 정보를 잘 보존하는 것이 멀리 떨어진 객체들과의 거리 정보를 보존하는 것보다 더 중요하다
    
    → SNE는 LLE (Locally Linear Embedding)과 기본 아이디어가 유사함 
        (실제로 Stochastic Neighbor Embedding 방법론을 제안한 논문의 저자 중 LLE 저자가 있음)
    
- Local Pairwise Distance를 확정적(Deterministic)이 아닌 **확률적(Probabilistic)**으로 정의함
→ LLE와의 주요 차이점
    
    ![Untitled](https://user-images.githubusercontent.com/56019094/195230601-9017bc91-2e2c-424f-84aa-1c87a6429736.png)
    
- 원래 차원과 임베딩 후 저차원에서 두 객체 간의 **이웃 관계는 잘 보존**이 되어야 함
= LLE의 기본 아이디어와 동일

---

- 고차원(원래 차원)에서 객체 i가 객체 j를 이웃으로 선택할 확률 $p_{j|i}$
    
    ![Untitled 1](https://user-images.githubusercontent.com/56019094/195230604-5849bd25-34eb-4a09-b794-ea0fab83fa30.png)

    

- 저차원(축소된 차원)에서 객체 i가 객체 j를 이웃으로 선택할 확률 $q_{j|i}$
    
    ![Untitled 2](https://user-images.githubusercontent.com/56019094/195230608-49cc5c97-3587-47a9-aba2-8e538ed0e038.png)

    

<aside>
💡 원래 차원에서 객체 i가 객체 j를 이웃으로 선택할 확률 $p_{j|i}$와 축소된 차원에서 객체 i가 객체 j를 이웃으로 선택할 확률 $q_{j|i}$를 최대한 같게 만들고 싶다

</aside>

( ${x}$: 고차원 공간에서의 Data Point(객체), ${y}$: 저차원 공간에서의 Data Point(객체))

- $p_{j|i}$의 분자에서 ${x}_i$와 ${x}_j$가 가까울수록 $||{x}_i - {x}_j||^2$ 값은 작아짐
    
    → $p_{j|i}$의 분자 값 커짐 → $p_{j|i}$ 값 커짐
    
    ↔ LLE에서는 $p_{j|i}$ = 1 if 객체 j가 객체 i의 K-NN, 0 otherwise (Deterministic)
    

⇒ $p_{j|i}$ 계산 시 $\sigma_i$ (= Radius of Gaussian)은 어떻게 정할 것인가?

- 고차원 데이터와 저차원 데이터의 밀도가 다를 수 있기 때문에, 선택되는 이웃의 수를 일정하게 유지하기 위해서는 서로 다른 Radius를 사용해야 함
- 너무 큰 Radius는 Entropy가 높아지는 반면, 너무 낮은 Radius는 Entropy가 낮아짐
    
    → 원하는 수준의 Entropy를 먼저 정하고 적합한 Radius를 결정
    
    - Entropy 계산식: $H(P_i) = \sum_jp_{j|i}log_2p{j|i}$

- 저차원 공간 상으로 임베딩이 잘 되었는지는 어떻게 평가할 것인가?
    
    → KL (Kullback-Leibler) Divergence Cost Function 사용
    
    </aside>
    
    - $Cost = \sum_i KL(P_i||Q_i) = \sum_i \sum_jp_{j|i}log{p_{j|i}\over q_{j|i}}$
        
        + 위 수식에서 볼 수 있듯이 KL Divergence는 Distance 지표로는 사용 불가
        

- Remind! $p_{j|i}$는 x에 대한 함수, $q_{j|i}$는 y에 대한 함수

![Untitled 1](https://user-images.githubusercontent.com/56019094/195230604-5849bd25-34eb-4a09-b794-ea0fab83fa30.png)

![Untitled 2](https://user-images.githubusercontent.com/56019094/195230608-49cc5c97-3587-47a9-aba2-8e538ed0e038.png)

→ ${x}$는 원래 차원에서의 좌표 (알고 있는 값)  ${y}$는 축소된 차원에서의 좌표
→ ${y}$: 알고자 하는 값(미지수)

⇒ 아래 식을 통해 Gradient Descent 방법으로 학습

![Untitled 3](https://user-images.githubusercontent.com/56019094/195230609-99fd05f6-5795-4e50-a354-3f75d608612b.png)


⇒ Gradient Update 수식

![Untitled 4](https://user-images.githubusercontent.com/56019094/195230610-89ce9a3e-66d7-453f-a31b-46e5804d8df6.png)


(기존 Cost Function수식에서 위와 같은 수식이 전개된 과정은 **“고려대학교 산업경영공학부 DSBA 연구실 01-7: Dimensionality Reduction - tSNE” 영상 22:21**부터를 참고해주시면 감사하겠습니다.)

# Symmetric SNE

- Standard SNE 수식에서는 $p(j|i) \neq p(i|j)$

![Untitled 5](https://user-images.githubusercontent.com/56019094/195230614-070b17a5-e97d-4f29-93bf-213b7c508233.png)


⇒ i와 j에 대한 조건부 확률을 다르게 설정하지 말고 Pairwise Probability로 표현

![Untitled 6](https://user-images.githubusercontent.com/56019094/195230617-1c03a5ee-2811-4bf3-9133-52a2c8e59fdf.png)

- “i 기준에서 j를 이웃으로 선택할 확률” $p_{j|i}$, “j기준에서 i를 이웃으로 선택할 확률” $p_{i|j}$ 대신 “i와 j가 이웃일 확률” $p_{ij}$로 바꾸자
    - $\sum_ip_{ij}$ > $1 \over 2n$: i와 j가 이웃이 될 확률이 적어도 일정 기준 이상은 되도록 하기 위함

⇒ Cost Function과 Gradient를 아래와 같이 단순화 시켜서 연산량을 줄일 수 있음

![Untitled 7](https://user-images.githubusercontent.com/56019094/195230619-ed9e0fb5-cb77-4757-83d0-54017cbc77af.png)

> ↔ Standard SNE의 경우
> 
> 
> ![Untitled 8](https://user-images.githubusercontent.com/56019094/195230623-b412a97c-4aae-4c9f-a114-bfbd7ea1c6d7.png)
> 

😵 그러나, Symmetric SNE는 Standard SNE와 동일하게 여전히 **Crowding Problem 문제** 존재

- Cost Function을 단순화 시켜 연산량을 줄일 수 있다는 장점이 존재하나, 여전히 Standard SNE와 동일하게 **Gaussian Distribution**을 사용
    
    ⇒ 객체 i로부터 적당히 거리가 있는 객체들이 선택될 확률은 i와 가까운 거리에 있는 객체들에 비해서 급격하게 감소
    
    ![Untitled 9](https://user-images.githubusercontent.com/56019094/195230627-af188cd1-738d-42e2-a3f3-aac96b1a9677.png)

    

# t-SNE

- Gaussian Distribution에서 비롯된 Crowding Problem을 해결하자
    - Gaussian Distribution은 평균에서 멀어지면 밀도 함수의 값이 급격히 감소함
    - Gaussian Distribution보다 완만한 형태의 분포 함수를 사용하자
    
    ⇒ 자유도가 1인 t-Distribution 사용
    
    ![Untitled 10](https://user-images.githubusercontent.com/56019094/195230633-88e75346-028b-4d64-87f5-4b2fcf507782.png)

    
    이미지 출처: [http://bigdata.dongguk.ac.kr/lectures/med_stat/_book/표본분포.html](http://bigdata.dongguk.ac.kr/lectures/med_stat/_book/%ED%91%9C%EB%B3%B8%EB%B6%84%ED%8F%AC.html)
    

- t-SNE에서의 Pairwise Probability 변환
    - 원래 차원에서는 Gaussian Distribution 사용, 축소된 저차원에서는 t-Distribution 사용
        
        ![Untitled 11](https://user-images.githubusercontent.com/56019094/195230636-dd3c2f0a-fc03-4540-805b-9dbaec3203bb.png)

        
- Cost Function으로는 이전과 동일하게 KL Divergence 사용
- t-SNE의 Gradient 수식
    
    ![Untitled 12](https://user-images.githubusercontent.com/56019094/195230638-9e0dc95e-ac3a-464b-a3fb-9518e816b08b.png)

    
- t-SNE 결과 예시

    
    ![Untitled 13](https://user-images.githubusercontent.com/56019094/195230640-a81cf97a-25bc-4467-9ed5-75988a847c8d.png)

    
    이미지 출처: 고려대학교 산업경영공학과 대학원 강필성 교수님 Business-Analytics 강의 01_4_Dimensionality Reduction_ISOMAP, LLE, tSNE 강의 자료
