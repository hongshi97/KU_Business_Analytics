# SVM
---
**Tutorial for Business Analytics**  
😉 If you are curious about the time it takes to establish an SVM model by kernel function, I tried some experiment. Check [코딩 실습]
- 본 Tutorial은 고려대학교 산업경영공학부 대학원 Business Analytics 강의 자료를 기반으로 작성되었습니다.
---

## 목차

1. [이론](#이론)
   
   1. [Margin](#Margin)
   
   2. [Optimization](#Optimization-문제)
   
   3. [Soft Margin SVM](#Soft-Margin-SVM)
   
   4. [Nonlinear&Kernel](#Nonlinear&Kernel)
   
2. [코딩 실습](#코딩-실습)
   
   1. [실험 주제](#실험-주제)
   
   2. [Main Experiment - Support Vector Classifier](#Main-Experiment---Support-Vector-Classifier)
   
   1. [SVC 결과 및 해석](#SVC-결과-및-해석)
   
   3. [Additional Experiment - Support Vector Regressor](#Additional-Experiment---Support-Vector-Regressor)
   
   1. [SVR 결과 및 해석](#SVR-결과-및-해석)
   
   4. [결론](#결론)
   
---
># **이론**

**S**uppor **V**ector **M**achine

Keywords: Margin, Hyperplane, Support Vector
 

📢 요약: Support Vector Machine은 Vector Space 상에서 Vector들을 가장 잘 분류하는 Hyperplane을 수립하는 것을 목표로 한다.

- Background
    - Hyperplane(초평면): a subspace of one dimension less than its ambient space
    
      
    <p align="center">
    <image src=https://user-images.githubusercontent.com/56019094/199520467-88e4f48d-11e1-42a2-868b-0ca4ddcf7b56.png
    height="300"/>  
    </p>
    이미지 출처: Support Vector Machines without tears
        
    - n차원의 공간에서 Hyperplane은 n-1차원의 subspace를 의미
        - 2차원의 경우 Hyperplane은 1차원(직선)
        - 3차원의 경우 Hyperplane은 2차원(평면)  
        ⇒ SVM에서 Hyperplane은 어떤 Vector Space 상에 존재하는 Vector들을 분류하는 Decision Boundary(결정 경계)에 해당
        

>## Margin

>>### “가장 잘 분류하는”의 기준이 무엇인가?

- SVM은 Vector Space 상에 있는 Vector 형태로 표현된 각 Data Point들을 가장 잘 분류하는 Hyperplane(초평면)을 수립하는 것을 목표로 함
  

<p align = 'center'>
<image src=https://user-images.githubusercontent.com/56019094/199521480-96c01d5c-4bc4-4f18-8273-b4814f0320b6.png height = '300'></p>
    

>>### Margin을 최대화하는 Hyperplane을 찾자

- Margin이란?
  
    : Hyperplane으로부터 등 간격으로 양쪽으로 확장시켰을 때 Hyperplane과 가장 가까운 객체(Support Vector)와의 거리
    
    <p align = 'center'>
    <image src=https://user-images.githubusercontent.com/56019094/199521778-96a75042-461b-4104-98e3-325bd3c33065.png height = '300'></p>
    
    - 위의 Hyperplane 네 개 모두 다 데이터들을 잘 분류하는데 Margin이 크면 뭐가 좋을까?
      
        **Margin 최대화 → 구조적 위험 최소화**
        
        Equation 1)
        
        $$
        h \le min(\lceil {{R^2 \over \triangle^2}}\rceil, D) + 1
        $$
        
        Equation 2)
        
        $$
        R[f] \le R_{emp}[f] + \sqrt{{h{(ln{2n\over h}}+1)-ln({\delta \over 4}) }\over{n}}
        $$
        
        $notation$
        
        $h$: VC Dimension
        
        $R$: 모든 Input Vector들을 포함하는 가장 작은 구
        
        $\triangle$: Margin
        
        $D$: Input Space의 차원(= 변수 개수)
        
        - Equation 1)에서 D와 R은 Input Data가 주어지면 정해진 값으로, 변화 가능한 것은 $\triangle$ (Margin)
            - $\triangle$(Margin) 증가 → $R^2 \over \triangle^2$ 감소 →  $min(\lceil{R^2 \over \triangle^2}, D\rceil)$ 감소 → $h$  (VC Dimension) 감소
              
                → $\sqrt{{h{(ln{2n\over h}}+1)-ln({\delta \over 4}) }\over{n}}$ (Capacity Term) 감소 → $R[f]$ (구조적 위험) 감소
                

>>### Margin을 어떻게 계산할 것인가?

<p align = 'center'>
<image src=https://user-images.githubusercontent.com/56019094/199521970-0e80400c-3430-4f84-9e91-2f041f01882a.png height = '300'> </p>

Hyperplane을 $\boldsymbol{w}^T\boldsymbol{x} + b$

where $\boldsymbol{w} = (w_1,w_2)^T$ 라고 가정

- 벡터 $\boldsymbol{w}$는 이 Hyperplane과 수직인 법선 벡터
- $\boldsymbol{w}$에 대해 원점과의 거리가 $b$인 직선의 방정식은 $\boldsymbol{w}^T\boldsymbol{x} + b = 0$  ⇒ $w_1x_1 + w_2x_2 + b = 0$
- 위 직선의 기울기는 $- {w_1\over w_2}$이고, 법선 벡터 $\boldsymbol{w}$의 기울기는 $w_2 \over w_1$ ⇒ 두 직선은 직교

⇒ Plus-plane 위에 있는 벡터 $\boldsymbol{x}^+$와 Minus-plane 위에 있는 벡터 $\boldsymbol{x}^-$ 사이의 관계를 다음과 같이 정의 가능
- $\boldsymbol{x}^+ = \boldsymbol{x}^- + \lambda \boldsymbol{w}$
    - 위 수식은 $\boldsymbol{x}^-$를 $\boldsymbol{w}$ 방향으로 $\lambda$만큼 평행이동시킨다는 의미
- $\lambda$는 계산할 수 있을까?
  
    : $\boldsymbol{w}^T\boldsymbol{x}^+ + b = 1$
    
    → $\boldsymbol{w}^T(\boldsymbol{x}^- + \lambda\boldsymbol{w}) + b = 1$
    
    → $\boldsymbol{w}^T\boldsymbol{x}^- + b + \lambda \boldsymbol{w}^T\boldsymbol{w} = 1$          where, $(\boldsymbol{w}^T\boldsymbol{x}^- + b = 1)$
    
    →  $-1 + \lambda\boldsymbol{w}^T\boldsymbol{w} = 1$
    
    ⇒ $\lambda = {2 \over \boldsymbol{w}^T\boldsymbol{w}}$ 
    

한편, Margin은 Plus-plane과 Minus-plane 사이의 거리 $distance(\boldsymbol{x}^+, \boldsymbol{x}^-)$와 같음

$Margin = distance(\boldsymbol{x}^+, \boldsymbol{x}^-)$

$= ||\boldsymbol{x}^+ - \boldsymbol{x}^-||_2$

$= ||\boldsymbol{x}^- + \lambda\boldsymbol{w}- \boldsymbol{x}^-||_2$ , where, $\boldsymbol{x}^+ = \boldsymbol{x}^- + \lambda \boldsymbol{w}$

$= ||\lambda\boldsymbol{w}||_2$

$= \lambda \sqrt{\boldsymbol{w}^T\boldsymbol{w}}$

 $= {2 \over \boldsymbol{w}^T\boldsymbol{w}}\sqrt{\boldsymbol{w}^T\boldsymbol{w}}$ , where, $\lambda = {2 \over \boldsymbol{w}^T\boldsymbol{w}}$

$= {2 \over \sqrt{\boldsymbol{w}^T\boldsymbol{w}}}$

$= {2 \over ||w||_2}$

>## Optimization 문제

**Remind!** SVM의 목적은 **Margin**을 **최대**로 하는 Hyperplane을 찾는 것

>>### 목적 함수 및 제약 조건

- Margin을 최대화:  $max$  ${2 \over ||w||^2}$    --역수->     $min$   ${1 \over 2}||w||^2$
  
    $$min   {1 \over 2}||w||^2$$
    
    $$s.t.  \quad y_i(\boldsymbol{w}^T\boldsymbol{x}_i + b) \ge 1 \quad   , \forall i$$
    
    <p align = 'center'>
    <img src = https://user-images.githubusercontent.com/56019094/199522537-dcbdf18f-d3d0-4e16-8130-23c6f1a73f14.png height = '300'></p>
    
    - Let $\boldsymbol{x}_i$ = 파란색 Data Object, $\boldsymbol{x}_j$ = 빨간색 Data Object
        - $\boldsymbol{w}\boldsymbol{x}_i \ge 1$      $(y_i = +1)$  →   $y_i(\boldsymbol{w} \cdot \boldsymbol{x}_i + b) \ge +1$
        - $\boldsymbol{w}\boldsymbol{x}_j \le -1$  $(y_j = -1)$ →   $y_j(\boldsymbol{w} \cdot \boldsymbol{x}_j + b ) \ge +1$
        
        ⇒ $y$ = $\pm 1$인 경우 모두, 수식이 동일하게 위 제약 조건과 같이 정리됨 
            (Plus. SVM에서 Class Label을 0/1이 아닌 +1/-1로 설정한 이유) 
        
    

>>### 라그랑지안 문제로 변환

- 기존 목적 함수 및 제약 조건
  
    $$min \quad {1 \over 2}||w||^2$$
    
    $$s.t. \quad   y_i(\boldsymbol{w}^T\boldsymbol{x}_i + b) \ge 1 \quad   , \forall i$$
    
    ⇒ 위 식에서 $y_i, \boldsymbol{x}_i$는 주어진 값이고, $\boldsymbol{w}$와 $b$가 미지수 즉, 최적화 대상
    
- 라그랑지안 문제
  
    $$\min L_p\left(\mathbf{w}, b, \alpha_i\right)=\frac{1}{2}\|\mathbf{w}\|^2-\sum_{i=1}^N \alpha_i\left(y_i\left(\mathbf{w}^T \mathbf{x}_i+b\right)-1\right)$$  
    $$s.t.$   $\alpha_i \ge 0$$

>>### 쌍대(Dual) 문제로 변환

- KKT 조건
  
    $${\partial L_p \over \partial \boldsymbol{w}} = 0 \quad   ⇒  \quad \boldsymbol{w} = \sum_{i=1}^N {\alpha_iy_i\boldsymbol{x}_i}$$  
    
    $${\partial L_p \over  \partial b} = 0 \quad   ⇒  \quad  \sum_{i=1}^N {\alpha_iy_i} = 0$$
    
- 원문제
  
    $$\min L_p\left(\mathbf{w}, b, \alpha_i\right)=\frac{1}{2}\|\mathbf{w}\|^2-\sum_{i=1}^N \alpha_i\left(y_i\left(\mathbf{w}^T \mathbf{x}_i+b\right)-1\right)$$  
    $$s.t. \quad   \alpha_i \ge 0$$ 

- 쌍대(Dual) 문제
  
    $$\max L_D\left(\alpha_i\right)=\sum_{i=1}^N \alpha_i-\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$    
    
    $$s.t. \quad \sum_{i=1}^N \alpha_i y_i=0, \quad \alpha_i \geq 0$$
  
    
    
    
    - 위 쌍대 문제에서 $\boldsymbol{x}$와 $y$는 데이터로부터 주어진 값이고 $\alpha$만 미지수
    - 이때, KKT 조건에 따라 $\alpha_i(y_i(\boldsymbol{w}^T\boldsymbol{x}_i + b) -1) = 0$ 이라는 수식이 성립함
      
        ⇒ $\alpha_i = 0, (y_i(\boldsymbol{w}^T\boldsymbol{x}_i + b) -1) \ne 0$ 이거나 $\alpha_i \ne 0, (y_i(\boldsymbol{w}^T\boldsymbol{x}_i + b) -1) = 0$
        
        - $\alpha_i \ne 0, (y_i(\boldsymbol{w}^T\boldsymbol{x}_i + b) -1) = 0$인 경우,
          
            $(y_i(\boldsymbol{w}^T\boldsymbol{x}_i + b) -1) = 0$   →   $y_i(\boldsymbol{w}^T\boldsymbol{x}_i + b) = 1$이라는 것은 
            
            $\boldsymbol{x}_i$가 Plus-plane과 Minus-plane 상에 위치한다는 것을 의미
            → 이 $\boldsymbol{x}_i$ ( = Support Vector)에 대해서만 $\alpha_i$는 0보다 큰 값을 가지게 됨.
            
            <p align = 'left'><img src = https://user-images.githubusercontent.com/56019094/199523250-335b3594-beef-4cba-8598-fe7931fdf682.png height = '300'></p>
            이미지 출처: [https://techblog-history-younghunjo1.tistory.com/m/78]
            

>>### 최종 Hyperplane 구하기

- SVM에서 찾고자 하는 것은 Margin이 최대화된 Hyperplane $\boldsymbol{w}^T\boldsymbol{x} + b$
  
    → $\boldsymbol{w}$와 $b$를 찾으면 Hyperplane 구할 수 있음
    
- 이전 단계에서 아래와 같은 수식을 얻었음
  
    $$\boldsymbol{w} = \sum_{i=1}^N {\alpha_iy_i\boldsymbol{x}_i}$$
    
    - $\boldsymbol{x}_i, y_i$는 주어진 데이터로부터 알아낼 수 있는 값이므로 단 하나뿐인 미지수인 $\alpha$를 알면 $\boldsymbol{w}$를 찾아낼 수 있음
    - $\boldsymbol{w}$를 구한 뒤, $(y_i(\boldsymbol{w}^T\boldsymbol{x}_i + b) -1) = 0$을 통해 $b$를 구할 수 있음
- $\boldsymbol{x}_{new}$ (새로운 Instance)가 들어오면 위 수식에 넣어서 그 값이 0보다 크면 Class Label을 +1로, 값이 0보다 작으면 Class Label을 -1로 예측함

>## Soft Margin SVM

- 이전까지 설명한 SVM은 Hyperplane과 Support Vectors 사이에 Instance가 존재하지 않도록 하는 Hard Margin SVM이었음
- Soft Margin SVM은 Hyperplane과 Support Vectors 사이에 어느정도 Instance가 존재하는 것을 허용

>>### 목적 함수 및 제약 조건

$$min \quad {1 \over 2}||\boldsymbol{w}||^2 +C\sum_{i=1}^N \xi_i$$  

$$s.t. \quad y_i(\boldsymbol{w}^T\boldsymbol{x}_i + b) \ge 1-\xi_i, \quad \xi_i \ge0, \forall \quad i$$

<p align = 'left'><img src = https://user-images.githubusercontent.com/56019094/199524039-91704ab9-95e5-40be-a78e-83a4f7151e5a.png height = '250'></p>

$notation$

$C$: Penalty의 정도를 조절하는 Hyperparameter

$\xi$: Penalty

⇒ 미지수: $\boldsymbol{w}, b, \xi$

>>### 라그랑지안 문제로 변환

$$\min L_p\left(\mathbf{w}, b, \alpha_i\right)=\frac{1}{2}\|\mathbf{w}\|^2+C \sum_{i=1}^N \xi_i-\sum_{i=1}^N \alpha_i\left(y_i\left(\mathbf{w}^T \mathbf{x}_i+b\right)-1+\xi_i\right)-\sum_{i=1}^N \mu_i \xi_i$$  

$$s.t.\quad\alpha_i \ge 0$$

>>### 쌍대(Dual) 문제로 변환

- 원문제

$$\min L_p\left(\mathbf{w}, b, \alpha_i\right)=\frac{1}{2}\|\mathbf{w}\|^2+C \sum_{i=1}^N \xi_i-\sum_{i=1}^N \alpha_i\left(y_i\left(\mathbf{w}^T \mathbf{x}_i+b\right)-1+\xi_i\right)-\sum_{i=1}^N \mu_i \xi_i$$  


$$s.t.\quad\alpha_i \ge 0$$

- KKT 조건
  
    $${\partial L_p \over {\partial \boldsymbol{w}}} = 0 \quad ⇒  \quad   \boldsymbol{w} = \sum_{i=1}^n\alpha_iy_i\boldsymbol{x}_i$$  
    
    $${\partial L_p \over \partial b} = 0 \quad ⇒ \quad     \sum_{i=1}^n\alpha_iy_i = 0$$  
    
    $${\partial L_p \over \partial \xi_i} = 0 \quad ⇒ \quad    C - \alpha_i - \mu_i = 0$$
    

 $$ ⇒ \quad L_D = {1 \over 2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j\boldsymbol{x}_i\cdot\boldsymbol{x}_j + C\sum_i \xi_i -\sum_i\sum_j\alpha_i\alpha_jy_iy_j\boldsymbol{x}_i\cdot\boldsymbol{x}_j - b\sum_i\alpha_iy_i + \sum_i\alpha_i - \sum_i\alpha_i\xi_i - \sum_i\mu_i\xi_i$$ 

 $$ → \quad \sum_i(C-\alpha_i-\mu_i)\xi_i = 0, \quad \sum_i\alpha_iy_i = 0$$  

 $$ → \quad L_D = {1 \over 2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j\boldsymbol{x}_i\cdot\boldsymbol{x}_j -\sum_i\sum_j\alpha_i\alpha_jy_iy_j\boldsymbol{x}_i\cdot\boldsymbol{x}_j + \sum_i\alpha_i$$   

 $$ → \quad L_D = \sum_i\alpha_i  - {1 \over 2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j\boldsymbol{x}_i\cdot\boldsymbol{x}_j$$

 $$ ⇒ \quad L_D({\alpha_i}) = \sum_{i=1}^N\alpha_i  - {1 \over 2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\boldsymbol{x}_i^T\boldsymbol{x}_j$$

- 쌍대 문제

$$max\quad L_D({\alpha_i}) = \sum_{i=1}^N\alpha_i  - {1 \over 2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\boldsymbol{x}_i^T\boldsymbol{x}_j$$    


$$s.t.\quad \sum_{i=1}^N \alpha_iy_i = 0, 0 \le \alpha_i \le C$$



>>### Plus) $\alpha_i$ 값에 따른 Instance 위치

KKT 조건으로부터 $\alpha_i(y_i(\boldsymbol{w}^T\boldsymbol{x} + b)-1+\xi_i) = 0$ 수식을 얻을 수 있었음
- Support Vector에 대해서만 $\alpha_i \ne 0$이 성립

또한 $C - \alpha_i - \mu_i = 0, \mu_i\xi_i = 0$이라는 수식이 성립함
- **Case 1)** $\alpha_i = 0\quad$⇒ Support Vector가 아닌 Instance
- **Case 2)** $0<\alpha_i < C\quad$
  →  $\mu_i < C$이면 $C - \alpha_i - \mu_i = 0$이 성립하기 위해 $\mu_i > 0$ 이어야 함. 
  → $\mu_i > 0$이라면 $\mu_i\xi_i = 0$이 성립하기 위해 $\xi_i = 0$이어야 함
  
    → $y_i(\boldsymbol{w}^T\boldsymbol{x} + b)-1 = 0$인 Instance
  
    ⇒ Margin 위에 위치하는 Support Vector
  
- **Case 3)** $\alpha_i = C \quad$
  → $C - \alpha_i - \mu_i = 0$에서 $\alpha_i = C$라면 $\mu_i = 0$
  
    → $\mu_i = 0$이라면 $\xi_i > 0$
  
    ⇒ Margin 밖에 위치하는 Support Vector
  
    <p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199524486-539f4d6a-37fe-4882-bbf7-97188c7d06d2.png height = '300'></p>
  

Hyperparameter C(오분류 비용)에 따른 분류 경계면 변화

<p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199524779-72b14258-12fb-4d6b-a601-8bf87d5f069d.png height = '300'></p>

$$
min \quad {1 \over 2}||\boldsymbol{w}||^2 +C\sum_{i=1}^N \xi_i
$$

Large C: 목적함수에서 Penalty가 더 큰 영향력을 가짐   
    → Penalty를 줄이는 방향으로 학습이 진행  
    → Margin이 좁고, $\alpha_i = C$인 Support Vector의 수가 상대적으로 적음  

Small C: 목적함수에서 Penalty의 영향력이 작아짐   
      → Penalty의 영향력이 적으므로 Margin을 조금 더 넓게 잡을 수 있음  
      → $\alpha_i = C$인 Support Vector의 수가 상대적으로 많음  

>## Nonlinear&Kernel

- Linear Model의 한계: 분류 경계면이 비선형일 경우 잘 찾아내지 함
  
    <p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199524989-6ae5a362-d1ba-4a01-b749-1e0be33d5296.png height = '300'></p>
    


🧐 선형 분류가 가능한 고차원으로 데이터를 Mapping해서 모델을 학습하자!

<p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199525195-e5ee6860-9c28-48cf-a3fd-f86223d3b91c.png height = '300'></p>
이미지 출처: [https://towardsdatascience.com/support-vector-machine-formulation-and-derivation-b146ce89f28]

⇒ 고차원 Mapping을 통해 Nonlinear(비선형) 분류 경계면 생성

>>### 고차원에서의 목적 함수 및 제약 조건

$$min\quad{1 \over 2}||\boldsymbol{w}||^2 + C\sum_{i=1}^N\xi_i$$  


$$s.t\quad y_i(\boldsymbol{w}^T\Phi(\boldsymbol{x}_i) + b) \ge 1-\xi_i,\quad \xi_i \ge0, \quad\forall i$$

⇒ **라그랑지안 문제로 변환**

$$\min L_p\left(\mathbf{w}, b, \alpha_i\right)=\frac{1}{2}\|\mathbf{w}\|^2+C \sum_{i=1}^N \xi_i-\sum_{i=1}^N \alpha_i\left(y_i\left(\mathbf{w}^T \Phi\left(\mathbf{x}_i\right)+b\right)-1+\xi_i\right)-\sum_{i=1}^N \mu_i \xi_i$$  
- KKT 조건
  
    $${\partial L_P \over \partial w} = 0\quad ⇒ \quad \boldsymbol{w} = \sum_{i=1}^n\alpha_iy_i\Phi(\boldsymbol{x}_i)$$  
    
    $${\partial L_P \over \partial b} = 0\quad ⇒ \quad \sum_{i=1}^n\alpha_iy_i = 0$$  
    
    $${\partial L_P \over \partial \xi_i} = 0\quad ⇒ \quad C - \alpha_i - \mu_i = 0$$
    

⇒ 쌍대(Dual) 문제로 변환

$$max\quad L_D({\alpha_i}) = \sum_{i=1}^N\alpha_i  - {1 \over 2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\Phi(\boldsymbol{x}_i)^T\Phi(\boldsymbol{x}_j)$$  

$$s.t.\quad \sum_{i=1}^N \alpha_iy_i = 0, \quad 0 \le \alpha_i \le C$$

😓 고차원으로 Mapping시키는 함수 $\Phi$를 어떻게 찾을까,,,?

👍🏻 Kernel Trick을 쓰자!

$$max\quad L_D({\alpha_i}) = \sum_{i=1}^N\alpha_i  - {1 \over 2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\Phi(\boldsymbol{x}_i)^T\Phi(\boldsymbol{x}_j)$$

  $$ ⇒ max\quad L_D({\alpha_i}) = \sum_{i=1}^N\alpha_i  - {1 \over 2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j K(\boldsymbol{x}_i, \boldsymbol{x}_j)$$

### Kernel Trick

$$max\quad L_D({\alpha_i}) = \sum_{i=1}^N\alpha_i  - {1 \over 2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\Phi(\boldsymbol{x}_i)^T\Phi(\boldsymbol{x}_j)$$  
위 수식에서와 같이 고차원에서는 항상 $\Phi({\boldsymbol{x}_i})^T\Phi(\boldsymbol{x}_j)$와 같이 벡터의 내적 형태로 존재

→ 저차원 데이터를 입력 받아서 고차원 공간상에 내적 결과값을 줄 수 있다면 굳이 $\Phi$를 찾지 않아도 된다!

<p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199525449-05428317-64ad-4ac0-bbc5-266015f5d493.png height = '300'></p>

- 유효한 Kernel 함수의 조건
    - Symmetric Matrix
    - Positive semi-definite Matrix

- 대표적인 Kernel 함수
    - Polynomial
      
        $K(x,y) = (x \cdot y + c)^d,\quad c>0$
        
    - Gaussian (RBF)
      
        $K(x,y) = exp(-{||x-y||^2 \over 2\sigma^2}),\quad \sigma \ne 0$
        
    - Sigmoid
      
        $K(x,y) = tanh(a(x\cdot y)+b),\quad a,b\ge0$
    
- Kernel 형태에 따른 분류 경계면
    - Linear Kernel: 선형 분류 경계면만 생성 가능
    - Non-linear Kernel: 복잡한 형태의 분류 경계면 생성 가능
    
    
    <p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199525684-704acd32-74e6-4432-85ab-8d1c778c7dff.png height = '500'></p>
    이미지 출처: [https://towardsdatascience.com/multiclass-classification-with-support-vector-machines-svm-kernel-trick-kernel-functions-f9d5377d6f02]



># **코딩 실습**

## 실험 주제
- Support Vector Machine에서 주로 사용되는 Kernel Function인 (Linear), Sigmoid, Poly, RBF **Kernel 별로 SVM 모델 수립에 소요되는 시간**에 유의미한 차이가 존재하는지 확인
- 해당 주제 선정 배경  
    1. Stackoverflow, Github 등 소스 코드 공유 사이트에는 Support Vector Machine 관련 실습 코드 중 Kernel, C 등의 하이퍼 파라미터 변경에 따른 모델의 성능 차이를 확인하는 경우는 다수 존재
    2. Kernel Function 별 모델 수립 시간에 소요되는 시간을 비교한 실험은 매우 드물었음. 
    3. Kernel Function 수식을 보면 $d$승 계산이 포함되는 Polynomial Kernel Function이 가장 학습에 오래 소요되고 RBF, Sigmoid, Linear이 후순일 것이라 추측되었으나, 이에 대해 실험을 통해 확인하고자 함

- ❗️ Remind  
    - **Polynomial**
      
        $K(x,y) = (x \cdot y + c)^d,\quad c>0$
        
    - **Gaussian (RBF)**
      
        $K(x,y) = exp(-{||x-y||^2 \over 2\sigma^2}),\quad \sigma \ne 0$
        
    - **Sigmoid**
      
        $K(x,y) = tanh(a(x\cdot y)+b),\quad a,b\ge0$
    
    - Kernel 형태에 따른 분류 경계면  
      - Linear Kernel: 선형 분류 경계면만 생성 가능
      - Non-linear Kernel: 복잡한 형태의 분류 경계면 생성 가능

- 실험 내용  
    1. Support Vector Classifier를 중심으로 Kernel 별 모델 수립 시간 비교  
    2. 사이킷런 데이터셋에 존재하는 make_moons, make_gaussian_quantiles 을 이용  
        - make_moons data: Binary Class로 구성되어 있으며 각 Label 별로 Data Instance가 초승달 형태로 분포함
        ```python
        # 데이터셋 이해를 위한 Moon datasets 시각화
        X,y = make_moons(n_samples = 3000, noise = 0.2, random_state = 1002)
        plt.scatter(X[:,0],X[:,1], marker = "o", c = y, s = 80, edgecolor = 'k', linewidth = 1)
        plt.xlabel("$X_1$")
        plt.ylabel("$X_2$")
        plt.show()
        ```
        <p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199547509-7c23f09f-caef-418f-af3b-d8788cac4016.png height = '300'> </p>  
        

        - make_gaussian_quantiles: Binary Class로 설정했으며 2차원 상에서 각 Label 별로 Data Instance가 등고선 형태로 분포
        ```python
        # 데이터셋 이해를 위한 Gaussian Quantile datasets 시각화 
        # make_gaussian_quantiles()는 독립 변수의 개수를 n_features를 이용해 설정 가능
        X,y = make_gaussian_quantiles(n_samples = 400, n_features = 2, n_classes = 2, random_state = 42)  # 
        plt.scatter(X[:,0],X[:,1], marker = 'o', c = y, s = 60, edgecolor = 'k', linewidth = 1)
        plt.xlabel("$X_1$")
        plt.ylabel("$X_2$")
        plt.show()
        ```
        <p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199548436-2f62154c-3053-42d1-a33a-adcbccc447c6.png height = '300'></p>
        
    3. 2. 실험 과정에서 각 Kernel 별 분류 성능(Accuracy, Recall, Precision, F1-Score) 및 모델 수립에 소요된 시간 측정
    4. make_gaussian_quantiles 데이터에서 독립 변수의 개수(n_features)를 늘려가며 각 Kernel Function 별 모델 수립에 소요된 시간 측정  
    (계기: 실험 중 make_gaussian_quantiles()의 n_features에 임의의 숫자를 대입해 학습 시간을 확인하는 과정에서 Kernel Function에 따라 일관된 양상을 보일 것으로 예상했으나 이와 다른 결과를 확인함)
    5. 추가적으로 SVR에서도 make_regression() 함수를 이용해 n_features를 늘려가며 Kernel Function에 따른 학습 시간 차이가 존재하는지 실험함


## *Main Experiment - Support Vector Classifier*
```python
# 코드 실습에 필요한 패키지 및 메소드 Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings

from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_gaussian_quantiles
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings(action='ignore') 
```
- 실험에 사용되는 함수 정의
```python
# Classfication 결과 출력 함수
def metric(model, test_X, test_y):
    preds = model.predict(test_X) # 수립된 모델의 예측값 (Binary)
    acc = accuracy_score(test_y,preds) # Accuracy
    f1 = f1_score(test_y,preds,average='macro') # F1-score
    precision = precision_score(test_y,preds,average='macro') # Precision
    recall  = recall_score(test_y,preds,average='macro') # Recall
    return [acc,precision,recall,f1]
```
```python
# Kernel 변경에 따른 SVM 모델 수립
def fit_kernel_svm(X,y):
    # 가장 널리 사용되는 Kernel Function인 Polynomial, RBF, Sigmoid와 일반 Linear SVM을 실험 대상으로 적용
    kernels = ['linear','poly','rbf','sigmoid'] 
    kernel_svms = []
    times = []

    for kernel in kernels:
        start_time = time.time() # 모델 수립에 소요되는 시간 측정을 위함
        
        # 'linear_clf', 'poly_clf'와 같이 각기 다른 커널을 사용한 SVC 모델 객체를 변수로 저장
        # 아래와 같이 globals()['{}'.format()] 형태를 이용하면 동적 변수 생성 가능
        globals()['{}_clf'.format(kernel)] = SVC(kernel = kernel).fit(X,y) 
        times.append(time.time()-start_time)
        kernel_svms.append(globals()['{}_clf'.format(kernel)]) # 각 Kernel Function 별 수립된 SVC 모델을 리스트에 저장
    
    return kernel_svms,times, kernels
```
```python
# Support Vector Classifier 시각화 함수
def svc_plot(X,y, svm_model):
    assert X.shape[1] == 2, "input X's Num of Feature should be 2" # 2차원 시각화를 위해 Input X의 변수 개수가 2개인 경우만 허용
    plt.title(f'{svm_model} Scatter Plot')
    plt.scatter(X[:,0],X[:,1], marker = 'o', c = y, cmap = plt.cm.Paired, edgecolors= 'k') # Data Instance 시각화

    # 초평면 시각화
    ax = plt.gca()    
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xx = np.linspace(xlim[0],xlim[1], 10)
    yy = np.linspace(ylim[0],ylim[1], 10)
    XX, YY = np.meshgrid(xx, yy)

    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    z = svm_model.decision_function(xy).reshape(XX.shape) # 분류 경계면 (Hyperplane)
    ax.contour(XX, YY, z, colors = ['k','r','k'], levels = [-1, 0, 1], alpha = 0.6, linestyles = ['--','-','--'])
    
    return plt.show()
```

```python
# 각 Kernel SVC의 결과(데이터 프레임 및 그래프)를 보여주는 함수
def show_kernel_svc_result(X, y):
    train_X, test_X, train_y, test_y = train_test_split(X, y, shuffle = True, test_size = 0.25)
    # 각 Kernel 별 SVC 모델 수립 + 각 Kernel 별 SVC 수립된 모델(kernel_svms)과 모델 수립에 소요된 시간(times), Kernel들의 명칭(kernels) 저장
    kernel_svms, times, kernels = fit_kernel_svm(train_X, train_y) 
    k_svms, acc, precision, recall, f1 = [], [], [], [], [] # Kernel 별 성능을 DataFrame 형태로 보여주기 위해 빈 리스트 생성

    for i, k_s in enumerate(kernel_svms):
        k_svms.append(kernels[i]) # kernels[i] 예시: 'linear' (str)
        acc.append(metric(k_s,test_X,test_y)[0]) # Classifier의 Accuracy 
        precision.append(metric(k_s,test_X,test_y)[1]) # Classifier의 Precision
        recall.append(metric(k_s,test_X,test_y)[2]) # Classifier의 Recall
        f1.append(metric(k_s,test_X,test_y)[3]) # Classifier의 F1 score

    # 각 커널별 SVC의 분류 성능 및 모델 수립에 소요된 시간을 데이터 프레임 형태로 저장
    k_svm_result = pd.DataFrame({'Kernel':k_svms, 'Accuracy':acc, 'Precision':precision, 'Recall':recall, 'F1 Score':f1, 'Time for Train(s)': times})
    k_svm_result.iloc[:,1:] = k_svm_result.iloc[:,1:].apply(lambda x:np.round(x,4)) # Accuracy, Precision, Recall, F1-score, Time for Train을 소수점 넷째자리에서 반올림 (꼭 필요한 과정은 아님)
    
    # Kernel 별 F1 Score 및 모델 수립 소요 시간 시각화
    fig, axes = plt.subplots(ncols = 2)
    fig.set_size_inches((8,5))
    fig.subplots_adjust(wspace = 0.3)

    axes[0].plot(k_svm_result['Kernel'], k_svm_result['F1 Score'], marker = 'o', color = 'blue')
    axes[0].set_title('F1 Score')
    axes[1].plot(k_svm_result['Kernel'], k_svm_result['Time for Train(s)'], marker = 'o', color = 'red')
    axes[1].set_title('Time for Train(s)')

    display(k_svm_result)
    plt.show()
```
- 위 사용자 정의 함수 코드 결과 예시
```python
# moon dataset 적용 예시
kernel_svms, times, kernels = fit_kernel_svm(X, y)

for k_s in kernel_svms:
    svc_plot(X,y,k_s)
```

<center class="half">
    <img src = "https://user-images.githubusercontent.com/56019094/199554240-0b8cd1e1-691e-46f0-bcfc-71ed7f774a9b.png" width = 400>
    <img src = "https://user-images.githubusercontent.com/56019094/199554255-46b85b76-916c-4516-b906-b992cfbefa2d.png" width = 400>
<figure>

<figure class="half">
    <img src = https://user-images.githubusercontent.com/56019094/199554282-46458a8e-21fd-474e-8e57-868cd0c560d6.png width = 400>
    <img src = https://user-images.githubusercontent.com/56019094/199554294-f908ba20-d081-48b3-90cf-0b981075ecc3.png width = 400>
</center>


### SVC 결과 및 해석
```python
# Moon Dataset 적용 결과 확인
X, y = make_moons(n_samples = 5000, noise = 0.2, random_state = 1002)
show_kernel_svc_result(X,y)
```	
<center>

|Kernel|Accuracy|Precision|Recall|F1 Score|Time for Train(s)|
|------|--------|---------|------|--------|-----------------|
linear|	0.8728	| 0.8728| 	0.8728|	0.8728|	0.0556|
poly |	0.9008	| 0.9084| 	0.8998|	0.9002|	0.0584|
rbf	| 0.9752	| 0.9752| 	0.9752|	0.9752|	0.0204|
sigmoid	| 0.6184 |	0.6183| 0.6183|	0.6183|	0.0935|

</center>

<p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199560207-277305dc-d1d7-4a7b-8151-3a084a76a2cf.png height = 300></p>

- 독립 변수 2개, Data Instance 수 5000개인 상황에서는 기존 예상과 달리 모델 수립에 소요된 시간이 Sigmoid > Polynomial > Linear > RBF 순으로 결과가 나왔음  
- RBF의 경우 모델 수립에 가장 적은 시간이 소요되었음에도 불구하고 F1 Score 기준으로 가장 높은 성능(약 0.98)을 보이며 "Kernel SVM을 적용할거라면 RBF를 우선 적용해보라"는 기존 관행을 뒷받침함
- Polynomial의 경우 모델 수립에 가장 많은 시간(RBF의 네 배 이상)이 소요되었음에도 불구하고 F1 Score 기준 가장 낮은 성능을 보였음

```python
# Gaussian Quantiles 데이터셋 적용 결과 확인
X, y = make_gaussian_quantiles(n_samples = 5000, n_features = 2, n_classes = 2, random_state = 42)
show_kernel_svc_result(X,y)
```
<center>

|Kernel|Accuracy|Precision|Recall|F1 Score|Time for Train(s)|
|------|--------|---------|------|--------|-----------------|
|linear|	0.5872|	0.6277|	0.5872|	0.5516|	0.1590|
|poly	|0.5144|	0.7537|	0.5144|	0.3646|	0.3005|
rbf|	0.9960	|0.9960|	0.9960|	0.9960|	0.0209|
sigmoid|	0.5328|	0.5328|	0.5328|	0.5328|	0.1886|

</center>

<p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199562004-620c57e1-8129-4cf0-8820-07d4d5e55579.png height = 300></p>
- 독립 변수 2개, Data Instance 수 5000개인 상황에서 이 데이터셋 역시 예상과 다른 결과를 보였으며, make moon 데이터셋과도 다른 결과를 보였음
- Make Moon 데이터셋의 경우 [Sigmoid > Polynomial > Linear > RBF] 순서였으나 Gaussian Quantiles 데이터셋의 경우 [Polynomial > Sigmoid > Linear > RBF] 순서
- Polynomial이 모델 수립 소요 시간이 가장 오래 걸린 것은 예상과 동일하나 RBF가 그 다음으로 오래 걸릴 것이라는 예상과 달리 Make Moon Dataset과 Gaussian Quantiles 데이터셋 역시 RBF는 모델 수립 소요 시간이 가장 짧았으며 F1 Score 기준으로 가장 높은 성능을 보이기까지 함
- 이 경우에도 Polynomial은 모델 수립 소요 시간이 가장 오래 걸렸으나 F1 Score 기준 가장 낮은 성능을 보였음  

> ####  독립 변수 개수를 증가(1개 -> 100개)시키면서 Kernel Function 별 모델 수립 소요 시간 변화 체크
---
* 해당 실험 진행 이유
: Support Vector Regressor 이용 실험 중 "Kernel Function 수식을 보니 독립 변수 개수가 증가하면 모델 수립 소요 시간이 어떻게 변할까"라는 의문을 바탕으로 독립 변수 개수를 [3, 6, 12, 24, 48]로 늘리며 실험을 해보니 Kernel Function 별 모델 수립 소요 시간 순위가 변경되는 현상을 발견함

<center>

독립 변수 3개
<p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199635510-dd217d8f-2c8b-4578-b02d-39e7b419f382.png height = 250 ></p>

독립 변수 6개
<p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199635778-db4773a0-4bef-4675-9d5c-0298216a63ed.png height = 250></p>

독립 변수 12개
<p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199635813-d348bd25-e594-464f-a6e0-3a812e9a9cda.png height = 250></p>

독립 변수 24개
<p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199635853-2ee21a27-afb0-4cbd-b80d-cf1122fe8822.png height = 250> </p>

독립 변수 48개
<p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199635894-597e423c-dadd-427d-a76e-009fc731915f.png height = 250></p>

</center>

---
- 해당 실험에서 사용된 코드는 이전에 사용되었던 코드를 기반으로 약간의 변형만 적용되어서 Tutorial 내용에는 포함하지 않음 (ipynb 파일 참고)
<p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199563451-993a7962-a18c-4d33-8de9-bb21d2834e45.png></p>



- 독립 변수 개수를 1개에서 100개까지 증가시키면서, 각 Kernel Function 별 독립 변수 개수 별로 모델 수립 소요 시간이 **최장 시간**이 걸렸던 경우 Count

<center>

|Sigmoid|RBF|Polynomial|Linear|
|-------|---|----------|------|
|39|29|17|15|

</center>



- 독립 변수 개수를 1개에서 100개까지 증가시키면서, 각 Kernel Function 별 독립 변수 개수 별로 모델 수립 소요 시간이 **최단 시간**이 걸렸던 경우 Count

<center>

|Polynomial|Linear|RBF|Sigmoid|
|-------|---|----------|------|
|35|33|20|12|

</center>



- 독립 변수 개수를 증가시킨다는 새로운 상황에서 Kernel Function 별 모델 학습에 소요되는 시간이 어떻게 변화하는지 확인하기 위해 실험함
- 수식에 $degree$ 승이 포함되는 Polynomial의 경우 지수적으로 증가할 것이라 예상했으나 독립 변수가 100개까지 증가하는 상황에서도 모든 Kernel Function이 선형적으로 증가함을 확인할 수 있었음.
- 네 가지 Kernel Functio의 경우 모두 독립 변수의 개수가 증가함에 따라 모델 수립 소요 시간의 분산이 커지는 것으로 확인됨
- 내적 연산이 수식에 포함되는 Polynomial과 Sigmoid의 경우 모델 수립 소요 시간이 유사할 것이라 생각했으나, 각각 가장 많이 최단 시간을 기록한 커널과 가장 많이 최장 시간을 기록한 커널이었음
- Sigmoid의 수식에는 $tanh$가 포함되고, Polynomial의 경우 $degree$승이 포함되어, 이 둘을 수식만 보았을 때에는 Polynomial이 더 오랜 시간이 걸릴 것이라 판단했으나 상반된 결과가 나와서 흥미로웠음. 이러한 결과가 나온 이유에 대해서는 데이터셋 변경을 통한 추가 실험 및 시간 복잡도에 대한 개념을 추가적으로 공부해야 이해할 수 있을 것으로 판단됨.

## *Additional Experiment - Support Vector Regressor*
- 해당 실험에서 사용된 코드는 이전에 사용되었던 SVC 코드를 기반으로 약간의 변형(SVC -> SVR 등)만 적용되어서 Tutorial 내용에는 포함하지 않음 (ipynb 파일 참고)
- 데이터셋 생성에는 사이킷런의 make_regression 메소드를 사용했음 (n_features를 통해 독립 변수 개수를 자유롭게 조정 가능함)
- SVC의 마지막 실험과 동일하게 독립 변수 개수를 1개부터 100개까지 증가시키면서 Kernel Function(Linear, Polynomial, RBF, Sigmoid) 별 모델 수립 소요 시간 변화 실험

### SVR 결과 및 해석
<p align = 'center'><img src = https://user-images.githubusercontent.com/56019094/199567437-870bbdf6-564a-485b-a6b0-6ec7d33a6e98.png></p>

- 독립 변수 개수를 1개에서 100개까지 증가시키면서, 각 Kernel Function 별 독립 변수 개수 별로 모델 수립 소요 시간이 **최장 시간**이 걸렸던 경우 Count

<center>

|Sigmoid|Linear|Polynomial|RBF|
|-------|---|----------|------|
|34|31|19|16|

</center>


- 독립 변수 개수를 1개에서 100개까지 증가시키면서, 각 Kernel Function 별 독립 변수 개수 별로 모델 수립 소요 시간이 **최단 시간**이 걸렸던 경우 Count


<center>

|Polynomial|RBF|Linear|Sigmoid|
|-------|---|----------|------|
|32|27|23|18|

</center>

- 독립 변수의 개수가 증가함에 따라 각 Kernel Function 별 모델 수립 소요 시간의 분산이 커지는 것은 SVC와 동일하나 분산의 크기가 더 큼
- 첫 번째 표를 보면, SVC의 결과와 동일하게 Sigmoid가 가장 많이 최장 모델 수립 소요 시간을 기록했음
- 두 번째 표를 보면, SVC의 결과와 동일하게 Polynomial이 가장 많이 최단 모델 수립 소요 시간을 기록했음

## 결론
- 실험 전 예상: [Linear < Sigmoid < RBF < Polynomial] 순으로 모델 수립에 많은 시간이 소요될 것
- SVC 결과(최단 시간 결과 표 기반): [Polynomial < Linear < RBF< Sigmoid]
- SVR 결과(최단 시간 결과 표 기반): [Polynomai < RBF < Linear < Sigmoid]
- 기존 예상과 반대로 Polynomial이 모델 학습에 가장 적은 시간이 소요되었으며 Sigmoid가 가장 많은 시간이 소요되었음
- 그러나 모델 성능을 고려했을 때에는 SVC와 SVR 모두 RBF가 대부분 가장 우수한 성능을 보였음 (SVR의 경우 해당 실험 파트는 코드에 포함되어 있지 않음)
- Polynomial Kernel SVM이 모델 학습 소요 시간 측면에서는 장점이 있으나 성능을 고려했을 때에는 다른 Kernel Function에 비해 성능이 낮음
- 여러 가지 Kernel Function을 수립해보기에 시간이 부족한 경우에는 RBF Kernel SVM을 우선적으로 시도해보는 것이 합리적 (사이킷런 SVM의 Kernel Default가 RBF인 이유도 이러한 배경이 있었을 것이라 유추됨)
- 최종 결론: **Kernel 별로 SVM 모델 수립에 소요되는 시간** 차이가 존재하는 것으로 판단. Polynomial의 가장 적은 시간이 소요되었고, Sigmoid가 가장 많은 시간이 소요되었음

- 한계점
    - SVC의 경우 두 가지 데이터셋(독립 변수 개수를 증가시키면서 모델 수립 소요 시간을 확인한 실험에서는 한 가지 데이터셋만 사용), SVR의 경우 한 가지 데이터셋만을 사용하여 실험함
    - Kernel Function 별 수식과 실험 결과와의 연관성을 정확하게 해석해내지 못했음
