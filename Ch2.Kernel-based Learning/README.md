# SVM

[TOC]

**S**uppor **V**ector **M**achine

> Keywords: Margin, Hyperplane, Support Vector
> 

<aside>
📢 요약: Support Vector Machine은 Vector Space 상에서 Vector들을 가장 잘 분류하는 Hyperplane을 수립하는 것을 목표로 한다.
- Background
    - Hyperplane(초평면): a subspace of one dimension less than its ambient space
      
        ![이미지 출처: Support Vector Machines without tears](SVM%20aeb6168f959e49e2a6652261fd99d5e4/Untitled.png)
        
        이미지 출처: Support Vector Machines without tears
        
        - n차원의 공간에서 Hyperplane은 n-1차원의 subspace를 의미
            - 2차원의 경우 Hyperplane은 1차원(직선)
            - 3차원의 경우 Hyperplane은 2차원(평면)
        
        ⇒ SVM에서 Hyperplane은 어떤 Vector Space 상에 존재하는 Vector들을 분류하는 Decision Boundary(결정 경계)에 해당
        

# Margin

### “가장 잘 분류하는”의 기준이 무엇인가?

- SVM은 Vector Space 상에 있는 Vector 형태로 표현된 각 Data Point들을 가장 잘 분류하는 Hyperplane(초평면)을 수립하는 것을 목표로 함
  
    ![Untitled](SVM%20aeb6168f959e49e2a6652261fd99d5e4/Untitled%201.png)
    

### Margin을 최대화하는 Hyperplane을 찾자

- Margin이란?
  
    : Hyperplane으로부터 등 간격으로 양쪽으로 확장시켰을 때 Hyperplane과 가장 가까운 객체(Support Vector)와의 거리
    
    ![Untitled](SVM%20aeb6168f959e49e2a6652261fd99d5e4/Untitled%202.png)
    
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
                

### Margin을 어떻게 계산할 것인가?

![Untitled](SVM%20aeb6168f959e49e2a6652261fd99d5e4/Untitled%203.png)

- Hyperplane을 $\boldsymbol{w}^T\boldsymbol{x} + b$
  
    where $\bold{w} = (w_1,w_2)^T$ 라고 가정
    
    - 벡터 $\bold{w}$는 이 Hyperplane과 수직인 법선 벡터
    - $\bold{w}$에 대해 원점과의 거리가 $b$인 직선의 방정식은 $\bold{w}^T\bold{x} + b = 0$  ⇒ $w_1x_1 + w_2x_2 + b = 0$
    - 위 직선의 기울기는 $- {w_1\over w_2}$이고, 법선 벡터 $\bold{w}$의 기울기는 $w_2 \over w_1$ ⇒ 두 직선은 직교
- ⇒ Plus-plane 위에 있는 벡터 $\bold{x}^+$와 Minus-plane 위에 있는 벡터 $\bold{x}^-$ 사이의 관계를 다음과 같이 정의 가능
    - $\bold{x}^+ = \bold{x}^- + \lambda \bold{w}$
        - 위 수식은 $\bold{x}^-$를 $\bold{w}$ 방향으로 $\lambda$만큼 평행이동시킨다는 의미
    - $\lambda$는 계산할 수 있을까?
      
        : $\bold{w}^T\bold{x}^+ + b = 1$
        
        → $\bold{w}^T(\bold{x}^- + \lambda\bold{w}) + b = 1$
        
        → $\bold{w}^T\bold{x}^- + b + \lambda \bold{w}^T\bold{w} = 1$          where, $(\bold{w}^T\bold{x}^- + b = 1)$
        
        →  $-1 + \lambda\bold{w}^T\bold{w} = 1$
        
        ⇒ $\lambda = {2 \over \bold{w}^T\bold{w}}$ 
        

- 한편, Margin은 Plus-plane과 Minus-plane 사이의 거리($distance(\bold{x}^+, \bold{x}^-)$)와 같음
    - $Margin = distance(\bold{x}^+, \bold{x}^-)$
      
                     $= ||\bold{x}^+ - \bold{x}^-||_2$
        
                     $= ||\bold{x}^- + \lambda\bold{w}- \bold{x}^-||_2$             where, $\bold{x}^+ = \bold{x}^- + \lambda \bold{w}$
        
                     $= ||\lambda\bold{w}||_2$
        
                     $= \lambda \sqrt{\bold{w}^T\bold{w}}$
    
    ​                    $= {2 \over \bold{w}^T\bold{w}}\sqrt{\bold{w}^T\bold{w}}$                          where, $\lambda = {2 \over \bold{w}^T\bold{w}}$
    
    ​                    $= {2 \over \sqrt{\bold{w}^T\bold{w}}}$
    
    ​                     $= {2 \over ||w||_2}$

# Optimization 문제

**Remind!** SVM의 목적은 **Margin**을 **최대**로 하는 Hyperplane을 찾는 것

### 목적 함수 및 제약 조건

- Margin을 최대화:  $max$  ${2 \over ||w||^2}$    --역수->     $$$min$   ${1 \over 2}||w||^2$
  
    $min$   ${1 \over 2}||w||^2$
    
    $s.t.$   $y_i(\bold{w}^T\bold{x}_i + b) \ge 1$   , $\forall i$
    
    ![Untitled](SVM%20aeb6168f959e49e2a6652261fd99d5e4/Untitled%204.png)
    
    - Let $\bold{x}_i$ = 파란색 Data Object, $\bold{x}_j$ = 빨간색 Data Object
        - $\bold{w}\bold{x}_i \ge 1$      $(y_i = +1)$  →   $y_i(\bold{w} \cdot \bold{x}_i + b) \ge +1$
        - $\bold{w}\bold{x}_j \le -1$  $(y_j = -1)$ →   $y_j(\bold{w} \cdot \bold{x}_j + b ) \ge +1$
        
        ⇒ $y$ = $\pm 1$인 경우 모두, 수식이 동일하게 위 제약 조건과 같이 정리됨 
            (Plus. SVM에서 Class Label을 0/1이 아닌 +1/-1로 설정한 이유) 
        
    

### 라그랑지안 문제로 변환

- 기존 목적 함수 및 제약 조건
  
    $min$   ${1 \over 2}||w||^2$
    
    $s.t.$   $y_i(\bold{w}^T\bold{x}_i + b) \ge 1$   , $\forall i$
    
    ⇒ 위 식에서 $y_i, \bold{x}_i$는 주어진 값이고, $\bold{w}$와 $b$가 미지수 즉, 최적화 대상
    
- 라그랑지안 문제
  
    ${\min{ L_{p}(\bold{w},b,{ \alpha  }_{ i }) }  } =\frac { 1 }{ 2 } { \left\| \bold{w} \right\|  }^{ 2 }-\sum _{ i=1 }^{ N }{ { \alpha  }_{ i }({ y }_{ i }({ \bold{w} }^{ T }{ \bold{x} }_{ i }+b)-1) }$
    $s.t.$   $\alpha_i \ge 0$

### 쌍대(Dual) 문제로 변환

- KKT 조건
  
    ${\partial L_p \over \partial \bold{w}} = 0$   ⇒  $\bold{w} = \sum_{i=1}^N {\alpha_iy_i\bold{x}_i}$
    
    ${\partial L_p \over  \partial b} = 0$   ⇒    $\sum_{i=1}^N {\alpha_iy_i} = 0$
    
- 원문제
  
    ${\min{ L_{p}(\bold{w},b,{ \alpha  }_{ i }) }  } =\frac { 1 }{ 2 } { \left\| \bold{w} \right\|  }^{ 2 }-\sum _{ i=1 }^{ N }{ { \alpha  }_{ i }({ y }_{ i }({ \bold{w} }^{ T }{ \bold{x} }_{ i }+b)-1) }$
    $s.t.$   $\alpha_i \ge 0$ 

- 쌍대(Dual) 문제
  
    $\max { { L }_{ D }({ \alpha  }_{ i }) } =\sum _{ i=1 }^{ N }{ { \alpha  }_{ i } } -\frac { 1 }{ 2 } \sum _{ i=1 }^{ N }{ \sum _{ j=1 }^{ N }{ { \alpha  }_{ i }{ { \alpha  }_{ j }y }_{ i }{ y }_{ j }{ \bold{x} }_{ i }^{ T }{ \bold{x} }_{ j } }  }$
    
    
    $s.t.$    $\sum _{ i=1 }^{ N }{ { \alpha  }_{ i }{ y }_{ i } } =0, \quad
    { \alpha  }_{ i }\ge 0$
    
    
    
    - 위 쌍대 문제에서 $\bold{x}$와 $y$는 데이터로부터 주어진 값이고 $\alpha$만 미지수
    - 이때, KKT 조건에 따라 $\alpha_i(y_i(\bold{w}^T\bold{x}_i + b) -1) = 0$ 이라는 수식이 성립함
      
        ⇒ $\alpha_i = 0, (y_i(\bold{w}^T\bold{x}_i + b) -1) \ne 0$ 이거나 $\alpha_i \ne 0, (y_i(\bold{w}^T\bold{x}_i + b) -1) = 0$
        
        - $\alpha_i \ne 0, (y_i(\bold{w}^T\bold{x}_i + b) -1) = 0$인 경우,
          
            $(y_i(\bold{w}^T\bold{x}_i + b) -1) = 0$   →   $y_i(\bold{w}^T\bold{x}_i + b) = 1$이라는 것은 
            
            $\bold{x}_i$가 Plus-plane과 Minus-plane 상에 위치한다는 것을 의미
            → 이 $\bold{x}_i$ ( = Support Vector)에 대해서만 $\alpha_i$는 0보다 큰 값을 가지게 됨.
            
            ![이미지 출처: [https://techblog-history-younghunjo1.tistory.com/m/78](https://techblog-history-younghunjo1.tistory.com/m/78)](SVM%20aeb6168f959e49e2a6652261fd99d5e4/Untitled%205.png)
            
            이미지 출처: [https://techblog-history-younghunjo1.tistory.com/m/78](https://techblog-history-younghunjo1.tistory.com/m/78)
            

### 최종 Hyperplane 구하기

- SVM에서 찾고자 하는 것은 Margin이 최대화된 Hyperplane $\bold{w}^T\bold{x} + b$
  
    → $\bold{w}$와 $b$를 찾으면 Hyperplane 구할 수 있음
    
- 이전 단계에서 아래와 같은 수식을 얻었음
  
    $\bold{w} = \sum_{i=1}^N {\alpha_iy_i\bold{x}_i}$
    
    - $\bold{x}_i, y_i$는 주어진 데이터로부터 알아낼 수 있는 값이므로 단 하나뿐인 미지수인 $\alpha$를 알면 $\bold{w}$를 찾아낼 수 있음
    - $\bold{w}$를 구한 뒤, $(y_i(\bold{w}^T\bold{x}_i + b) -1) = 0$을 통해 $b$를 구할 수 있음
- 새로운 Instance$(\bold{x}_{new}$가 들어오면) $y_i(\bold{w}^T\bold{x}_{new} + b) -1$에 넣어서 그 값이 0보다 크면 Class Label을 +1로, 값이 0보다 작으면 Class Label을 -1로 예측함

# Soft Margin SVM

- 이전까지 설명한 SVM은 Hyperplane과 Support Vectors 사이에 Instance가 존재하지 않도록 하는 Hard Margin SVM이었음
- Soft Margin SVM은 Hyperplane과 Support Vectors 사이에 어느정도 Instance가 존재하는 것을 허용

### 목적 함수 및 제약 조건

$min \quad {1 \over 2}||\bold{w}||^2 +C\sum_{i=1}^N \xi_i$

$s.t. \quad y_i(\bold{w}^T\bold{x}_i + b) \ge 1-\xi_i, \quad \xi_i \ge0, \forall i$

![Untitled](SVM%20aeb6168f959e49e2a6652261fd99d5e4/Untitled%206.png)

$notation$

$C$: Penalty의 정도를 조절하는 Hyperparameter

$\xi$: Penalty

⇒ 미지수: $\bold{w}, b, \xi$

### 라그랑지안 문제로 변환

${\min{ L_{p}(\bold{w},b,{ \alpha  }_{ i }) }  } =\frac { 1 }{ 2 } { \left\| \bold{w} \right\|  }^{ 2 } + {C\sum_{i=1}^N\xi_i}-\sum _{ i=1 }^{ N }{ { \alpha  }_{ i }({ y }_{ i }({ \bold{w} }^{ T }{ \bold{x} }_{ i }+b)-1 + \xi_i) } - \sum_{i=1}^N\mu_i\xi_i$

$s.t.\quad\alpha_i \ge 0$

### 쌍대(Dual) 문제로 변환

- 원문제

${\min{ L_{p}(\bold{w},b,{ \alpha  }_{ i }) }  } =\frac { 1 }{ 2 } { \left\| \bold{w} \right\|  }^{ 2 } + {C\sum_{i=1}^N\xi_i}-\sum _{ i=1 }^{ N }{ { \alpha  }_{ i }({ y }_{ i }({ \bold{w} }^{ T }{ \bold{x} }_{ i }+b)-1 + \xi_i) } - \sum_{i=1}^N\mu_i\xi_i$


$s.t.\quad\alpha_i \ge 0$

- KKT 조건
  
    ${\partial L_p \over {\partial \bold{w}}} = 0 \quad$ ⇒     $\bold{w} = \sum_{i=1}^n\alpha_iy_i\bold{x}_i$
    
    ${\partial L_p \over \partial b} = 0\quad$⇒     $\sum_{i=1}^n\alpha_iy_i = 0$
    
    ${\partial L_p \over \partial \xi_i} = 0\quad$⇒     $C - \alpha_i - \mu_i = 0$
    

⇒ $L_D = {1 \over 2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j\bold{x}_i\cdot\bold{x}_j + C\sum_i \xi_i -\sum_i\sum_j\alpha_i\alpha_jy_iy_j\bold{x}_i\cdot\bold{x}_j - b\sum_i\alpha_iy_i + \sum_i\alpha_i - \sum_i\alpha_i\xi_i - \sum_i\mu_i\xi_i$

→ $\sum_i(C-\alpha_i-\mu_i)\xi_i = 0$, $\sum_i\alpha_iy_i = 0$

→ $L_D = {1 \over 2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j\bold{x}_i\cdot\bold{x}_j -\sum_i\sum_j\alpha_i\alpha_jy_iy_j\bold{x}_i\cdot\bold{x}_j + \sum_i\alpha_i$ 

→ $L_D = \sum_i\alpha_i  - {1 \over 2}\sum_i\sum_j\alpha_i\alpha_jy_iy_j\bold{x}_i\cdot\bold{x}_j$

⇒ $L_D({\alpha_i}) = \sum_{i=1}^N\alpha_i  - {1 \over 2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\bold{x}_i^T\bold{x}_j$

- 쌍대 문제

$max\quad L_D({\alpha_i}) = \sum_{i=1}^N\alpha_i  - {1 \over 2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\bold{x}_i^T\bold{x}_j$


$s.t.\quad \sum_{i=1}^N \alpha_iy_i = 0, 0 \le \alpha_i \le C$



### Plus) $\alpha_i$ 값에 따른 Instance 위치

- KKT 조건으로부터 $\alpha_i(y_i(\bold{w}^T\bold{x} + b)-1+\xi_i) = 0$ 수식을 얻을 수 있었음
    - Support Vector에 대해서만 $\alpha_i \ne 0$이 성립
- 또한 $C - \alpha_i - \mu_i = 0, \mu_i\xi_i = 0$이라는 수식이 성립함
    - **Case 1)** $\alpha_i = 0\quad$⇒ Support Vector가 아닌 Instance
    - **Case 2)** $0<\alpha_i < C\quad$
      →  $\mu_i < C$이면 $C - \alpha_i - \mu_i = 0$이 성립하기 위해 $\mu_i > 0$ 이어야 함. 
      → $\mu_i > 0$이라면 $\mu_i\xi_i = 0$이 성립하기 위해 $\xi_i = 0$이어야 함
      
        → $y_i(\bold{w}^T\bold{x} + b)-1 = 0$인 Instance
      
        ⇒ Margin 위에 위치하는 Support Vector
      
    - **Case 3)** $\alpha_i = C \quad$
      → $C - \alpha_i - \mu_i = 0$에서 $\alpha_i = C$라면 $\mu_i = 0$
      
        → $\mu_i = 0$이라면 $\xi_i > 0$
      
        ⇒ Margin 밖에 위치하는 Support Vector
      
        ![Untitled](SVM%20aeb6168f959e49e2a6652261fd99d5e4/Untitled%207.png)
      

- Hyperparameter C(오분류 비용)에 따른 분류 경계면 변화
  
    ![Untitled](SVM%20aeb6168f959e49e2a6652261fd99d5e4/Untitled%208.png)
    
    $$
    min \quad {1 \over 2}||\bold{w}||^2 +C\sum_{i=1}^N \xi_i
    $$
    
    - Large C: 목적함수에서 Penalty가 더 큰 영향력을 가짐
                  → Penalty를 줄이는 방향으로 학습이 진행
                  → Margin이 좁고, $\alpha_i = C$인 Support Vector의 수가 상대적으로 적음
    - Small C: 목적함수에서 Penalty의 영향력이 작아짐
      
                  → Penalty의 영향력이 적으므로 Margin을 조금 더 넓게 잡을 수 있음
        
                  → $\alpha_i = C$인 Support Vector의 수가 상대적으로 많음
    

# Nonlinear & Kernel

- Linear Model의 한계: 분류 경계면이 비선형일 경우 잘 찾아내지 함
  
    ![Untitled](SVM%20aeb6168f959e49e2a6652261fd99d5e4/Untitled%209.png)
    

<aside>
🧐 선형 분류가 가능한 고차원으로 데이터를 Mapping해서 모델을 학습하자!
![이미지 출처: [https://towardsdatascience.com/support-vector-machine-formulation-and-derivation-b146ce89f28](https://towardsdatascience.com/support-vector-machine-formulation-and-derivation-b146ce89f28)](SVM%20aeb6168f959e49e2a6652261fd99d5e4/Untitled%2010.png)

이미지 출처: [https://towardsdatascience.com/support-vector-machine-formulation-and-derivation-b146ce89f28](https://towardsdatascience.com/support-vector-machine-formulation-and-derivation-b146ce89f28)

⇒ 고차원 Mapping을 통해 Nonlinear(비선형) 분류 경계면 생성

### 고차원에서의 목적 함수 및 제약 조건

$min\quad{1 \over 2}||\bold{w}||^2 + C\sum_{i=1}^N\xi_i$


$s.t\quad y_i(\bold{w}^T\Phi(\bold{x}_i) + b) \ge 1-\xi_i,\quad \xi_i \ge0, \quad\forall i$

⇒ **라그랑지안 문제로 변환**

${\min{ L_{p}(\bold{w},b,{ \alpha  }_{ i }) }  } =\frac { 1 }{ 2 } { \left\| \bold{w} \right\|  }^{ 2 } + {C\sum_{i=1}^N\xi_i}-\sum _{ i=1 }^{ N }{ { \alpha  }_{ i }({ y }_{ i }({ \bold{w} }^{ T }{ \Phi({\bold{x}_i)} }+b)-1 + \xi_i) } - \sum_{i=1}^N\mu_i\xi_i$

- KKT 조건
  
    ${\partial L_P \over \partial w} = 0\quad$⇒ $\bold{w} = \sum_{i=1}^n\alpha_iy_i\Phi(\bold{x}_i)$
    
    ${\partial L_P \over \partial b} = 0\quad$⇒ $\sum_{i=1}^n\alpha_iy_i = 0$
    
    ${\partial L_P \over \partial \xi_i} = 0\quad$⇒ $C - \alpha_i - \mu_i = 0$
    

⇒ 쌍대(Dual) 문제로 변환

$max\quad L_D({\alpha_i}) = \sum_{i=1}^N\alpha_i  - {1 \over 2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\Phi(\bold{x}_i)^T\Phi(\bold{x}_j)$

$s.t.\quad \sum_{i=1}^N \alpha_iy_i = 0, 0 \le \alpha_i \le C$

😓 고차원으로 Mapping시키는 함수 $\Phi$를 어떻게 찾을까,,,?

👍🏻 Kernel Trick을 쓰자!

$max\quad L_D({\alpha_i}) = \sum_{i=1}^N\alpha_i  - {1 \over 2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\Phi(\bold{x}_i)^T\Phi(\bold{x}_j)$

⇒  $max\quad L_D({\alpha_i}) = \sum_{i=1}^N\alpha_i  - {1 \over 2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j K(\bold{x}_i, \bold{x}_j)$

### Kernel Trick

$max\quad L_D({\alpha_i}) = \sum_{i=1}^N\alpha_i  - {1 \over 2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\Phi(\bold{x}_i)^T\Phi(\bold{x}_j)$에서와 같이 고차원에서는 항상 $\Phi({\bold{x}_i})^T\Phi(\bold{x}_j)$와 같이 벡터의 내적 형태로 존재

→ 저차원 데이터를 입력 받아서 고차원 공간상에 내적 결과값을 줄 수 있다면 굳이 $\Phi$를 찾지 않아도 된다!

![Untitled](SVM%20aeb6168f959e49e2a6652261fd99d5e4/Untitled%2011.png)

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
    
    ![이미지 출처: [https://towardsdatascience.com/multiclass-classification-with-support-vector-machines-svm-kernel-trick-kernel-functions-f9d5377d6f02](https://towardsdatascience.com/multiclass-classification-with-support-vector-machines-svm-kernel-trick-kernel-functions-f9d5377d6f02)](SVM%20aeb6168f959e49e2a6652261fd99d5e4/Untitled%2012.png)
    
    이미지 출처: [https://towardsdatascience.com/multiclass-classification-with-support-vector-machines-svm-kernel-trick-kernel-functions-f9d5377d6f02](https://towardsdatascience.com/multiclass-classification-with-support-vector-machines-svm-kernel-trick-kernel-functions-f9d5377d6f02)



# 코딩 실습

### 실험 주제

### Main Experiment - Support Vector Classifier

#### 결과 해석



### Additional Experiment - Support Vector Regressor

#### 결과 해석


