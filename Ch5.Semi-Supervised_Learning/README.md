# **Semi-Supervised Learning**
- 본 Tutorial은 고려대학교 산업경영공학부 대학원 Business Analytics 강의 자료 및 외부 자료를 기반으로 작성되었습니다.
- 본 Tutorial에서는 Semi-Supervised Learning 방법론 중 Temporal Ensemble 구현에 관한 내용을 포함하고 있습니다.

---
## Overview
🤔 What is Semi-supervised Learning?
<p align = 'center'>
<img src = 'https://user-images.githubusercontent.com/56019094/209630453-8fa04984-26b1-4d7b-8a1e-8d125c61e263.png' width = 85%>
</p>

<center>
이미지 출처: https://blog.est.ai/2020/11/ssl/
</center>

**Supervised Learning**은 Labeled Data를 이용하고, **Unsupervised Learning**은 Label 정보가 없는 즉, Unlabeled Data를 이용한다는 것은 다들 알고 계실 겁니다. 그렇다면 ```Semi-supervised Learning```은 무엇을 이용할까요?  

위 Figure를 보면 Semi-supervised Learning은 대부분이 Unlabeled Data(색이 없는 원)고, 조금의 Labeled Data(색이 있는 원)가 있습니다.   

>semi-  
: '반', '어느 정도의'의 뜻을 나타냄  

위 Figure와 ```semi-```의 사전적 의미를 통해 **Semi**-supervised Learning이라는 학습 방법은 **'어느 정도'** Supervised Learning입니다. 즉, Semi-supervised Learning에는 Labeled Data와 Unlabeled Data가 둘 다 사용됩니다.  

➢ 소량의 Labeled Data에는 Supervised Learning을, 대량의 Unlabeled Data에는 Unsupervised Learning을 적용하여 전체 성능을 향상시키는 것을 목적으로 합니다.
  
💰 Why?  

Label 정보를 수집하는 "Labeling" 작업에 소요되는 많은 자원과 비용 때문에 Semi-supervised Learning 방법론이 등장하게 되었습니다. 

<hr/>

## Temporal Ensemble

본 Tutorial에서 리뷰할 Temporal Ensemble은 Semi-supervised Learning 방법론 중 ```Consistency Regularization```에 해당합니다. 

- Consistency Regularization란?  
    - Unlabeled Data에 Realistic Perturbation을 적용해도 예측 결과는 일관적일 것이라는 가정
    - Unlabeled Data와 Perturbed Unlabeled Data간의 Consistency를 유지하도록 모델 학습
    - ☛ 헷갈리는 Sample에 대한 유연한 예측이 가능  
  

```Temporal Ensemble```은 𝝅-Model 논문에서 𝝅-Model의 한계점을 보완하기 위해 같이 제안한 방법론입니다.

<p align = 'center'>
<img src = 'https://user-images.githubusercontent.com/56019094/209636928-9c9d7a79-b176-40fe-ac58-03e19ea3f8c8.png' width = 90%>
</p>

- 𝝅-Model
    - Input에 Gaussian Noise와 Stochastic Augmentation을 적용해 두 가지 Augmented Input 구성  
    - 동일한 Network 구조에 다른 Dropout Regularization을 적용하여 모델 구성
    - Augmentation 1 모델의 Output과 Label로 Cross Entropy Loss 산출 (Supervised Loss)
    - Augmentation 1 모델의 Output과 Augmentation 2 모델의 Output 간의 Consistency를 유지하도록 MSE Loss 산출 (Consistency Loss)

    - 한계점
        - Network의 Single Evaluation에 기반해 Training Target이 구성되기에 Noisy
    

- Temporal Ensemble  
    >🤩 Temporal Ensemble은 위와 같은 𝝅-Model의 한계점을 극복하기 위해 Network Evaluation(Output)들을 Ensemble하여 Target 값으로 사용

    <p align = 'center'>
    <img src = 'https://user-images.githubusercontent.com/56019094/209637719-78b704f4-435d-4e1f-9d11-59f7987578d3.png' width = 90%>
    </p>

    - 위 Figure의 수도 코드에서도 확인할 수 있다시피 Temporal Ensemble은 Network Evaluation(Output)을 Ensemble하기 위해 Exponential Moving Average (EMA)를 사용합니다.
    - Output의 EMA를 Target으로 사용해 Noise가 줄어든다는 장점이 있지만, EMA를 계산하기 위해 Network의 Output들을 추가적으로 저장해야 한다는 것과 EMA 수식에 포함되는 $\alpha$가 추가적인 하이퍼 파라미터로 사용됩니다.  
    본 Tutorial에서 또한 이 $\alpha$의 값을 변경하며 모델의 성능 변화를 측정해보았습니다.

<hr/>

## Code

- 데이터셋 불러오기
```python
def mnist_dataset(root, transform):
    # Load Train Data
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        transform=transform,
        download=True
    )

    # Load Test Data
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        transform=transform,
        download=True
    )
    return train_dataset, test_dataset
```
본 Tutorial에 사용된 데이터셋은 가장 널리 알려진 데이터셋인 MNIST 데이터셋을 이용했습니다.


- 전처리 함수 정의
```python
def preprocess_data(train_dataset, test_dataset, batch_size, k, n_classes, seed, shuffle_train=False, return_idx=True):
    # Randomly form unlabeled data in training dataset
    n = len(train_dataset)  # Dataset size
    rand_seed = np.random.RandomState(seed) # Set seed 
    indices = torch.zeros(k)  # Empty tensor for saving indices for keeping labeled data
    unlabel_indices = torch.zeros(n - k)  # Empty tensor for indices of unlabeled data
    quot = k // n_classes 
    temp_index = 0

    for i in range(n_classes):
        class_items = (train_dataset.train_labels == i).nonzero()  # indices of samples with label i
        # train_dataset.train_labels == i : Train Data 중 Label이 i인 것들만 True
        # .nonzero(): Element 값이 0이 아닌 Element들의 Indices만 반환 (Element 값이 0이면 .nonzero() 결과에도 반환되지 않음)

        n_class = len(class_items)  # number of samples with label i
        shuffled = rand_seed.permutation(np.arange(n_class))  # shuffle them  |shuffled| = |n_class|
        indices[i * quot: (i+1) * quot] = torch.squeeze(class_items[shuffled[:quot]]) # |class_items[shuffled[:quot]]| = (alpha, 1)이라서 2차원이라 Squeeze 적용
        unlabel_indices[temp_index: temp_index+n_class-quot] = torch.squeeze(class_items[shuffled[quot:]])
        temp_index += (n_class-quot)

    unlabel_indices = unlabel_indices.long() # tensor as indices must be long, byte or bool
    train_dataset.train_labels[unlabel_indices] = -1 # Unsupervised의 경우 Label = -1 할당

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=0,
                                               shuffle=shuffle_train)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=0,
                                              shuffle=False)

    if return_idx:
        return train_loader, test_loader, indices
    return train_loader, test_loader
```

- Input Data에 적용할 Gaussian Noise 적용 클래스 정의
```python
class GaussianNoise(nn.Module):
    def __init__(self, batch_size, input_shape, std):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape # |self.shape| = (batch_size, input_shape's first element, input_shape's second element, input_shape's third element)
        self.std = std
        self.noise = torch.zeros(self.shape).cuda()

    def forward(self, x):
        self.noise.normal_(mean=0, std=self.std) # torch.normal(mean, std): Returns a tensor of random numbers from separate normal distributions
        return x + self.noise
```
- Labeled Data에 사용할 Cross Entropy Loss 함수 정의
```python
def labeled_ce_loss(out, labels):
    cond = (labels >= 0) # 참고: preprocess_data()에서 Unlabeled Data의 Label = -1 할당함 
    labeled_arr = torch.nonzero(cond) # Array of Labeled Sample Index
    num_sup = len(labeled_arr) # Num of Supervised Samples
    
    # Supervised Instance 수가 0보다 많다면, 즉 존재한다면
    if num_sup > 0:
        labeled_outputs = torch.index_select(input=out, dim=0, index=labeled_arr.view(num_sup)) # labeled_arr.view(num_sup): Flatten() 역할
        labeled_labels = labels[cond] 
        loss = F.cross_entropy(labeled_outputs, labeled_labels)
        return loss, num_sup
    
    # Supervised Instance가 없다면 CE Loss = 0
    loss = torch.tensor([0.], requires_grad=False).cuda()
    return loss, 0 # num_sup == 0 이면 loss와 0(num_sup이 없다는 뜻) 반환
```
위 함수가 본 Tutorial에서 Network Output을 Ensemble하는 방법 다음으로 핵심이라고 할 수 있는 함수입니다. Labeled Data와 Unlabeled Data를 함께 사용하기에 Labeled Data가 존재한다면, Labeled Data에 해당하는 Data만 골라서 Cross Entropy 함수를 계산하는 모습을 `if num_sup > 0` 블록에서 확인할 수 있습니다.

- Unlabeled Data에 사용할 MSE Loss 함수 정의
```python
def mse_loss(cur_out, ensem_out):
    # Current Output과 Ensemble Output 간의 MSE
    se = torch.sum((F.softmax(cur_out, dim=1) - F.softmax(ensem_out, dim=1)) ** 2)
    
    return se/len(cur_out)
```
Unlabeled Data에 대해서 MSE Loss를 구할 때 Input으로 들어오는 데이터를 보면 Cross Entropy Loss를 계산할 때와 달리 Label이 없는 모습을 확인할 수 있습니다.

- Supervised Loss, Unsupervised Loss(=Consistency Loss), Total Loss 반환 함수 정의
```python
def return_losses(cur_out, ensem_out, w, labels):
    # cur_out: Current output
    # ensem_out: Ensemble output
    # w: Weight for summation loss

    sup_loss, nbsup = labeled_ce_loss(cur_out, labels)
    unsup_loss = mse_loss(cur_out, ensem_out)
    total_loss = sup_loss + w * unsup_loss # 최종 Loss = Supervised Loss + w*Unsupervised Loss

    return total_loss, sup_loss, unsup_loss, nbsup
```

- Supervised Loss와 Unsupervised Loss를 더해 Total Loss 계산 시 Unsupervised Loss Term의 가중치를 조정하는 함수 정의 (아래 수식에서 $w$에 해당)

$$L_{total} = wL_u + L_s$$

```python
def weight_ramp_up(epoch:int, max_epochs:int, max_val:float, mult, n_labeled:int, n_samples:int):

    max_val = max_val * (n_labeled/n_samples)

    if epoch == 0:
        return 0.
    elif epoch >= max_epochs:
        return max_val

    return max_val * np.exp(-mult * (1. - float(epoch)/max_epochs)**2)
```
$w$를 단순히 어떤 스칼라 값 하나로 고정해서 사용하는 것이 아니라는 점이 또 하나의 특징이라고 할 수 있습니다. 실제 논문에서는 아래 형광펜으로 표시한 부분에 해당합니다. 해당 함수를 통해 특정 Epoch(max_epochs) 이후에는 $w(t)$가 특정한 값으로 고정되어 사용됩니다.

<p align = 'center'>
<img src = 'https://user-images.githubusercontent.com/56019094/209660612-56f94824-00e5-4117-af6c-a26f349adcf9.png' width = 80%>
</p>

- Network (CNN Model) 모델 정의
```python
# Base Model로 CNN 사용
class CNN(nn.Module):
    def __init__(self, batch_size:int, std:float, input_shape:tuple = (1,28,28), drop_out:float = 0.5, first_layer:int = 16, second_layer:int = 32):
        super(CNN, self).__init__()
        self.std = std
        self.drop_out = drop_out
        self.first_layer = first_layer
        self.second_layer = second_layer
        self.input_shape = input_shape
        self.conv_block1 = nn.Sequential(nn.Conv2d(1, self.first_layer, 3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.first_layer),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, stride=2, padding=1))
        self.conv_block2 = nn.Sequential(nn.Conv2d(self.first_layer, self.second_layer, 3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.second_layer),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, stride=2, padding=1))
        self.drop = nn.Dropout(self.drop_out)
        self.fc = nn.Linear(self.second_layer * 7 * 7, 10)

    def forward(self, x):
        if self.training:
            b = x.size(0)
            noise = GaussianNoise(b, self.input_shape, self.std) # Input에 Gaussian Noise를 적용해서 Augmentation 수행
            x = noise(x)

        # first block
        x = self.conv_block1(x)

        # second block
        x = self.conv_block2(x)

        # Classifier (FC)
        x = x.view(-1, self.second_layer * 7 * 7) # Flatten
        x = self.fc(self.drop(x)) # Apply Dropout

        return x
```

- Train 함수 정의 
```python
def train(model, 
        train_loader, 
        val_loader,
        k:int, 
        alpha:float, 
        lr:float, 
        num_epochs:int, 
        batch_size:int, 
        n_instances:int, 
        n_classes:int = 10, 
        max_epochs:int = 80, 
        max_val:float = 1.
        ):    
    
    # Feed model to GPU if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Setting Optimizer Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Set First Ensemble Output as zeros
    Z = torch.zeros(n_instances, n_classes).float().to(device)
    z = torch.zeros(n_instances, n_classes).float().to(device)
    outputs = torch.zeros(n_instances, n_classes).float().to(device)

    losses = [] # Total Loss
    sup_losses = [] # Supervised Loss (Cross-Entropy Loss)
    unsup_losses = [] # Unsupervised Loss (MSE Loss)
    best_loss = 10_000

    for epoch in range(num_epochs):
        model.train()

        # Calculate Unsupervised Loss Weight
        w = weight_ramp_up(epoch, max_epochs, max_val, 5, k, 60000)
        w = torch.tensor(w, requires_grad=False).to(device)

        # Targets change only once per Epoch
        for i, (images, labels) in enumerate(train_loader):
            batch_size = images.size(0)  # retrieve batch size again cause drop last is false
            images = images.to(device)
            labels = labels.requires_grad_(False).to(device)

            optimizer.zero_grad()
            out = model(images)

            # 현재 Batch에 해당하는 Ensemble 결과 가져오기
            z_ = z[i*batch_size: (i+1)*batch_size]
            z_.requires_grad_(False)
            loss, sup_loss, unsup_loss, nbsup = return_losses(out, z_, w, labels)

            # Save outputs
            outputs[i*batch_size: (i+1)*batch_size] = out.detach().clone()
            losses.append(loss.item())
            sup_losses.append(nbsup*sup_loss.item())
            unsup_losses.append(unsup_loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()

        loss_mean = np.mean(losses)
        sup_loss_mean = np.mean(sup_losses)
        unsup_loss_mean = np.mean(unsup_losses)

        # 5 Epoch마다 Total Loss, Supervised Loss, Unsupervised Loss 출력
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_mean:.4f}, Supervised Loss: {sup_loss_mean:.4f}, Unsupervised Loss: {unsup_loss_mean:.4f}')
    
        # Model의 Outputs에 EMA 이용해서 Ensemble Outputs으로 Update
        Z = alpha * Z + (1. - alpha) * outputs
        z = Z * (1. / (1. - alpha ** (epoch + 1)))

        if loss_mean < best_loss:
            best_loss = loss_mean
            
            print('='*10, f'{epoch + 1} Epoch Model is Saved', '='*10)
            torch.save({'state_dict': model.state_dict()}, f'model_best.pth')
```
Temporal Ensemble의 핵심은 방법론의 이름에도 나와있듯이 바로 Ensemble을 한다는 것입니다. 이전에 Ensemble Learning Chapter에서 보았던 방법에서 보았던 방법들과 어떻게 보면 일맥상통하게 Output을 Aggregate하는 모습을 위 `train` 함수 중 아래 블록에서 확인할 수 있습니다. 
```python
# Model의 Outputs에 EMA 이용해서 Ensemble Outputs으로 Update
Z = alpha * Z + (1. - alpha) * outputs
z = Z * (1. / (1. - alpha ** (epoch + 1)))
```

- 모델 성능(Accuracy) 평가 함수 정의
```python
def evaluation(model, loader):
    # Evaluation using Best Model
    checkpoint = torch.load('model_best.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    correct = 0
    total = 0

    for i, (samples, labels) in enumerate(loader):
        samples = samples.cuda()
        labels = labels.requires_grad_(False).cuda()
        outputs = model(samples)
        _, predicted = torch.max(outputs.detach(), 1)
        total += labels.size(0)
        correct += (predicted == labels.detach().view_as(predicted)).sum()
    
    accuracy = 100 * float(correct) / total
    
    print("="*10, "Evaluation Result", "="*10)
    print(f'Evaluation Result - Accuracy: {accuracy:.2f}')
    return np.round(accuracy,2)  
```

- 성능 측정 결과 (결과 중 일부)
```python
Epoch [50/50], Loss: 0.1354, Supervised Loss: 0.6107, Unsupervised Loss: 0.1115
========== 50 Epoch Model is Saved ==========
========== Evaluation Result ==========
Evaluation Result - Accuracy: 95.48
```

## Mini Experiment
```python
# Model의 Outputs에 EMA 이용해서 Ensemble Outputs으로 Update
Z = alpha * Z + (1. - alpha) * outputs
z = Z * (1. / (1. - alpha ** (epoch + 1)))
```
>EMA에서 핵심인 `alpha` 값을 0부터 1까지 변경하며 성능을 비교해보는 간단한 실험을 추가적으로 진행해보았습니다. alpha = 0 ~ 0.9로 실험을 진행해 보았습니다.  

😅 alpha = 1로 설정할 경우, `(1. / (1. - alpha ** (epoch + 1)))`에서 첫 번째 Epoch = 0에서 `1. / 0'이 되어 에러가 발생해 피치 못하게 실험을 진행하지 못했습니다.


<center>

|alpha|Accuracy|
|-----|--------|
|0|94.43
|0.1|94.75
|0.2|94.61
|0.3|95.35
|0.4|95.96
|0.5|94.12
|0.6|95.35
|0.7|94.36
|0.8|95.67
|0.9|94.28
</center>

- 결론
    - 본 Tutorial에서 alpha 값을 변화시키며 실험을 진행한 결과 Accuracy 상에서 큰 차이가 발생하지는 않았으나, alpha = 0.7인 경우를 제외하고는 전반적으로 alpha = 0.3 ~ 0.8인 경우가 alpha = 0, 0.1, 0.9보다 높은 성능을 보였습니다.
    - 이러한 성능 차이가 발생한 이유는 Temporal Ensemble의 Motivation인 Output의 Noise를 Network Output의 EMA를 통해 감소시키는 것이 효과가 있었다고 판단됩니다.
    - 한계점  
        1. 본 Tutorial에서 사용된 Base Model의 경우 파라미터 값 초기화를 위해 특정한 값을 고정시킨 것이 아니기 때문에, Seed에 따라 성능에 변화가 발생할 수 있습니다.
        2. alpha = 1인 경우를 코드 상의 문제로 인해 실험을 진행해보지 못했습니다.