
# Very Deep Convolutional Networks For Large-Scale Image Recognition '10 Apr 2015'

## Abstract

  - 60 million parameters, 650,000 neurons을 사용하였다.
  
  - five convolutional layers with max-pooling layers, 3 Fully-connected layers final 1000 softmax를 통하여 학습을 진행하였다. 
      
  - 'Dropout'의 기법을 제시하며, overfitting을 방지하는 방법을 제시하였다. 
  
  - ILSVRC-2012 15.3%의 Error로 우승!
    
***

## Model


### DataSet
  
  - 22,000 categories with 15 million labed high-resolution images on ImageNet
  
  - (ILSVRC) 1000 images each have 1000 categories. 
  
*** 

### ConvNet Architecture Setting & Reasons for selection

1. Input Image
    
    - 227 X 227 resolution RGB 이미지 
    
2. Activation Func

    - 이전에는, f(x) = tanh(x)가 공용되는 함수였으나,
    
    - ReLu f(x) = Max(0,x)를 사용하였다. (연산이 빠르고, tanh(x)에서 Gradient Vanishing을 방지 할 수 있다.)
    
    - 다만, ReLU에서는 Dying ReLU라는 문제점. (값이 음수일 경우 무시)가 나타난다.
 
    
3. Overlapping Pooling

    - Stride를 통하여 얻어지는 값이 output의 값이, 입력한 output값보다 적을 경우 Overlapping을 통하여 stride를 줄이도록 설계.
    
4. Filter

    - 논문에서 가장 자주 거론한 Architecture 구성 요소이며, 3x3 filter를 사용한다.

      1x1 filter, 5x5 filter , 7x7 filter 를 사용하지 않은 이유는 정리하자면 다음과 같다.

      - 우선 1x1 filter를 사용하지 않은 이유는 다음과 같다.

        - 1x1 Conv를 사용하면 입력과 출력 사이즈가 같아 언뜻 보기엔 무의미해 보이지만, 그 효과는 Non-linearity를 추가하는 것이다. 즉, Receptive Field는 유지하면서 비선형성을 높일 수 있는 방법이다. <br> 하지만 이 filter를 사용하는 것은 큰 효과를 보지 못하였는데, 논문에서 논하는 이유는 다음과 같다. 
          - 1x1 filter를 3x3 filter와 섞어서 쓴 모델이 3x3 filter 만을 사용한 모델보다 성능이 낮은 이유는 단순히 Receptive field를 유지한채 비선형성을 늘리는 것보다 , Receptive field가 변화하며 Spartial Context를 포착하도록 학습시키는 것이 더 중요하다는 것.

             + Receptive Field : 얼마나 많은 문맥 정보를 사용하는 가로 이해하면 될 것 같다.
             + 1x1 filter는 이후에 연산량을 줄이는 데에도 사용되었다. -- Bottle Neck in Inception Net

  
        - 3x3 filter 를 쓴 모델을 5x5 filter 로 바꾸면 Error rate 7% 상승
        
          -  Large Scale 이미지에 있어서 작은 Receptive Field를 갖는게 더 좋다.
      
      - 7x7 filter 를 1번 사용한 것과 3x3 filter를 3번 사용한 결과는 동일하다. <br> 즉 , Output Size가 동일하다. <br> 하지만 filter를 3개를 쓰는 것이 1개를 쓰는 것을 택한 이유는 다음과 같다.

        - Non-Linearity를 더 많이 반영할 수 있다.
        - 연산에 필요한 Parameter 수가 더 적다.

           3x(3x3xCxC) vs. 7x7xCxC for C channels per layer (3x3xC+3x3xC+3x3xC < 7x7xC) 


5. FC Layer
  
  - 4096 -> 4096 -> 1000 - Softmax의 구조 
  
    - Test 과정에서는 Fully Convolutional Network로 바뀌며, FC layer를 잘라낸다.
    
      - 논문에서는 여기에 대해 따로 언급하지 않았지만, 논문을 읽으며 추측한 바 Large-Scale 이미지의 Spartial Context를 최대한 유지하기 위해 1x1 filter 대신 3x3 filter를 사용한 점, 그리고 입력 이미지의 크기를 계속 다르게 주어 모델이 이미지 Scale에 Robust하도록 학습시키려고 노력한 점으로 미루어 보아 그 이유가 여기 있는 듯 하다.
      

  
7. Addional Skill 
  
    - Data Argumentation
    - Drop out in first 2 FC layers
    - LRN
    - Mini-Batch Train
    - Batch Norm 
    - Weight Decay 


    - 논문에서는 동일한 Depth의 동일한 구조를 가진 모델에서 LRN, Batch Norm을 추가한 결과와 추가하지 않은 결과를 비교하며 그 영향과 효과를 논한다. 
    
      - 결과만 논하자면, LRN, Batch Norm과 같은 기법은 Computing Power, 메모리 소비 감소량 등에서 그 역할을 할 수 있지만 모델의 성능에는 큰 영향이 없다고 한다.
 
    * LRN -  Local Response Normalization : LRN으로 인해 정규화 된 layer들은 lateral inhibition의 역할을 한다

      lateral inhibition (측면 억제) : relu를 사용할 때 발생하는 문제인데, relu는 양수 값을 그대로 사용하기 때문에 만일 어떤 한 픽셀의 값이 매우 크게 되면 그 주변의 픽셀에 영향을 미치게 된다. <br>예를 들어 Average Pooling / Max Pooling 에서 큰 값을 가진 해당 pixel이 외에도 중요한 의미를 가진 pixel이 있다고 한다면 그 값들이 의미 없이 모두 사라지게 되는 경향이 있다. <br> 따라서 해당 Activation map의 같은 위치에 위치한 픽셀들 끼리 정규화를 진행해준다. 
        
        -AlexNet에서 사용한다. 
        
        -요즘에는 Batch Normalization을 사용한다.

*** 

### Classification FrameWork 

#### Train

  - 1000개의 Class를 가진 ILSVRC 2012 Dataset 사용
  
  - Training / Validation / Test Set

  - Mini - Batch (256) , Learning_rate = 0.01 , Gradient Descent + Momentum 0.9 로 맞추어 Train한다.
  
      - Learning_rate은 정확도 상승이 없을 때, 1/10으로 감소시킨다. 

        - Model Train 과정에서 총 3번의 감소가 발생


  - Depth가 큰 경우 학습의 방향이 튈 수 있기 떄문에 초기 Weight Initialization이 중요. 다음과 같은 방법을 사용한다.
  
      - Depth가 얕은 A 모델을 학습하여 적당한 Weight 초기값을 찾는다.
      - Depth가 더 큰 모델을 학습할 경우, A 모델의 처음 4개의 Conv layer와 뒷 단의 FC layer 의 Weight를 초기값으로 할당한다.
      - 나머지 layer의 Weight은 평균이 0, 분산이 0.01 인 Normal Distribution에서 구한다.

        + Glorot & Bengio의 Random Initalization Procedure Method를 사용하면 이렇게 귀찮게 초기화하지 않아도 된다고 한다.
   
  - 입력이미지는 224x224 로 Rescale된 or Crop된 이미지이다. 
    
    * Rescale되는 이미지는 224 x 224 보다 큰 이미지이면 모두 가능하다. 
  
    입력이미지를 더 광범위한 범위에서 학습하기 위해 이미지를 무작위로 상하 반전시키고, <br> RGB Color를 바꾸는 등의 Data argumentation을 진행한다.
    
      - 이후 이 이미지를 Crop 하거나, 입력 이미지 크기를 다르게 하여 학습시키는 과정도 있는데, <br> 이는 모델이 이미지의 Spartial Context를 조금 더 잘 이해하고 학습하도록 하기 위함이다.

        - 총 2가지 방법으로 입력 이미지를 Setting한다.
        
          1. 입력 이미지 사이즈를 고정시키는 것. (Fixed-Scale)
            
            ex) Scaled Size = 224 x 224 / 384 x 384
            
          2. 입력 이미지 사이즈를 범위를 주어 무작위로 Scaling하는 것. (Multi-Scale)
            
            ex) [224 , 384] 범위에서 Scale =  -> 300 x 300 or 318 x 318 , ... 
          
              - Object 들은 실제로 모두 크기가 다르기 때문에 이 방법이 분명 이점이 있다. 
              - 빠르게 학습하기 위해 1번 Method의 Weight로 초기화시켜 훈련을 진행한다.
          
      - 이 결과를 확인하기 위해 CNN 모델을 FCN 모델로 바꾸어 Test 를 진행한다.


#### Test

  - Output : a Class score map of channels equal to the number of classes.
    
    - FCN을 사용하므로, 각 Class 개수와 같은 수의 Score map을 출력으로 낸다.
    - 가장 높은 Score를 갖는 Class가 예측 이미지이다.

      - Test에서 FCN을 사용하는 이유 
        
        - 일반적인 CNN 알고리즘은 FC layer에서 Spartial 정보를 잃게 된다.<br>반면에 FCN은 물체의 공간 정보 또한 유지하며 학습한다. Segmentation에서 사용하는 이유. 

        - 물체 크기 변화에 Robust 해진다.
        

      - FCN을 사용하므로, 입력 이미지의 Size를 신경쓸 필요가 없다.
        - FCN의 출력은 입력 이미지에 대한 비율로 나타난다.
          - Ex) N x N -> (N/10) x (N/10)


ILSVRC - 2010              

  ![2010](https://user-images.githubusercontent.com/69898343/118216162-f0f71a80-b4ad-11eb-98bc-95368ae21e5c.png)
      
      
ILSVRC - 2012 
  
  ![2012](https://user-images.githubusercontent.com/69898343/118216401-65ca5480-b4ae-11eb-8410-d068e17589ff.png)


  - 모델의 성능은 2가지 방법으로 측정
  
    - 1 Top / 5 Top
    
      1. 1 Top : 잘 못 예측된 이미지의 비율
      2. 5 Top : 예측 이미지가 상위 5개의 카테고리 밖에 있을 때의 비율 

    ![hjgkhjgj](https://user-images.githubusercontent.com/59076451/118116407-34597680-b425-11eb-800c-0f126b31203c.PNG)
    
    
  
***

[VGG Paper link](https://arxiv.org/abs/1409.1556)


