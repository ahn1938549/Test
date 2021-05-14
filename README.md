![dropout](https://user-images.githubusercontent.com/69898343/118241994-c6ba5280-b4d7-11eb-97c4-d330f8d84a05.png)

- 다음과 같이, overfitting을 방지하기 위하여 랜덤한 뉴런을 배제한 후 학습을 통하여 결과를 도출 하는 효과인 Dropout의 개념을 도입하였다. 

### Overall Architecture

![alxnet](https://user-images.githubusercontent.com/69898343/118252613-ea839580-b4e3-11eb-800c-7a41214f7a65.png)

1. Convolution 1 Layer

- 227 X 227 X 3 의 이미지를 각 각 48개의 11 X 11 X 3의 필터로 Convolution (stride =4, padding =0) = 2개의 (55 X 55 X 48)

- 55 X 55 X 48 의 이미지를 3 X 3 (stride =2) 으로 Max Pooling = 2개의 (27 X 27 X 48)

2. Convolution 2 Layer

- 27 X 27 X 48의 이미지를 각 각 128개의 5 X 5 X 48 필터로 Convolution (stride =1, Padding =2) = 2 X (27 X 27 X 128)

- 27 X 27 X 128의 이미지를 3 X 3 (stride =2) 으로 Max Pooling 2 X (13 X 13 X 128)

3. Convolution 3 Layer (GPU connection)

- 13 X 13 X 256의 이미지를 384개의 3 X 3 X 256의 필터로 Convolution(stride = 0, Padding =1) = 13 X 13 X 384

4. Convolution 4 Layer

- 2개의 (13 X 13 X 192)의 이미지를 각 각 192개의 3 X 3 X 192의 필터로 Convolution(stride = 0, Padding =1) = 2 X (13 X 13 X 192)

5. Convolution 5 Layer

- 2개의 (13 X 13 X 192)의 이미지를 각 각 128개의 3 X 3 X 192의 필터로 Convolution(stride = 0, Padding =1) = 2 X (13 X 13 X 128)

- 13 X 13 X 128의 이미지를 3 X 3(stride =2)로 Max Pooling = 2 X (6 X 6 X 128)

6. FC

- 2 X (6 X 6 X 128) = 9216을 Flatten 후 4096개의 뉴런과 FC를 하여 계산한다.

7. FC

- 4096개의 뉴런을 4096개의 뉴런과 FC를 하여 계산한다.

8. FC

- 4096개의 뉴런을 1000개의 뉴런과 FC한 후, softmax 함수를 사용 1000개의 클래스에 대한 값으로 나타낸다.


### Results

ILSVRC - 2010              

  ![2010](https://user-images.githubusercontent.com/69898343/118216162-f0f71a80-b4ad-11eb-98bc-95368ae21e5c.png)

- 2010년의 ILSVRC에 적용한 결과

ILSVRC - 2012 

  ![2012](https://user-images.githubusercontent.com/69898343/118216401-65ca5480-b4ae-11eb-8410-d068e17589ff.png)

    - ILSVRC-2012 15.3%의 Error로 2위와의 10.9%의 격차로 우승!
