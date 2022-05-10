# Training-a-classifier(CNN을 이용한 이미지 분류_파이토치)
<!--Line-->
For this tutorial, we will use the CIFAR10 dataset. torchvision.datasets에서의 CIFAR10 데이터셋을 활용했습니다.

참고 블로그 :
* https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
* https://gaussian37.github.io/dl-pytorch-conv2d/
* https://velog.io/@dltjrdud37/CNNConvolutional-Neural-Network


## CNN 정의하기
<!--Line-->
```Python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #(in channels, out channels, kernel size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #FC, 선형회귀함수 이용 (input_dim, output_dim) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch / fc를 위한 것
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```


## CNN 정의하기

해당 CNN은 "Convolution - Maxpooling - ReLu - FC" 구조로 이뤄져있다. 

<!--Image-->
![MNIST_CNN](https://user-images.githubusercontent.com/84561436/167672878-802536bd-30db-4b24-be32-ddc6e446fd55.png)


<!--Image-->
![images_dltjrdud37_post_b5830019-e195-4b85-8f41-f573630f0c51_Screen Shot 2021-01-27 at 2 45 45 PM](https://user-images.githubusercontent.com/84561436/167672266-f2163918-7ee5-402e-bb04-53fe16f2099d.png)

CNN은 보통 특징을 추출하는 Convolution, pooling 부분과 결합하고 분류하는 부분으로 구성되어있다. 모델은 다양한데 대부분 convolution-pooling-activation-FC 패턴이 일반적이다. 경우에 따라 다양한 구조를 갖기도 한다. CNN의 CONV, POOLING 등 용어정리는 참고블로그의 세번째 링크에 정리 잘 되어있다.


## Loss function and optimizer 정의하기

```Python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

Loss func로는 crossEntropyLoss 를 이용하고 optimizer는 모멘텀SGD를 이용한다. 


