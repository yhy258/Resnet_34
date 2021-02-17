import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, infilter, outfilter, stride):
        super().__init__()  # 이 클래스로 훈련을 진행하고 이럴게 아니고, block을 생성하는데 사용할 거라서 super 안에 자신을 안넣어도 된다.
        self.conv = nn.Sequential(
            nn.Conv2d(infilter, outfilter, 3, stride, padding=1),
            nn.BatchNorm2d(outfilter),
            nn.ReLU(),
            nn.Conv2d(outfilter, outfilter, 3, 1, padding=1),
            nn.BatchNorm2d(outfilter)
        )
        self.shortcut = nn.Sequential()

        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(infilter, outfilter, 3, stride, padding=1),
                nn.BatchNorm2d(outfilter)  #
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.conv(x) + self.shortcut(x))


class Resnet(nn.Module):
    def __init__(self, block, num_blocks):  # block = block 생성 class (BasicBlock)
        super().__init__()# num_blocks에 따라 레스넷을 따로 선언해 줄 생각이다, 때문에, super 안에 자신을 안넣어도 된다.
        # num_blocks는 리스트이다. Resnet Architecture에 따른 블락 생성 위함.
        """
          self.conv1 =  !!
          self.conv2 = self._make_layer(block,num_blocks[0], 64,64,1)
          self.conv3 = self._make_layer(block,num_blocks[1], 64,128,2)
          self.conv4 = self._make_layer(block,num_blocks[2], 128,,256, 2)
          ... 이런식으로 선언 할 수 있다. 위의 코드도 훨씬 줄어들었다.
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = self._make_layer(block, num_blocks[0], 64, 64, 1)
        self.conv3 = self._make_layer(block, num_blocks[1], 64, 128, 2)
        self.conv4 = self._make_layer(block, num_blocks[2], 128, 256, 2)
        self.conv5 = self._make_layer(block, num_blocks[3], 256, 512, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcs = nn.Linear(512, 100)

    def _make_layer(self, block, num_block, infilter, outfilter, stride, ):
        layers = []
        strides = [stride] + [1] * (num_block - 1)

        for stride in strides:
            layers.append(block(infilter, outfilter, stride))
            infilter = outfilter

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fcs(x)
        return x