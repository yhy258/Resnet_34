from load_data import Cifar100_data_load
from model import *
from evaluate import evaluate
from torchvision import transforms
import torch



transform = transforms.Compose([transforms.ToTensor()])
train_loader, test_laoder = Cifar100_data_load(transform=transform)


"""
    DEVICE 설정
"""
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')


def resnet34(DEVICE):
    """
        전형적인 Vanila Resnet 호출
    """
    return Resnet(BasicBlock, [3, 4, 6, 3]).to(DEVICE)

model = resnet34(DEVICE)

"""
    Train 시작!
"""

optim = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss() # loss 형식 선언
model.train()
epochs = 10 # 적은 에폭으로도 좋은 성능을 낸다.
for epoch in range(epochs):
    print("{}/{} Epochs".format(epoch + 1, epochs))
    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)
        # _, preds = torch.max(pred, 1)
        loss = criterion(pred, y)

        optim.zero_grad()
        loss.backward()

        optim.step()
    print("Loss is : ", loss.item())

"""
    Evaluate (Train Set)
"""

evaluate(model, test_laoder,DEVICE)
