# lr = 1e-4 (adam) 10 epochs -> 60% accuracy
import torch


def evaluate(model,test_loader,DEVICE):
    model.eval()
    correct = 0

    total = 0
    with torch.no_grad():
        for test_x, test_y in test_loader:
            batch_size = test_x.size(0)
            test_x = test_x.to(DEVICE)
            test_y = test_y.to(DEVICE)
            pred = model(test_x)
            _, pred = torch.max(pred, 1)

            total += batch_size

            print("이번 iter에서 맞은 횟수는 {} 입니다.".format((pred == test_y).sum().item()))

            correct += (pred == test_y).sum().item()

        print('Accuracy for {} images: {:.2f}%'.format(total, correct / total * 100))