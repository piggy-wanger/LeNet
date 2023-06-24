import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from models.LeNet import LeNet


def main():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # 5000张训练图片
    # 第一次使用时需要将download设为True
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=36, shuffle=True, num_workers=0)

    # 1000张验证图片
    # 第一次使用时需要将download设为True
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_set, batch_size=5000, shuffle=False, num_workers=0)

    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the input; data is a list of [inputs, labels]
            inputs, labels = data

            # zeros the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 500 == 499:
                with torch.no_grad():
                    outputs = net(val_image)
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' % (
                        epoch + 1, step + 1, running_loss / 500, accuracy
                    ))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './weights/LeNet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
