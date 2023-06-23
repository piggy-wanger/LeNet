import torch
import torchvision.transforms as transforms

from PIL import Image
from models.LeNet import LeNet


def main():
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    classes = (
        'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    )

    net = LeNet()
    net.load_state_dict(torch.load(''))

    img = Image.open('')
    img = transform(img)  # [C, H, W]
    img = torch.unsqueeze(img, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(img)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
