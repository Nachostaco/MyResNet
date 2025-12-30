import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 1000) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 64
        #first layer 7x7 conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #max pool
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #Blocks for ResNet18
        self.layer1 = self._make_layer(block, 64, layers[0], 2)
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        #Avg Pool and Fc
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, stride, num_block):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels

        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.max_pool1(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])


def main():
    model = ResNet18()
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print(y.shape)



if __name__ == "__main__":
    main()
