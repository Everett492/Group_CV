import jittor
import os
from jittor import nn, Module
import numpy as np
from tqdm import tqdm
from jittor.dataset import CIFAR100
import jittor.transform as transforms

# Parameters
data_root = '~/.cache/'
batch_size = 64
lr = 1e-2
momentum = 0.9
epochs = 200
img_size = 64

jittor.flags.use_cuda = 1

transform = transforms.Compose([
    transforms.RandomCrop(28),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(size=img_size),
    transforms.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Use Cifar100 dataset
train_loader = CIFAR100(os.path.expanduser(data_root), train=True, transform=transform, download=True).set_attrs(batch_size=batch_size, shuffle=True)
test_loader = CIFAR100(os.path.expanduser(data_root), train=False, transform=transform, download=True).set_attrs(shuffle=True)
print(f'Dataset loaded with trainging dataset with a batchsize of {train_loader.batch_size}')

# Model Defination
# Basic block of ResNet 34
class BasicBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.residual_func = nn.Sequential(
            nn.Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1 ,bias=False),
            nn.BatchNorm(out_channels),
            nn.ReLU(),
            nn.Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm(out_channels)
        )

        self.short_cut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.short_cut = nn.Sequential(
                nn.Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(out_channels)
            )

    def execute(self, x):
        output = self.residual_func(x) + self.short_cut(x)
        output = nn.relu(output)
        return output

# ResNet 34
class ResNet34(Module):
    def __init__(self, block, num_classes=100):
        super(ResNet34, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm(64),
            nn.ReLU()
        )
        self.conv2_x = self.make_layer(block, 64, 3, 1)
        self.conv3_x = self.make_layer(block, 128, 4, 2)
        self.conv4_x = self.make_layer(block, 256, 6, 2)
        self.conv5_x = self.make_layer(block, 512, 3, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)
    
    def execute(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)
        return output

def train(model, train_loader, loss_func, optimizer, epoch, f):
    model.train()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        optimizer.step(loss)
        data = f'Train epoch {epoch} [{(batch_idx + 1) * batch_size}/{len(train_loader)}]\tLoss: {loss.numpy()[0]}'
        print(data)
        f.write(data + '\n')

all_acc = []

def test(model, test_loader, epoch, f):
    model.eval()

    correct = 0
    total = 0
    accuracy = []

    for batch_idx, (inputs, labels) in tqdm(enumerate(test_loader)):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        acc = np.sum(labels.data==pred)
        correct += acc
        total += batch_size
        accuracy.append(acc / batch_size * 100)
    
    all_acc.append(sum(accuracy) / len(accuracy))
    data = f'Test epoch {epoch} accuracy: {sum(accuracy) / len(accuracy)}%'
    print(data)
    f.write(data + '\n')

if __name__ == '__main__':
    with open('Result_3_14_2.txt', 'w') as f:
        f.write(f'image_size: {img_size}, conv1 kernel_size: 5x5, epochs: {epochs}\n')
        resnet34 = ResNet34(BasicBlock, 100)
        loss_func = nn.CrossEntropyLoss()
        optimizer = nn.SGD(params=resnet34.parameters(), lr=lr, momentum=momentum)

        for epoch in range(epochs):
            train(resnet34, train_loader, loss_func, optimizer, epoch, f)
            test(resnet34, test_loader, epoch, f)
            if(all_acc[len(all_acc) - 1] >= max(all_acc)):
                resnet34.save('./resnet34_3_14_2.pkl')

        print(f'Accuracy on all data in each epoch: {all_acc}')
        f.write(f'Accuracy on all data in each epoch: {all_acc}\n')