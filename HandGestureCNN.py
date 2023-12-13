import torch
import torch.nn as nn
import torch.nn.functional as F

class HandGestureCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(HandGestureCNN, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  # Additional Conv Layer
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)  # Adjusted for additional conv layer
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.4)  # Adjusted dropout rate

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))  # Passing through the additional layer
        x = x.view(-1, 256 * 8 * 8)  # Adjusted for additional conv layer
        x = self.dropout(F.leaky_relu(self.fc_bn1(self.fc1(x))))
        x = self.fc2(x)
        return x
