import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import HandGestureCNN


# Define the transformation
transform = transforms.Compose([
    transforms.Grayscale(),  # Ensure images are grayscale
    transforms.Resize((128, 128)),  # Resize images to a fixed size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalization for grayscale images
])

# Load the datasets
train_dataset = datasets.ImageFolder(root='/Users/kristidodaj/Desktop/Number Recognition/dataset/train', transform=transform)
test_dataset = datasets.ImageFolder(root='/Users/kristidodaj/Desktop/Number Recognition/dataset/test', transform=transform)


# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Import and initialize the model
num_classes = len(train_dataset.classes)
model = HandGestureCNN.HandGestureCNN(num_classes=num_classes)  # Initialize with the correct number of classes

# Loss function and optimizer with L2 regularization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # Added L2 regularization

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Training loop
num_epochs = 4  
for epoch in range(num_epochs):
    model.train()
    for images, labels in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()  # Update the learning rate

    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch {epoch+1}, Test Accuracy: {correct / total}')

# Save the model state
torch.save(model.state_dict(), 'hand_gesture_model.pth')