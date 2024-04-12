import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.labels = np.loadtxt(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, f"{str(idx).zfill(4)}.png")
        image = Image.open(img_name)
        label = self.labels[idx]

        # Mã hóa nhãn
        if label < 0:
            encoded_label = 0  # Rẽ trái
        elif label == 0:
            encoded_label = 1  # Đi thẳng
        else:
            encoded_label = 2  # Rẽ phải

        if self.transform:
            image = self.transform(image)

        return image, encoded_label

# Hàm chuyển đổi nhãn số thành tên nhãn
def label_to_str(label):
    if label == 0:
        return "Rẽ trái"
    elif label == 1:
        return "Đi thẳng"
    elif label == 2:
        return "Rẽ phải"

# Transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Dataset và DataLoader
dataset = CustomDataset(r'D:\new\frames_num', r'D:\new\label.txt', transform=transform)
train_set, test_set = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*16*16, 64)
        self.fc2 = nn.Linear(64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*16*16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()

# Loss function và Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).float()
    return correct.mean()

# Training loop
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(10):
    model.train()
    total_train_loss = 0
    correct_train = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        correct_train += calculate_accuracy(outputs, labels)
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracy = correct_train / len(train_loader)
    train_accuracies.append(train_accuracy)

    model.eval()
    total_test_loss = 0
    correct_test = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            correct_test += calculate_accuracy(outputs, labels)
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    test_accuracy = correct_test / len(test_loader)
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

# Save model
torch.save(model.state_dict(), 'classifier_model.h5')

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot test results
def plot_test_results(loader, model, num_images=5):
    model.eval()
    images, labels = next(iter(loader))
    with torch.no_grad():
        outputs = model(images)
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i].permute(1, 2, 0))
        _, predicted = torch.max(outputs, 1)
        plt.title(f"Đúng: {label_to_str(labels[i])}\nDự đoán: {label_to_str(predicted[i])}")
        plt.axis('off')
    plt.show()

plot_test_results(test_loader, model)
