import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define transformations
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Dataset download check
dataset_dir = os.path.join(os.getcwd(), "dataset", "fruits")
if not os.path.exists(dataset_dir):
    print(f"Dataset not found at {dataset_dir}. Please download it manually from Kaggle and place it in the 'dataset/fruits' folder.")
    exit()

# Load dataset
dataset = ImageFolder(root=dataset_dir, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# Define the CNN model
class FruitClassifier(nn.Module):
    def __init__(self):
        super(FruitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 50 * 50, 131)  # 131 classes (fruits)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 50 * 50)
        x = self.fc1(x)
        return x

# Initialize the model
model = FruitClassifier()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set model directory relative to the project
model_dir = os.path.join(os.getcwd(), "saved_model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Training loop with tqdm progress bar
num_epochs = 10
interrupt_training = False  # Flag to check if the user wants to interrupt the training loop

for epoch in range(num_epochs):
    epoch_train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_train_loss:.4f}", leave=False):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        
        # Check if the user wants to interrupt the training loop
        if interrupt_training:
            break
    
    if interrupt_training:
        print("Training interrupted.")
        break
    
    # Calculate average loss for the epoch
    avg_epoch_train_loss = epoch_train_loss / len(train_loader)
    
    # Print average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_epoch_train_loss:.4f}")

    # Save model parameters after each epoch
    torch.save(model.state_dict(), os.path.join(model_dir, f"fruit_classifier_epoch_{epoch+1}.pt"))

# Validation
model.eval()
correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

val_accuracy = 100 * correct / total

# Save validation accuracy
with open(os.path.join(model_dir, "validation_accuracy.txt"), "w") as f:
    f.write(f"Validation Accuracy: {val_accuracy:.2f}%")

# Plot confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))

plt.show()
