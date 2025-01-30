import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

# Set the base directory for dataset and model storage
base_dir = os.getcwd()

# Dataset path
dataset_dir = os.path.join(base_dir, "dataset")
test_dir = os.path.join(dataset_dir, "Test")

# Check if Test folder exists
if not os.path.exists(test_dir):
    print(f"Test folder is missing. Make sure the 'Test' folder exists in the 'dataset' directory.")
    exit()

# Define transformations (same as the training transform)
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Load the test dataset
test_dataset = ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the saved model
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

# Load the model weights (from epoch 10)
model_dir = os.path.join(base_dir, "saved_model")
model.load_state_dict(torch.load(os.path.join(model_dir, "fruit_classifier_epoch_10.pt")))
model.eval()

# Get the class names (labels)
class_names = test_dataset.classes

# Pick 5 random images from the test dataset
random_images_indices = random.sample(range(len(test_dataset)), 5)

# Plot the images and their predictions
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for i, idx in enumerate(random_images_indices):
    image, label = test_dataset[idx]
    
    # Make a prediction
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
    
    predicted_label = class_names[predicted.item()]
    true_label = class_names[label]
    
    # Display the image and prediction
    axes[i].imshow(np.transpose(image.numpy(), (1, 2, 0)))  # Convert to HWC format for plt.imshow
    axes[i].set_title(f"Pred: {predicted_label}\nTrue: {true_label}")
    axes[i].axis('off')

# Display the plot
plt.tight_layout()
plt.show()
