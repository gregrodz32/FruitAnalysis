import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Load the Test dataset
test_dir = "/Users/greg/Downloads/archive/fruits-360_dataset/fruits-360/Test"  # Adjust the path to the Test folder
test_dataset = ImageFolder(root=test_dir, transform=transform)

# Create a data loader for the test set
batch_size = 1  # Set batch size to 1 for testing individual images
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load the trained model
model_dir = "/Users/greg/Downloads/archive/fruits-360_dataset/fruits-360/saved_model"
model_path = os.path.join(model_dir, "fruit_classifier_epoch_10.pt")  # Adjust epoch number if needed
model = torch.load(model_path)
model.eval()

# Class labels
class_labels = test_dataset.classes

# Function to predict the class of an image
def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    predicted_label = class_labels[predicted.item()]
    return predicted_label

# Perform simple test for fruit identification
num_correct = 0
total_images = len(test_dataset)
for images, labels in test_loader:
    predicted_label = predict_image(test_loader.dataset.samples[0][0])
    ground_truth_label = test_loader.dataset.classes[labels[0]]
    if predicted_label == ground_truth_label:
        num_correct += 1

# Calculate accuracy
accuracy = (num_correct / total_images) * 100

# Print results
print(f"Total Images: {total_images}")
print(f"Correctly Identified: {num_correct}")
print(f"Accuracy: {accuracy:.2f}%")
