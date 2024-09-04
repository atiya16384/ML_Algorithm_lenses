import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, UnidentifiedImageError
import torch.nn.functional as F

# Define the PyTorch model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load images and labels
def load_images_and_labels(augmented_dir):
    images = []
    labels = []
    label_map = {}  # Mapping from directory name to label
    label_counter = 0

    for root, dirs, files in os.walk(augmented_dir):
        for filename in files:
            if filename.endswith(".png") or filename.endswith(".jpeg"):
                img_path = os.path.join(root, filename)
                try:
                    img = Image.open(img_path)
                    img = img.resize((64, 64))  # Resize to a fixed size
                    img = img.convert('L')  # Convert to grayscale
                    img = np.array(img)
                    images.append(img)

                    label = os.path.basename(root)  # Use the directory name as the label
                    if label not in label_map:
                        label_map[label] = label_counter
                        label_counter += 1
                    labels.append(label_map[label])

                except UnidentifiedImageError:
                    print(f"Unidentified image error at {img_path}, skipping.")
                except OSError:
                    print(f"OS error at {img_path}, skipping.")

    print(f"Loaded {len(images)} images with {len(labels)} labels.")
    images = np.array(images)
    labels = np.array(labels)
    num_classes = len(label_map)  # Calculate the number of unique classes
    return images, labels, label_map, num_classes

# Example function for training the model
def train_model():
    # Load images and labels
    augmented_dir = 'augmented_symbols'
    images, labels, label_map, num_classes = load_images_and_labels(augmented_dir)

    # Create the dataset and dataloaders
    dataset = CustomDataset(images, labels, transform=transform)
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Training and evaluation setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2, min_lr=1e-5)

    num_epochs = 50
    early_stopping_patience = 10
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = train_correct / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(test_loader)
        val_accuracy = val_correct / len(test_loader.dataset)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate on the test set
    model.eval()
    test_loss = 0
    test_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = test_correct / len(test_loader.dataset)

    print(f'Test Set - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')

# Make num_classes accessible for import
_, _, _, num_classes = load_images_and_labels('augmented_symbols')

if __name__ == "__main__":
    train_model()
