import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, UnidentifiedImageError
import imgaug.augmenters as iaa
from scipy.ndimage import gaussian_filter, map_coordinates
import csv
import shutil
import torch.nn.functional as F

# Ensure these directories exist
extracted_dir = 'extracted_engravings'  # Directory containing the original images
augmented_dir = 'augmented_symbols'     # Directory to save augmented images
os.makedirs(augmented_dir, exist_ok=True)  # Create the augmented directory if it doesn't exist

# Clear local directory
def clear_local_directory(directory):
    """
    Deletes all files and subdirectories in the specified directory.
    """
    for root, dirs, files in os.walk(directory):  # Traverse the directory
        for file in files:
            file_path = os.path.join(root, file)  # Full file path
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):  # Check if it's a file or symlink
                    os.unlink(file_path)  # Delete the file or symlink
                elif os.path.isdir(file_path):  # If it's a directory
                    shutil.rmtree(file_path)  # Recursively delete the directory
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    print(f'Cleared local directory: {directory}')

# Define a function to perform elastic transformation
def elastic_transform(image, alpha, sigma):
    """
    Applies elastic transformation to an image. This transformation
    distorts the image by moving pixels around in a smooth, random fashion.
    """
    random_state = np.random.RandomState(None)  # Create a random state for reproducibility
    shape = image.shape  # Get the shape of the image

    # Random displacement fields
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # Create a meshgrid of coordinates (x, y)
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))  # Apply the displacements

    # Map the input image to the distorted coordinates
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

# Data augmentation with elastic transformation using PyTorch's transforms
def augment_images(original_dir, augmented_dir):
    """
    Performs data augmentation on images in the original directory.
    Augmented images are saved to the augmented directory.
    """
    base_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.Grayscale(),       # Convert images to grayscale
    ])
    
    augment_transform = transforms.Compose([
        transforms.RandomRotation(15),  # Randomly rotate the image by up to 15 degrees
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),  # Randomly crop and resize the image
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  # Random color adjustments
        transforms.ToTensor()  # Convert the image to a tensor
    ])

    elastic = iaa.ElasticTransformation(alpha=15.0, sigma=5.0)  # Elastic transformation with specific parameters

    for filename in os.listdir(original_dir):  # Iterate over each file in the original directory
        if filename.endswith((".png") or filename.endswith(".jpeg")):  # Process only PNG or JPEG images
            img_path = os.path.join(original_dir, filename)  # Full path to the image
            try:
                img = Image.open(img_path)  # Open the image
                x = np.array(img)  # Convert the image to a NumPy array

                # Create a directory for the augmented images of this specific image
                img_dir = os.path.join(augmented_dir, os.path.splitext(filename)[0])
                os.makedirs(img_dir, exist_ok=True)

                for i in range(20):  # Generate 20 augmented images per original image
                    augmented_img = elastic(image=x)  # Apply elastic transformation
                    augmented_img = Image.fromarray(augmented_img.astype(np.uint8))  # Convert back to an image
                    augmented_img = base_transform(augmented_img)  # Apply base transformations
                    augmented_img = augment_transform(augmented_img)  # Apply augment transformations
                    save_path = os.path.join(img_dir, f'aug_{i}.png')  # Define save path
                    augmented_img = transforms.ToPILImage()(augmented_img)  # Convert tensor back to PIL image
                    augmented_img.save(save_path)  # Save the augmented image

            except UnidentifiedImageError:
                print(f"Unidentified image error at {img_path}, skipping.")  # Handle unrecognized images

    print(f"Augmented images saved to directory: {augmented_dir}")

# Clear the local augmented_symbols directory before augmenting
clear_local_directory(augmented_dir)

# Perform data augmentation
augment_images(extracted_dir, augmented_dir)

# Load images and labels
def load_images_and_labels(augmented_dir):
    """
    Loads images and their corresponding labels from the augmented directory.
    Returns images, labels, and a file-label mapping for later use.
    """
    images = []  # List to store image data
    labels = []  # List to store labels
    label_map = {}  # Dictionary to map directory names to numeric labels
    file_label_mapping = []  # Store file path and label name for CSV export
    label_counter = 0  # Counter for assigning unique numeric labels

    for root, dirs, files in os.walk(augmented_dir):  # Traverse the augmented directory
        for filename in files:
            if filename.endswith(".png") or filename.endswith(".jpeg"):  # Process only PNG or JPEG images
                img_path = os.path.join(root, filename)  # Full path to the image
                try:
                    img = Image.open(img_path)  # Open the image
                    img = img.resize((64, 64))  # Resize the image to 64x64
                    img = img.convert('L')  # Convert the image to grayscale
                    img = np.array(img)  # Convert the image to a NumPy array
                    images.append(img)  # Add the image to the list

                    label = os.path.basename(root)  # Use the directory name as the label
                    if label not in label_map:  # If the label is new, add it to the map
                        label_map[label] = label_counter
                        label_counter += 1
                    labels.append(label_map[label])  # Add the corresponding numeric label
                    file_label_mapping.append((img_path, label))  # Save file path and label

                except UnidentifiedImageError:
                    print(f"Unidentified image error at {img_path}, skipping.")  # Handle unrecognized images
                except OSError:
                    print(f"OS error at {img_path}, skipping.")  # Handle OS-related errors

    print(f"Loaded {len(images)} images with {len(labels)} labels.")
    images = np.array(images)  # Convert list of images to NumPy array
    labels = np.array(labels)  # Convert list of labels to NumPy array
    return images, labels, file_label_mapping

# Load images and labels from the augmented directory
images, labels, file_label_mapping = load_images_and_labels(augmented_dir)

# Write file path to label mappings to a CSV file
output_csv_file = 'file_label_mapping.csv'
with open(output_csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)  # Create a CSV writer object
    writer.writerow(['File Path', 'Label'])  # Write the header row
    writer.writerows(file_label_mapping)  # Write all file path and label pairs

# Define the PyTorch model
class CNN(nn.Module):
    """
    Defines a Convolutional Neural Network (CNN) with 4 convolutional layers,
    batch normalization, dropout, and fully connected layers.
    """
    def __init__(self, num_classes):
        super(CNN, self).__init__()  # Call the constructor of the base class (nn.Module)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 1st convolutional layer: 1 input channel, 64 output channels
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization for the 1st conv layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 2nd convolutional layer
        self.bn2 = nn.BatchNorm2d(128)  # Batch normalization for the 2nd conv layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 3rd convolutional layer
        self.bn3 = nn.BatchNorm2d(256)  # Batch normalization for the 3rd conv layer
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 4th convolutional layer
        self.bn4 = nn.BatchNorm2d(512)  # Batch normalization for the 4th conv layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling to reduce feature map size
        self.fc1 = nn.Linear(512, 1024)  # 1st fully connected layer
        self.dropout = nn.Dropout(0.5)  # Dropout layer for regularization (drop 50% of neurons during training)
        self.fc2 = nn.Linear(1024, num_classes)  # 2nd fully connected layer, output layer

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Apply 1st conv layer, batch norm, and ReLU activation
        x = F.max_pool2d(x, 2)  # Apply max pooling with a 2x2 window
        x = F.relu(self.bn2(self.conv2(x)))  # Apply 2nd conv layer, batch norm, and ReLU activation
        x = F.max_pool2d(x, 2)  # Apply max pooling with a 2x2 window
        x = F.relu(self.bn3(self.conv3(x)))  # Apply 3rd conv layer, batch norm, and ReLU activation
        x = F.max_pool2d(x, 2)  # Apply max pooling with a 2x2 window
        x = F.relu(self.bn4(self.conv4(x)))  # Apply 4th conv layer, batch norm, and ReLU activation
        x = F.max_pool2d(x, 2)  # Apply max pooling with a 2x2 window
        x = self.global_pool(x)  # Apply global average pooling
        x = torch.flatten(x, 1)  # Flatten the tensor to feed into the fully connected layer
        x = F.relu(self.fc1(x))  # Apply 1st fully connected layer and ReLU activation
        x = self.dropout(x)  # Apply dropout for regularization
        x = self.fc2(x)  # Apply 2nd fully connected layer (output layer)
        return x  # Return the final output

# Transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize all images to 64x64 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images (mean=0.5, std=0.5)
])

# Custom dataset class
class CustomDataset(Dataset):
    """
    Custom dataset class to load images and labels.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images  # Store images
        self.labels = labels  # Store labels
        self.transform = transform  # Store any transformations to apply
    
    def __len__(self):
        return len(self.images)  # Return the total number of images
    
    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])  # Convert the image to a PIL image
        label = self.labels[idx]  # Get the corresponding label
        if self.transform:
            image = self.transform(image)  # Apply transformations, if any
        return image, label  # Return the image and label

# Create the dataset and dataloaders
dataset = CustomDataset(images, labels, transform=transform)  # Initialize the custom dataset with images and labels
train_size = int(0.85 * len(dataset))  # Define the training set size (85% of the dataset)
test_size = len(dataset) - train_size  # Define the test set size (remaining 15%)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # Split the dataset into training and testing sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # DataLoader for training data
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # DataLoader for testing data

# Training and evaluation setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else use CPU
num_classes = len(np.unique(labels))  # Determine the number of unique classes in the dataset
model = CNN(num_classes).to(device)  # Initialize the CNN model and move it to the device (GPU/CPU)

criterion = nn.CrossEntropyLoss()  # Define the loss function (cross-entropy loss)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Define the optimizer (Adam with weight decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2, min_lr=1e-5)  # Learning rate scheduler to reduce LR when validation loss plateaus

num_epochs = 50  # Set the number of epochs for training
early_stopping_patience = 10  # Set patience for early stopping
best_val_loss = float('inf')  # Initialize the best validation loss to infinity
early_stopping_counter = 0  # Initialize the early stopping counter

for epoch in range(num_epochs):  # Loop over each epoch
    model.train()  # Set the model to training mode
    train_loss = 0  # Initialize training loss for this epoch
    train_correct = 0  # Initialize count of correct predictions in the training set
    
    for images, labels in train_loader:  # Loop over each batch of training data
        images, labels = images.to(device), labels.to(device)  # Move images and labels to the GPU/CPU
        optimizer.zero_grad()  # Zero the gradients from the previous step
        outputs = model(images)  # Forward pass: get model predictions
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model parameters
        train_loss += loss.item()  # Accumulate the training loss
        _, predicted = torch.max(outputs, 1)  # Get the index of the maximum output (predicted label)
        train_correct += (predicted == labels).sum().item()  # Count correct predictions
    
    train_loss /= len(train_loader)  # Compute average training loss
    train_accuracy = train_correct / len(train_loader.dataset)  # Compute training accuracy
    
    model.eval()  # Set the model to evaluation mode
    val_loss = 0  # Initialize validation loss for this epoch
    val_correct = 0  # Initialize count of correct predictions in the validation set
    
    with torch.no_grad():  # No need to track gradients during evaluation
        for images, labels in test_loader:  # Loop over each batch of validation data
            images, labels = images.to(device), labels.to(device)  # Move images and labels to the GPU/CPU
            outputs = model(images)  # Forward pass: get model predictions
            loss = criterion(outputs, labels)  # Compute the loss
            val_loss += loss.item()  # Accumulate the validation loss
            _, predicted = torch.max(outputs, 1)  # Get the index of the maximum output (predicted label)
            val_correct += (predicted == labels).sum().item()  # Count correct predictions
    
    val_loss /= len(test_loader)  # Compute average validation loss
    val_accuracy = val_correct / len(test_loader.dataset)  # Compute validation accuracy
    
    scheduler.step(val_loss)  # Adjust the learning rate based on validation loss
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')  # Print metrics for this epoch
    
    # Early stopping check
    if val_loss < best_val_loss:  # If the validation loss improves
        best_val_loss = val_loss  # Update the best validation loss
        torch.save(model.state_dict(), 'best_model.pth')  # Save the model
        early_stopping_counter = 0  # Reset early stopping counter
    else:
        early_stopping_counter += 1  # Increment early stopping counter
        if early_stopping_counter >= early_stopping_patience:  # If no improvement for a while
            print("Early stopping triggered")  # Trigger early stopping
            break  # Exit the training loop

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))  # Load the best model from file

# Evaluate on the test set
model.eval()  # Set the model to evaluation mode
test_loss = 0  # Initialize test loss
test_correct = 0  # Initialize count of correct predictions on the test set

with torch.no_grad():  # No need to track gradients during evaluation
    for images, labels in test_loader:  # Loop over each batch of test data
        images, labels = images.to(device), labels.to(device)  # Move images and labels to the GPU/CPU
        outputs = model(images)  # Forward pass: get model predictions
        loss = criterion(outputs, labels)  # Compute the loss
        test_loss += loss.item()  # Accumulate the test loss
        _, predicted = torch.max(outputs, 1)  # Get the index of the maximum output (predicted label)
        test_correct += (predicted == labels).sum().item()  # Count correct predictions

test_loss /= len(test_loader)  # Compute average test loss
test_accuracy = test_correct / len(test_loader.dataset)  # Compute test accuracy

print(f'Test Set - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')  # Print test metrics
