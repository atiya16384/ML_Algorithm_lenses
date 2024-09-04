import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, UnidentifiedImageError
import imgaug.augmenters as iaa
import firebase_admin
from firebase_admin import credentials, storage, firestore
import time
from google.api_core.exceptions import ServiceUnavailable, ResourceExhausted
from scipy.ndimage import gaussian_filter, map_coordinates
import csv
import shutil
import requests
import torch.nn.functional as F
import random

# Firebase setup
cred = credentials.Certificate("/Users/atiya/Documents/ML_Algorithm_lenses/capturelensapp-firebase-adminsdk-tl5tq-dc854ec4d5.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'capturelensapp.appspot.com'
})
bucket = storage.bucket()
db = firestore.client()

# Ensure these directories exist
extracted_dir = 'extracted_engravings'
augmented_dir = 'augmented_symbols'
os.makedirs(augmented_dir, exist_ok=True)

# Clear Firebase Storage directories
def clear_firebase_directory(directory, max_retries=5):
    blobs = bucket.list_blobs(prefix=directory)
    for blob in blobs:
        success = False
        retries = 0
        while not success and retries < max_retries:
            try:
                blob.delete()
                print(f'Deleted {blob.name} from Firebase Storage')
                success = True
            except (ServiceUnavailable, requests.exceptions.ReadTimeout, ResourceExhausted) as e:
                print(f'Error deleting {blob.name}: {e}')
                retries += 1
                wait_time = (2 ** retries) + random.uniform(0, 1)
                print(f'Waiting {wait_time} seconds before retrying...')
                time.sleep(wait_time)
        if not success:
            print(f'Failed to delete {blob.name} after retries.')

# Clear Firestore collection
def clear_firestore_collection(collection_name, max_retries=5):
    docs = db.collection(collection_name).stream()
    for doc in docs:
        success = False
        retries = 0
        while not success and retries < max_retries:
            try:
                doc.reference.delete()
                print(f'Deleted document {doc.id} from Firestore collection {collection_name}')
                success = True
            except (ServiceUnavailable, requests.exceptions.ReadTimeout, ResourceExhausted) as e:
                print(f'Error deleting document {doc.id}: {e}')
                retries += 1
                wait_time = (2 ** retries) + random.uniform(0, 1)  # Exponential backoff with jitter
                print(f'Waiting {wait_time} seconds before retrying...')
                time.sleep(wait_time)
        if not success:
            print(f'Failed to delete document {doc.id} after retries.')

# Clear local directory
def clear_local_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    print(f'Cleared local directory: {directory}')

# Upload directories to Firebase Storage
def upload_directory_to_firebase(local_dir, remote_dir, max_retries=5):
    for root, _, files in os.walk(local_dir):
        for file in files:
            if not (file.endswith(".png") or file.endswith(".jpeg")):
                continue  # Skip non-image files
            local_path = os.path.join(root, file)
            remote_path = os.path.join(remote_dir, os.path.relpath(local_path, local_dir))
            blob = bucket.blob(remote_path)
            success = False
            retries = 0
            while not success and retries < max_retries:
                try:
                    blob.upload_from_filename(local_path, timeout=300)  # Increase timeout
                    print(f'File {local_path} uploaded to {remote_path}')
                    success = True
                except (ServiceUnavailable, ConnectionError, requests.exceptions.ReadTimeout, ResourceExhausted) as e:
                    print(f'Error uploading {local_path}: {e}')
                    retries += 1
                    wait_time = (2 ** retries) + random.uniform(0, 1)
                    print(f'Waiting {wait_time} seconds before retrying...')
                    time.sleep(wait_time)
            if not success:
                print(f'Failed to upload {local_path} after retries.')

# Define a function to perform elastic transformation
def elastic_transform(image, alpha, sigma):
    random_state = np.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

# Data augmentation with elastic transformation using PyTorch's transforms
def augment_images(original_dir, augmented_dir):
    base_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
    ])
    
    augment_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.ToTensor()
    ])

    elastic = iaa.ElasticTransformation(alpha=15.0, sigma=5.0)  # Adjusted parameters

    for filename in os.listdir(original_dir):
        if filename.endswith((".png") or filename.endswith(".jpeg")):
            img_path = os.path.join(original_dir, filename)
            try:
                img = Image.open(img_path)
                x = np.array(img)

                img_dir = os.path.join(augmented_dir, os.path.splitext(filename)[0])
                os.makedirs(img_dir, exist_ok=True)

                for i in range (20):  # Generate 20 augmented images per original image
                    augmented_img = elastic(image=x)
                    augmented_img = Image.fromarray(augmented_img.astype(np.uint8))
                    augmented_img = base_transform(augmented_img)
                    augmented_img = augment_transform(augmented_img)
                    save_path = os.path.join(img_dir, f'aug_{i}.png')
                    augmented_img = transforms.ToPILImage()(augmented_img)
                    augmented_img.save(save_path)

            except UnidentifiedImageError:
                print(f"Unidentified image error at {img_path}, skipping.")

    print(f"Augmented images saved to directory: {augmented_dir}")

# Clear the Firebase directories before uploading new files
clear_firebase_directory('extracted_engravings')
clear_firebase_directory('augmented_symbols')

# Clear the Firestore collection
clear_firestore_collection('augmented_symbols_database')

# Clear the local augmented_symbols directory
clear_local_directory(augmented_dir)

augment_images(extracted_dir, augmented_dir)

# Upload extracted_engravings and augmented_symbols directories
upload_directory_to_firebase(extracted_dir, 'extracted_engravings')
upload_directory_to_firebase(augmented_dir, 'augmented_symbols')

# Upload augmented images to Firestore
def upload_augmented_images_to_firestore(augmented_dir, max_retries=5):
    for folder_name in os.listdir(augmented_dir):
        folder_path = os.path.join(augmented_dir, folder_name)
        if os.path.isdir(folder_path):
            image_urls = []
            for file_name in os.listdir(folder_path):
                if (file_name.endswith(".png") or file_name.endswith(".jpeg")) and len(image_urls) < 12:
                    local_path = os.path.join(folder_path, file_name)
                    remote_path = os.path.join('augmented_symbols', folder_name, file_name)
                    blob = bucket.blob(remote_path)
                    success = False
                    retries = 0
                    while not success and retries < max_retries:
                        try:
                            blob.upload_from_filename(local_path, timeout=300)  # Increase timeout to 300 seconds
                            print(f'File {local_path} uploaded to {remote_path}')
                            image_url = blob.public_url
                            image_urls.append(image_url)
                            success = True
                        except (ServiceUnavailable, ConnectionError, requests.exceptions.ReadTimeout, ResourceExhausted) as e:
                            print(f'Error uploading {local_path}: {e}')
                            retries += 1
                            wait_time = (2 ** retries) + random.uniform(0, 1)
                            print(f'Waiting {wait_time} seconds before retrying...')
                            time.sleep(wait_time)
                    if not success:
                        print(f'Failed to upload {local_path} after retries.')

            # Store image URLs in Firestore
            if image_urls:
                doc_ref = db.collection('augmented_symbols_database').document(folder_name)
                doc_ref.set({
                    'folder_name': folder_name,
                    'image_urls': image_urls
                })
                print(f'Metadata for folder {folder_name} stored in Firestore.')

# Upload augmented images and store metadata
upload_augmented_images_to_firestore(augmented_dir)

# Delete existing CSV file if it exists
csv_file_path = 'file_label_mapping.csv'
if os.path.exists(csv_file_path):
    os.remove(csv_file_path)
    print(f"Deleted existing CSV file: {csv_file_path}")

# Load images and labels
def load_images_and_labels(augmented_dir):
    images = []
    labels = []
    label_map = {}  # Mapping from directory name to label
    file_label_mapping = []  # Store file path and label name for CSV export
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
                    file_label_mapping.append((img_path, label))

                except UnidentifiedImageError:
                    print(f"Unidentified image error at {img_path}, skipping.")
                except OSError:
                    print(f"OS error at {img_path}, skipping.")

    print(f"Loaded {len(images)} images with {len(labels)} labels.")
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, file_label_mapping

# Load images and labels
images, labels, file_label_mapping = load_images_and_labels(augmented_dir)

# Write file path to label mappings to a CSV file
output_csv_file = 'file_label_mapping.csv'
with open(output_csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['File Path', 'Label'])
    writer.writerows(file_label_mapping)

# Upload the CSV file to Firebase Storage
def upload_csv_to_firebase(local_file, remote_file, max_retries=5):
    blob = bucket.blob(remote_file)
    success = False
    retries = 0
    while not success and retries < max_retries:
        try:
            blob.upload_from_filename(local_file, timeout=300)  # Increase timeout
            print(f'CSV file {local_file} uploaded to {remote_file}')
            success = True
        except (ServiceUnavailable, ConnectionError, ResourceExhausted) as e:
            print(f'Error uploading CSV file {local_file}: {e}')
            retries += 1
            wait_time = (2 ** retries) + random.uniform(0, 1)
            print(f'Waiting {wait_time} seconds before retrying...')
            time.sleep(wait_time)
    if not success:
        print(f'Failed to upload CSV file {local_file} after retries.')

upload_csv_to_firebase(output_csv_file, 'file_label_mapping.csv')

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

# Create the dataset and dataloaders
dataset = CustomDataset(images, labels, transform=transform)
train_size = int(0.85 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training and evaluation setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(np.unique(labels))
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

# Store extracted engravings and augmented symbols in Firestore
def store_images_in_firestore(extracted_dir, augmented_dir, collection_name, lenses_collection_name, max_retries=5):
    for folder_name in os.listdir(augmented_dir):
        folder_path = os.path.join(augmented_dir, folder_name)
        extracted_path = os.path.join(extracted_dir, folder_name + '.png')
        if os.path.isdir(folder_path) and os.path.isfile(extracted_path):
            image_urls = []
            success = False
            retries = 0

            # Upload extracted engraving
            blob = bucket.blob(f'{collection_name}/{folder_name}/extracted.png')
            while not success and retries < max_retries:
                try:
                    blob.upload_from_filename(extracted_path, timeout=300)
                    extracted_url = blob.public_url
                    print(f'Extracted image {extracted_path} uploaded to {blob.name}')
                    success = True
                except (ServiceUnavailable, ConnectionError, requests.exceptions.ReadTimeout, ResourceExhausted) as e:
                    print(f'Error uploading {extracted_path}: {e}')
                    retries += 1
                    wait_time = (2 ** retries) + random.uniform(0, 1)
                    print(f'Waiting {wait_time} seconds before retrying...')
                    time.sleep(wait_time)
            if not success:
                print(f'Failed to upload {extracted_path} after retries.')

            # Upload augmented symbols
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".png") or file_name.endswith(".jpeg"):
                    local_path = os.path.join(folder_path, file_name)
                    remote_path = f'{collection_name}/{folder_name}/{file_name}'
                    blob = bucket.blob(remote_path)
                    success = False
                    retries = 0
                    while not success and retries < max_retries:
                        try:
                            blob.upload_from_filename(local_path, timeout=300)
                            image_url = blob.public_url
                            image_urls.append(image_url)
                            print(f'File {local_path} uploaded to {remote_path}')
                            success = True
                        except (ServiceUnavailable, ConnectionError, requests.exceptions.ReadTimeout, ResourceExhausted) as e:
                            print(f'Error uploading {local_path}: {e}')
                            retries += 1
                            wait_time = (2 ** retries) + random.uniform(0, 1)
                            print(f'Waiting {wait_time} seconds before retrying...')
                            time.sleep(wait_time)
                    if not success:
                        print(f'Failed to upload {local_path} after retries.')

            # Store image URLs in Firestore
            if image_urls and success:
                doc_ref = db.collection(collection_name).document(folder_name)
                doc_ref.set({
                    'folder_name': folder_name,
                    'extracted_url': extracted_url,
                    'augmented_urls': image_urls
                })
                print(f'Metadata for folder {folder_name} stored in Firestore.')

                # Link to lenses collection
                lenses_docs = db.collection(lenses_collection_name).stream()
                for lenses_doc in lenses_docs:
                    lenses_doc_ref = db.collection(lenses_collection_name).document(lenses_doc.id)
                    lenses_doc_ref.update({
                        'engraving_ref': db.document(f'{collection_name}/{folder_name}')
                    })
                print(f'Linked {folder_name} in {collection_name} to {lenses_collection_name}')

store_images_in_firestore(extracted_dir, augmented_dir, 'engravings', 'lenses')
