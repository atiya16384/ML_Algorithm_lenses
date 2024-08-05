# Updated Model Training with Enhanced Data Augmentation and Learning Rate Scheduler
import fitz  # PyMuPDF
import os
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
import firebase_admin
from firebase_admin import credentials, storage, firestore
import time
from google.api_core.exceptions import ServiceUnavailable
from scipy.ndimage import gaussian_filter, map_coordinates
import csv
import shutil
import requests  # Added import for requests

# Firebase setup
cred = credentials.Certificate("/Users/atiyamahboob/Documents/ML_Algorithm_Lenses/capturelensapp-firebase-adminsdk-tl5tq-dc854ec4d5.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'capturelensapp.appspot.com'
})
bucket = storage.bucket()
db = firestore.client()

# Ensure these directories exist
extracted_dir = 'extracted_engravings'
augmented_dir = 'augmented_symbols'

# Create directories if they don't exist
os.makedirs(extracted_dir, exist_ok=True)
os.makedirs(augmented_dir, exist_ok=True)

# Clear Firebase Storage directories
def clear_firebase_directory(directory):
    blobs = bucket.list_blobs(prefix=directory)
    for blob in blobs:
        blob.delete()
        print(f'Deleted {blob.name} from Firebase Storage')

# Clear Firestore collection
def clear_firestore_collection(collection_name):
    docs = db.collection(collection_name).stream()
    for doc in docs:
        doc.reference.delete()
        print(f'Deleted document {doc.id} from Firestore collection {collection_name}')

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
def upload_directory_to_firebase(local_dir, remote_dir):
    for root, _, files in os.walk(local_dir):
        for file in files:
            if not (file.endswith(".png") or file.endswith(".jpeg")):
                continue  # Skip non-image files
            local_path = os.path.join(root, file)
            remote_path = os.path.join(remote_dir, os.path.relpath(local_path, local_dir))
            blob = bucket.blob(remote_path)
            success = False
            retries = 3
            while not success and retries > 0:
                try:
                    blob.upload_from_filename(local_path)
                    print(f'File {local_path} uploaded to {remote_path}')
                    success = True
                except (ServiceUnavailable, ConnectionError) as e:
                    print(f'Error uploading {local_path}: {e}')
                    retries -= 1
                    time.sleep(5)  # Wait before retrying
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

# Data augmentation with elastic transformation
def augment_images(original_dir, augmented_dir):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    elastic = iaa.ElasticTransformation(alpha=15.0, sigma=5.0)  # Adjusted parameters

    for filename in os.listdir(original_dir):
        if filename.endswith(".png") or filename.endswith(".jpeg"):
            img_path = os.path.join(original_dir, filename)
            try:
                img = load_img(img_path)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)

                img_dir = os.path.join(augmented_dir, os.path.splitext(filename)[0])
                os.makedirs(img_dir, exist_ok=True)

                i = 0
                for batch in datagen.flow(x, batch_size=1):
                    # Apply elastic transformation
                    batch[0] = elastic(image=batch[0].astype(np.uint8))
                    save_path = os.path.join(img_dir, f'aug_{i}.png')
                    img_to_save = Image.fromarray(batch[0].astype(np.uint8))
                    img_to_save.save(save_path)
                    i += 1
                    if i >= 20:  # Generate exactly 20 augmented images per original image
                        break

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
def upload_augmented_images_to_firestore(augmented_dir):
    for folder_name in os.listdir(augmented_dir):
        folder_path = os.path.join(augmented_dir, folder_name)
        if os.path.isdir(folder_path):
            image_urls = []
            for file_name in os.listdir(folder_path):
                if (file_name.endswith(".png") or file_name.endswith(".jpeg")) and len(image_urls) < 20:
                    local_path = os.path.join(folder_path, file_name)
                    remote_path = os.path.join('augmented_symbols', folder_name, file_name)
                    blob = bucket.blob(remote_path)
                    success = False
                    retries = 3
                    while not success and retries > 0:
                        try:
                            blob.upload_from_filename(local_path, timeout=300)  # Increase timeout to 300 seconds
                            print(f'File {local_path} uploaded to {remote_path}')
                            image_url = blob.public_url
                            image_urls.append(image_url)
                            success = True
                        except (ServiceUnavailable, ConnectionError, requests.exceptions.ReadTimeout) as e:
                            print(f'Error uploading {local_path}: {e}')
                            retries -= 1
                            time.sleep(5)  # Wait before retrying
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
def upload_csv_to_firebase(local_file, remote_file):
    blob = bucket.blob(remote_file)
    success = False
    retries = 3
    while not success and retries > 0:
        try:
            blob.upload_from_filename(local_file)
            print(f'CSV file {local_file} uploaded to {remote_file}')
            success = True
        except (ServiceUnavailable, ConnectionError) as e:
            print(f'Error uploading CSV file {local_file}: {e}')
            retries -= 1
            time.sleep(5)  # Wait before retrying
    if not success:
        print(f'Failed to upload CSV file {local_file} after retries.')

upload_csv_to_firebase(output_csv_file, 'file_label_mapping.csv')

# Check if the dataset is loaded correctly
if len(images) == 0 or len(labels) == 0:
    print("No images or labels found. Please check the augmentation step.")
else:
    images = images.astype('float32') / 255.0
    images = np.expand_dims(images, axis=-1)  # Add channel dimension

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Online Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # Build CNN model
    model = Sequential([
        Input(shape=(64, 64, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(len(np.unique(labels)), activation='softmax')
    ])

    # Define the optimizer with a specific learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Define callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    
    lr_schedule = LearningRateScheduler(lr_scheduler)

    # Train the model
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              epochs=100,
              validation_data=(X_test, y_test),
              callbacks=[early_stopping, reduce_lr, model_checkpoint, lr_schedule])

    # Load the best model
    model.load_weights('best_model.keras')

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Set - Loss: {loss}, Accuracy: {accuracy}")

    model.save('symbol_classifier.keras')
