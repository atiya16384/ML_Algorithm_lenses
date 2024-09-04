import os
import firebase_admin
from firebase_admin import credentials, storage, firestore
import time
from google.api_core.exceptions import ServiceUnavailable, ResourceExhausted
import shutil
import requests
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
                            base_wait_time = 2 ** retries  # Exponential backoff
                            jitter = random.uniform(0, 1)  # Random jitter
                            wait_time = base_wait_time + jitter  # Combine them
                            print(f'Waiting {wait_time} seconds before retrying...')  # Debugging output
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

if __name__ == "__main__":
    # Clear the Firebase directories before uploading new files
    # Uncomment if you need to clear the directories before new uploads
    # clear_firebase_directory('extracted_engravings')
    # clear_firebase_directory('augmented_symbols')

    # Clear the Firestore collection (Uncomment if needed)
    # clear_firestore_collection('augmented_symbols_database')

    # Clear the local augmented_symbols directory
    # clear_local_directory(augmented_dir)

    # Upload extracted_engravings and augmented_symbols directories
    upload_directory_to_firebase(extracted_dir, 'extracted_engravings')
    upload_directory_to_firebase(augmented_dir, 'augmented_symbols')

    # Upload augmented images and store metadata
    upload_augmented_images_to_firestore(augmented_dir)

    # Store extracted engravings and augmented symbols in Firestore
    store_images_in_firestore(extracted_dir, augmented_dir, 'engravings', 'lenses')
