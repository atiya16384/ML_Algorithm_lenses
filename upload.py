import csv
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Initialize Firebase
cred_path = "/Users/atiya/Documents/ML_Algorithm_lenses/capturelensapp-firebase-adminsdk-tl5tq-dc854ec4d5.json"

# Check if the Firebase credentials file exists
if not os.path.exists(cred_path):
    print("Firebase credentials file not found.")
    exit(1)  # Exit the script if the credentials file is not found

# Initialize the Firebase app with the credentials
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

# Create a Firestore client to interact with the Firestore database
db = firestore.client()

# Clear existing documents in the 'lenses' collection
lenses_ref = db.collection('lenses')
docs = lenses_ref.stream()
for doc in docs:
    # Delete each document in the 'lenses' collection to ensure we start with a clean slate
    doc.reference.delete()

print('Cleared existing lenses data.')  # Confirmation message that the lenses collection has been cleared

# Function to upload data from a CSV file to Firestore
def upload_csv_to_firestore(csv_path):
    # Check if the specified CSV file exists
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found.")
        return  # Exit the function if the CSV file is not found

    # Open the CSV file and read its contents
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)  # Use DictReader to read rows as dictionaries
        row_count = 0  # Initialize a counter for rows processed
        for row in reader:
            row_count += 1  # Increment the row counter
            print(f"Processing row {row_count}: {row}")

            # Clean the data by replacing None keys and values with "N/A"
            cleaned_row = {}  # Dictionary to hold cleaned data
            for k, v in row.items():
                if k is None:
                    continue  # Skip any None keys
                if v is None or v.strip() == "":
                    v = "N/A"  # Replace None or empty values with "N/A"
                cleaned_row[k] = v  # Add cleaned data to the dictionary

            try:
                # Add the cleaned row data to the 'lenses' collection in Firestore
                db.collection('lenses').add(cleaned_row)
                print(f"Added row {row_count}: {row.get('name', 'Unknown')} to Firestore from {csv_path}")
            except ValueError as e:
                # Handle specific errors such as missing data types
                print(f"Error adding row {row_count} to Firestore: {e} | Row: {cleaned_row}")
            except Exception as e:
                # Catch any other unexpected errors
                print(f"Unexpected error adding row {row_count} to Firestore: {e} | Row: {cleaned_row}")

    # Confirm that the upload is complete for the specified CSV file
    print(f'Upload complete for {csv_path}')

# List of CSV files to import into Firestore
csv_files = [
    'lenses/hoya.csv',  # Example CSV file path
    # Additional CSV files can be added to this list
    'lenses/essilor_1.csv',
    'lenses/Nikon.csv',
    'lenses/Rodenstock_1.csv',
    'lenses/Rodenstock_2.csv',
    'lenses/Rodenstock_3.csv',
    'lenses/seiko_1.csv',
    'lenses/seiko_2.csv',
    'lenses/zeiss_1.csv',
    'lenses/zeiss_2.csv',
    'lenses/zeiss_3.csv',
]

# Iterate over each CSV file in the list and upload its contents to Firestore
for csv_file in csv_files:
    upload_csv_to_firestore(csv_file)  # Call the upload function for each file
