from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from model1 import CNN, num_classes
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
import io
from flask_cors import CORS

# Initialize the Flask application
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the pre-trained model
model = CNN(num_classes)  # Initialize the CNN model with the number of classes
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))  # Load the model weights from a file
model.eval()

# Define image transformation to match the preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize the image to 64x64 pixels
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor to have a mean of 0.5 and standard deviation of 0.5
])

# Initialize Firebase Admin SDK
cred = credentials.Certificate("/Users/atiyamahboob/Documents/ML_Algorithm_lenses/capturelensapp-firebase-adminsdk-tl5tq-dc854ec4d5.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'your-project-id.appspot.com'  # Replace with your actual bucket name
})
bucket = storage.bucket()  # Define the bucket for Firebase Storage

# Initialize Firestore DB
db = firestore.client()  # Initialize the Firestore client

# Function to predict the label of a symbol given its image
def predict_symbol(image):
    image = transform(image).unsqueeze(0)  # Apply the transformation and add a batch dimension
    with torch.no_grad():  # Disable gradient calculation (not needed during inference)
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Function to upload the image to Firebase Storage
def upload_image_to_firebase(image_file, filename):
    blob = bucket.blob(f'engravings/{filename}')  # Upload to 'engravings' directory
    blob.upload_from_string(image_file.read(), content_type='image/png')
    blob.make_public()  # Optionally make the file publicly accessible
    return blob.public_url  # Return the public URL of the uploaded image

# Flask route for handling prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in the request'}), 400

    # Get the image file from the request
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    # Convert image to a PIL Image
    try:
        image = Image.open(image_file)
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 400

    # Generate a filename for the image (you can customize this as needed)
    filename = f'{image_file.filename.split(".")[0]}_uploaded.png'

    # Upload the image to Firebase Storage
    image_file.seek(0)  # Reset the file pointer to the start
    firebase_image_url = upload_image_to_firebase(image_file, filename)

    # Run prediction on the image
    predicted_label = predict_symbol(image)  # Predict the label of the symbol in the image

    # Fetch matching lenses from Firestore based on the predicted label
    matches = get_lens_matches(predicted_label)  # Query Firestore for lenses that match the predicted label

    return jsonify({
        'firebase_image_url': firebase_image_url,  # Return the uploaded image's Firebase URL
        'predicted_label': predicted_label,
        'matches': matches
    })

# Function to get lens matches from Firestore based on the predicted label
def get_lens_matches(label):
    # Query Firestore for lenses that match the predicted label
    lenses_ref = db.collection('lenses')  # Reference the 'lenses' collection in Firestore
    query = lenses_ref.where('label', '==', label).stream()  # Modify the query to fit your Firestore structure

    matches = []
    for doc in query:  # Iterate through each document in the query result
        lens_data = doc.to_dict()

        match = {}  # Create an empty dictionary to store the available fields
        
        # Only add fields that are present in the Firestore document
        if 'name' in lens_data:
            match['name'] = lens_data.get('name')  # Get the lens name
        if 'manufacturer' in lens_data:
            match['manufacturer'] = lens_data.get('manufacturer')  # Get the lens manufacturer
        if 'index' in lens_data:
            match['index'] = lens_data.get('index')  # Get the lens index
        if 'fitting_cross' in lens_data:
            match['fitting_cross'] = lens_data.get('fitting_cross')  # Get the lens fitting cross
        if 'engraving_ref' in lens_data:
            match['engraving_ref'] = lens_data.get('engraving_ref')  # Get the engraving reference (URL or path)
        if 'engraving_ref2' in lens_data:
            match['engraving_ref2'] = lens_data.get('engraving_ref2')
        if 'property' in lens_data:
            match['property'] = lens_data.get('property')
        if 'corridor_length_11' in lens_data:
            match['corridor_length_11'] = lens_data.get('corridor_length_11')
        if 'corridor_length_12' in lens_data:
            match['corridor_length_12'] = lens_data.get('corridor_length_12')
        if 'corridor_length_13' in lens_data:
            match['corridor_length_13'] = lens_data.get('corridor_length_13')
        if 'corridor_length_14' in lens_data:
            match['corridor_length_14'] = lens_data.get('corridor_length_14')
        if 'corridor_length_15' in lens_data:
            match['corridor_length_15'] = lens_data.get('corridor_length_15')
        if 'corridor_length_16' in lens_data:
            match['corridor_length_16'] = lens_data.get('corridor_length_16')
        if 'minimum fitting height' in lens_data:
            match['minimum fitting height'] = lens_data.get('minimum fitting height')
        if 'minimum frame depth' in lens_data:
            match['minimum frame depth'] = lens_data.get('minimum frame depth')
        if 'corridor length'  in lens_data:
            match['corridor length'] = lens_data.get('corridor length')
        if 'Recommened frame depth' in lens_data:
            match['Recommened frame depth'] = lens_data.get('Recommened frame depth')

        # Add the match to the list of matches
        if match:
            matches.append(match)

    return matches  # Return the list of matches

# Main entry point of the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
