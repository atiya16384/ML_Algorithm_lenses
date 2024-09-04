from flask import Flask, request, jsonify  # Flask for web app, request for handling requests, jsonify for creating JSON responses
import torch  
from torchvision import transforms  
from PIL import Image  # PIL for image handling
from model import CNN, num_classes  # Importing the CNN model and num_classes from model.py
import firebase_admin  
from firebase_admin import credentials, firestore 

# Initialize the Flask application
app = Flask(__name__)

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
cred = credentials.Certificate("/Users/atiya/Documents/ML_Algorithm_lenses/capturelensapp-firebase-adminsdk-tl5tq-dc854ec4d5.json")
firebase_admin.initialize_app(cred)  
db = firestore.client() 

# Function to predict the label of a symbol given its image
def predict_symbol(image_path):
    image = Image.open(image_path)  # Open the image from the provided file path
    image = transform(image).unsqueeze(0)  # Apply the transformation and add a batch dimension
    with torch.no_grad():  # Disable gradient calculation (not needed during inference)
        output = model(image) 
        _, predicted = torch.max(output, 1) 
    return predicted.item()  

# Flask route for handling prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the JSON data from the request
    image_path = data['imageUri']  # Extract the image file path from the JSON data
    predicted_label = predict_symbol(image_path)  # Predict the label of the symbol in the image

    # Fetch matching lenses from Firestore based on the predicted label
    matches = get_lens_matches(predicted_label)  # Query Firestore for lenses that match the predicted label
    return jsonify({'matches': matches})  # Return the matches as a JSON response

# Function to get lens matches from Firestore based on the predicted label
def get_lens_matches(label):
    # Query Firestore for lenses that match the predicted label
    lenses_ref = db.collection('lenses')  # Reference the 'lenses' collection in Firestore
    query = lenses_ref.where('label', '==', label).stream() 

    matches = [] 
    for doc in query:  # Iterate through each document in the query result
        lens_data = doc.to_dict()  
        matches.append({
            'name': lens_data.get('name'),  # Get the lens name
            'description': lens_data.get('description'),  # Get the lens description
            'url': lens_data.get('image_url')  # Get the lens image URL
        })

    return matches  # Return the list of matches

# Main entry point of the Flask application
if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode