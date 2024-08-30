from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from PIL import Image
import io
import torchvision.transforms as transforms

# Load your model
model = torch.load('best_model.pth')
model.eval()

app = Flask(__name__)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('L')
    img = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        label = predicted.item()

    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
