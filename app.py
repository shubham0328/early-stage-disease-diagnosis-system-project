import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your pre-trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vgg-16-nail-disease.h5")  # Update with your model path
model = load_model(MODEL_PATH)

# Define the 19 disease classes based on your subfolder names
disease_classes = [
    'Darier_s disease',
    'Muehrck-e_s lines',
    'alopecia areata',  # Corrected spelling from "aloperia areata"
    'beau_s lines',
    'bluish nail',
    'clubbing',
    'eczema',
    'half and half nail(Lindsay_s nails)',
    'koilonychia',
    'leukonychia',
    'onycholysis',
    'pale nail',
    'red lunula',
    'splinter hemmorrage',
    'terry_s nail',
    'white nail',
    'yellow nails',
    'healthy',  # Placeholder: replace with your 18th folder name
    'inflected'   # Placeholder: replace with your 19th folder name
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(saved_path)
    
    # Load and preprocess the image
    img = image.load_img(saved_path, target_size=(224, 224))  # Adjust size to match your model
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize if your model requires it
    
    # Make prediction
    preds = model.predict(x)
    predicted_class_index = np.argmax(preds, axis=1)[0]
    
    # Map the predicted index to the disease name
    if predicted_class_index < len(disease_classes):
        predicted_disease = disease_classes[predicted_class_index]
    else:
        return jsonify({'error': 'Prediction out of range'}), 500
    
    # Return the prediction as JSON
    return jsonify({'prediction': predicted_disease})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)