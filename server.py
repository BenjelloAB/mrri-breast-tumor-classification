from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

# === Dummy Cast layer (for deserialization only) ===
class Cast(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# ==

model_path = "./final_model_resnet50.h5"
model = load_model(model_path, custom_objects={"Cast": Cast}, compile=False)

app = Flask(__name__)
CORS(app)  # Allowing  CORS for frontend requests . 



@app.route('/')
def home():
    return render_template('mri_classification_front.html')  

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucun fichier image reçu'}), 400

    file = request.files['image']

    # Loading and preprocess img
    image = Image.open(file.stream).convert('RGB')  
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ResNet50 preprocessing

    # Making prediction
    prediction = model.predict(img_array)[0][0]  

    # Determinin label and confidence
    threshold = 0.4
    if prediction > threshold:
        label = "Malignant"
        confidence = prediction
    else:
        label = "Benign"
        confidence = 1 - prediction

    return jsonify({
        'prediction': label,
        'confidence': f"{confidence:.2%}"
    })

if __name__ == '__main__':
    app.run(debug=True)