from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import tensorflow
from io import BytesIO
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("Flask-Cors")

app = Flask(__name__)
CORS(app)

# Load the model
model = tensorflow.keras.models.load_model("model_keras.h5")
classes = ["R", "U", "I", "N", "G", "Z", "T", "S", "A", "F", "O", "H", " ", "M", "J", "C", "D", "V", "Q", "X", "E", "B", "K", "L", "Y", "P", "W"]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')


@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    image_data = request.files['image'].read()
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
    img = img.resize((300, 300 * img.size[1] // img.size[0]), Image.ANTIALIAS)
    inp_numpy = np.array(img)[None]

    class_scores = model.predict(inp_numpy)[0]
    predicted_class_index = np.argmax(class_scores)
    predicted_class = classes[predicted_class_index]

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
