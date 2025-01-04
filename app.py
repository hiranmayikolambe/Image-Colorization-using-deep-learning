# app.py
from flask import Flask, request, render_template, send_file
import os
from inference import colorize_image
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)
model = load_model('colorization_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    gray_image = cv2.imread(file_path)
    colorized_image = colorize_image(model, gray_image)
    output_path = os.path.join('static', 'colorized_' + file.filename)
    cv2.imwrite(output_path, colorized_image)

    return send_file(output_path)

if __name__ == "__main__":
    app.run(debug=True)