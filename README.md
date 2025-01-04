# Image Colorization Project

## Project Overview
This project involves building an image colorization application that uses deep learning techniques. The application accepts grayscale images as input and generates colorized images as output. The workflow consists of several steps:

1. **Data Preparation**: Preprocess images and split them into luminance (L) and chrominance (AB) channels.
2. **Model Training**: Build and train a convolutional neural network (CNN) using TensorFlow/Keras.
3. **Inference**: Use the trained model to colorize new grayscale images.
4. **Web Interface**: Provide a user-friendly web interface for uploading images and viewing results.
5. **Frontend**: Create a simple HTML/CSS/JavaScript interface for file uploads and interactions.

## Directory Structure
```plaintext
project_root/
|-- data_preparation.py
|-- model.py
|-- inference.py
|-- app.py
|-- templates/
|   |-- index.html
|-- static/
|-- uploads/
|-- colorization_model.h5
```

---

## Step 1: Data Preparation
We preprocess the CIFAR-10 dataset to extract luminance and chrominance channels for model training.

### Code
```python
# data_preparation.py
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10

def preprocess_images():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Convert images to LAB color space
    x_train_lab = cv2.cvtColor(x_train, cv2.COLOR_RGB2LAB)
    x_test_lab = cv2.cvtColor(x_test, cv2.COLOR_RGB2LAB)

    # Split into L and AB channels
    L_train = x_train_lab[:, :, :, 0] / 255.0
    AB_train = x_train_lab[:, :, :, 1:] / 128.0
    
    L_test = x_test_lab[:, :, :, 0] / 255.0
    AB_test = x_test_lab[:, :, :, 1:] / 128.0

    # Reshape for model input
    L_train = L_train.reshape(-1, 32, 32, 1)
    AB_train = AB_train.reshape(-1, 32, 32, 2)
    L_test = L_test.reshape(-1, 32, 32, 1)
    AB_test = AB_test.reshape(-1, 32, 32, 2)

    return (L_train, AB_train), (L_test, AB_test)

if __name__ == "__main__":
    preprocess_images()
```

---

## Step 2: Model Training
We build a CNN model to learn the mapping from luminance (L) to chrominance (AB) channels.

### Code
```python
# model.py
import tensorflow as tf
from tensorflow.keras import layers, models
from data_preparation import preprocess_images

def create_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(32, 32, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(2, (3, 3), activation='tanh', padding='same'))  # Output AB channels
    return model

def train_model():
    (L_train, AB_train), (L_test, AB_test) = preprocess_images()
    model = create_model()
    model.compile(optimizer='adam', loss='mse')
    model.fit(L_train, AB_train, epochs=50, batch_size=64, validation_data=(L_test, AB_test))
    model.save('colorization_model.h5')

if __name__ == "__main__":
    train_model()
```

---

## Step 3: Inference
This step applies the trained model to new grayscale images for colorization.

### Code
```python
# inference.py
import numpy as np
import cv2
from tensorflow.keras.models import load_model

def colorize_image(model, gray_image):
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (32, 32))
    L = gray_image / 255.0
    L = L.reshape(1, 32, 32, 1)

    AB = model.predict(L)
    AB = AB * 128.0  # Rescale AB channels

    lab_image = np.zeros((32, 32, 3))
    lab_image[:, :, 0] = L[0, :, :, 0] * 255.0  # L channel
    lab_image[:, :, 1:] = AB[0]  # AB channels

    # Convert LAB to RGB
    colorized_image = cv2.cvtColor(lab_image.astype('uint8'), cv2.COLOR_LAB2BGR)
    return colorized_image

if __name__ == "__main__":
    model = load_model('colorization_model.h5')
    gray_image = cv2.imread('path_to_grayscale_image.jpg')  # Load your grayscale image
    colorized_image = colorize_image(model, gray_image)
    cv2.imwrite('colorized_image.jpg', colorized_image)  # Save the colorized image
```

---

## Step 4: Web Interface
A Flask-based web application allows users to upload grayscale images for colorization.

### Code
```python
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
```

---

## Step 5: Frontend
The user interface consists of an HTML form for uploading images.

### Code
```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Colorization</title>
</head>
<body>
    <h1>Upload Grayscale Image for Colorization</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <input type="submit" value="Upload">
    </form>
</body>
</html>
```

---

## Example Outputs
### Input: Grayscale Image
![Grayscale Image](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Grayscale_8bits_palette_sample_image.png/320px-Grayscale_8bits_palette_sample_image.png)

### Output: Colorized Image
![Colorized Image](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Color_example.png/320px-Color_example.png)


---

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install tensorflow keras flask opencv-python
   ```
2. **Run the Web Application**:
   ```bash
   python app.py
   ```
3. **Access the Application**:
   Open `http://127.0.0.1:5000/` in your browser.

---

## Conclusion
This project demonstrates a complete pipeline for image colorization, from preprocessing to deployment. Users can easily interact with the model through a web interface, making it both functional and accessible.

