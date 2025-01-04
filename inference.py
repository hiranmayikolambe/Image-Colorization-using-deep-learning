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

    # Convert back to LAB lab_image = np.zeros((32, 32, 3))
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