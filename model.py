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