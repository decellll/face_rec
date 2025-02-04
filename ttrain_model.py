import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from model import create_emotion_model

def load_fer2013():
    # Load dataset FER2013 (asumsikan dataset dalam format CSV)
    data = pd.read_csv('fer2013.csv')
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).values

    return faces, emotions

def train_emotion_model():
    # Load dataset
    faces, emotions = load_fer2013()
    
    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42)
    
    # Normalisasi data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Buat model
    model = create_emotion_model()
    
    # Training model
    model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=50,
        validation_data=(x_test, y_test),
        shuffle=True
    )
    
    # Simpan model
    model.save('static/models/emotion_model.h5')
    
    # Evaluasi model
    scores = model.evaluate(x_test, y_test)
    print(f"Accuracy: {scores[1]*100:.2f}%")

if __name__ == "__main__":
    train_emotion_model()
