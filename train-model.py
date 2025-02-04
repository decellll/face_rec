import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from model import create_emotion_model

def load_jaffe_dataset(path):
    emotions = ['AN', 'DI', 'FE', 'HA', 'SA', 'SU', 'NE']
    faces = []
    labels = []
    
    for emotion_code in emotions:
        for filename in os.listdir(path):
            if filename.startswith(emotion_code):
                img_path = os.path.join(path, filename)
                
                # Baca gambar
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Resize ke ukuran standar
                img = cv2.resize(img, (48, 48))
                
                faces.append(img)
                labels.append(emotions.index(emotion_code))
    
    # Konversi ke numpy array
    faces = np.array(faces)
    faces = np.expand_dims(faces, -1)
    labels = to_categorical(labels)

    return faces, labels

def train_emotion_model(dataset_path='dataset/JAFFE'):
    # Load dataset
    faces, emotions = load_jaffe_dataset(dataset_path)
    
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
        batch_size=32,
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
