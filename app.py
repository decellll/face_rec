from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime

app = Flask(__name__)

# Pastikan folder untuk menyimpan gambar ada
if not os.path.exists('static/captures'):
    os.makedirs('static/captures')

# Load model CNN yang sudah dilatih
emotion_model = load_model('static/models/emotion_model.h5')
emotions = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']

def get_emotion_recommendation(emotion):
    recommendations = {
        'Marah': 'Cobalah teknik pernapasan dalam dan meditasi untuk menenangkan diri.',
        'Jijik': 'Alihkan perhatian Anda ke hal-hal yang lebih menyenangkan.',
        'Takut': 'Identifikasi sumber ketakutan dan coba diskusikan dengan orang yang dipercaya.',
        'Senang': 'Bagikan kebahagiaan Anda dengan orang lain!',
        'Sedih': 'Lakukan aktivitas yang Anda sukai atau berbicara dengan teman dekat.',
        'Terkejut': 'Ambil waktu sejenak untuk menenangkan diri.',
        'Netral': 'Ini waktu yang baik untuk refleksi diri.'
    }
    return recommendations.get(emotion, 'Tidak ada rekomendasi spesifik.')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    camera = VideoCamera()
    frame, emotions_dict = camera.capture_frame()
    
    # Simpan gambar
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_name = f'capture_{timestamp}.jpg'
    cv2.imwrite(f'static/captures/{image_name}', frame)
    
    # Ambil emosi dominan
    dominant_emotion = max(emotions_dict.items(), key=lambda x: x[1])[0]
    confidence = max(emotions_dict.values())
    
    # Dapatkan rekomendasi
    recommendation = get_emotion_recommendation(dominant_emotion)
    
    return render_template('result.html',
                         image_name=image_name,
                         dominant_emotion=dominant_emotion,
                         confidence=round(confidence * 100, 2),
                         emotions=emotions_dict,
                         recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
