from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from moviepy import VideoFileClip
import os
import cv2
import torch
import torchaudio
from torchvision.io import read_video
from transformers import ViTFeatureExtractor, ViTModel
from transformers import Wav2Vec2Model
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import sqlite3
from datetime import datetime

# Model Loading and Setup
class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.audio_net = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size + self.audio_net.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes))

    def forward(self, images, audio):
        batch_size, num_frames, channels, height, width = images.shape
        images = images.view(batch_size * num_frames, channels, height, width)
        batch_size, *_ = audio.shape
        audio = audio.view(batch_size, -1)
        x1 = self.vit(images).pooler_output
        x2 = self.audio_net(audio).last_hidden_state.mean(dim=1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x1 = x1.view(batch_size, num_frames, -1).mean(dim=1)
        x = torch.cat((x1, x2), dim=1)
        return self.classifier(x)

model = FusionModel(num_classes=2)
model.load_state_dict(torch.load(r"C:\Users\WELCOME\Desktop\pro - Copy\late_fusion_transformer_model1.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)

app = Flask(__name__)
app.secret_key = '3d6f45a5fc12445dbac2f59c3b6c7cb1'

DATABASE = 'DeepFake.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    if not os.path.exists(DATABASE):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
init_db()

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['username'] = username
            session['login_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user:
            conn.close()
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        hashed_password = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                       (username, email, hashed_password))
        conn.commit()
        conn.close()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('reg.html')

@app.route('/home')
def home():
    if 'logged_in' in session:
        return render_template('home.html')
    else:
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    session.pop('login_time', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def process_input():
    if 'logged_in' in session:
        if request.method == 'POST':
            file = request.files.get('videoUpload')
            if file:
                filename = secure_filename(file.filename)
                file.save(os.path.join("upload_video", filename))
                video = VideoFileClip(os.path.join("upload_video", filename))
                audio = video.audio
                audio.write_audiofile(os.path.join("upload_video", "video.wav"), codec='pcm_s16le')
                video, audio, _ = read_video(os.path.join("upload_video", filename), pts_unit='sec', output_format='TCHW')
                waveform, sample_rate = torchaudio.load(os.path.join("upload_video", "video.wav"), format="wav")
                video = video / 224.0
                if audio.shape[0] == 2:
                    audio = torch.mean(audio, dim=0, keepdim=True)
                audio = torchaudio.transforms.Vad(sample_rate)(waveform)
                if audio.shape[1] < sample_rate:
                    audio = torch.nn.functional.pad(audio, (0, sample_rate - audio.shape[1]))
                elif audio.shape[1] > sample_rate:
                    audio = audio[:, :sample_rate]
                audio = audio.reshape(1, -1)
                video = torch.nn.functional.interpolate(video.squeeze(0), size=(224, 224), mode='bilinear')
                video = video.unsqueeze(0)
                audio = audio.unsqueeze(0)
                video = video.to(device)
                audio = audio.to(device)
                with torch.no_grad():
                    outputs = model(video, audio)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probs, dim=1).item()
                    classes = ['real', 'fake']
                    predicted_class_name = classes[predicted_class]
                    print("probs", probs)
                print(f"Predicted class: {predicted_class_name}")
                result = "The Video is " + predicted_class_name
                torch.cuda.empty_cache()
                return render_template('home.html', result=result)
            else:
                return render_template('home.html', result="No file was submitted")
        else:
            return redirect(url_for('home'))
    else:
        return redirect(url_for('login'))

@app.route('/test', methods=['GET'])
def test_route():
    if 'logged_in' in session:
        return render_template('test.html')
    else:
        return redirect(url_for('login'))

if __name__ == '__main__':
    if not os.path.exists("upload_video"):
        os.makedirs("upload_video")
    app.run(debug=True)