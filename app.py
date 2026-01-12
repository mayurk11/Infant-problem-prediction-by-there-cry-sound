
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import torch
import torchaudio
import torch.nn as nn
from torchvision import models
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = 'supersecretkey123'
app.config['UPLOAD_FOLDER'] = r"C:\Users\mayur\Desktop\Infant Problem\recordings"  # Correctly set the upload folder

# Function to load the trained model
def load_model(model_path, num_classes=7):
    model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    return model

# Function to preprocess audio files into spectrograms
def preprocess_audio(audio_path, image_size=224):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=128)
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(mel_spec + 1e-9)
    mel_spec = torch.nn.functional.interpolate(mel_spec.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0)
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
    mel_spec_3ch = mel_spec.repeat(3, 1, 1).unsqueeze(0)
    return mel_spec_3ch

# Function to predict the class of an audio file
def predict(model, audio_path, device='cpu'):
    model.to(device)
    input_tensor = preprocess_audio(audio_path).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
    class_names = ["cold_hot", "burping", "belly pain", "discomfort", "tired", "silence", "tired"]
    return class_names[predicted_idx.item()]

# Load the trained model
model = load_model(r'C:\Users\mayur\Desktop\Infant Problem\models\best_model_pretrain_model.pth')

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['user'] = request.form['username']
        return redirect(url_for('record'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/record', methods=['GET', 'POST'])
def record():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files['audio_data']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        session['audio_path'] = filepath
        return redirect(url_for('predict_page'))
    return render_template('record.html')

@app.route('/predict')
def predict_page():
    audio_path = session.get('audio_path', None)
    if audio_path:
        prediction = predict(model, audio_path)
        return render_template('predict.html', prediction=prediction)
    return redirect(url_for('record'))



@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

if __name__ == '__main__':
    app.run(debug=True)












##############################
# from flask import Flask, render_template, request, redirect, url_for, session
# from werkzeug.utils import secure_filename
# import os
# import torch
# import torchaudio
# import torch.nn as nn
# from torchvision import models
# from dotenv import load_dotenv

# load_dotenv()

# app=Flask(__name__)
# app.secret_key = 'supersecretkey123'
# app.config['UPLOAD_FOLDER'] = r'D:\infant_cry_detection\recordings'  # Correctly set the upload folder

# # Function to load the trained model
# def load_model(model_path, num_classes=7):
#     model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
#     model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
#     state_dict = torch.load(model_path, map_location='cpu')
#     model.load_state_dict(state_dict['model_state_dict'])
#     model.eval()
#     return model

# # Function to preprocess audio files into spectrograms
# def preprocess_audio(audio_path, image_size=224):
#     waveform, sample_rate = torchaudio.load(audio_path)
#     if waveform.shape[0] > 1:
#         waveform = waveform.mean(dim=0, keepdim=True)
#     mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=128)
#     mel_spec = mel_transform(waveform)
#     mel_spec = torch.log(mel_spec + 1e-9)
#     mel_spec = torch.nn.functional.interpolate(mel_spec.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0)
#     mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
#     mel_spec_3ch = mel_spec.repeat(3, 1, 1).unsqueeze(0)
#     return mel_spec_3ch

# # Function to predict the class of an audio file
# def predict(model, audio_path, device='cpu'):
#     model.to(device)
#     input_tensor = preprocess_audio(audio_path).to(device)
#     with torch.no_grad():
#         output = model(input_tensor)
#         _, predicted_idx = torch.max(output, 1)
#     class_names = ["cold_hot", "burping", "belly pain", "discomfort", "tired", "silence", "tired"]
#     return class_names[predicted_idx.item()]

# # Load the trained model
# model = load_model(r'C:\Users\mayur\Desktop\Infrant Problem\models\best_model_pretrain_model.pth')


# @app.route('/')
# def home():
#     return redirect(url_for('login'))

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         session['user'] = request.form['username']
#         return redirect(url_for('record'))
#     return render_template('login.html')

# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         return redirect(url_for('login'))
#     return render_template('signup.html')

# @app.route('/record', methods=['GET', 'POST'])
# def record():
#     if 'user' not in session:
#         return redirect(url_for('login'))
#     if request.method == 'POST':
#         file = request.files['audio_data']
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         session['audio_path'] = filepath
#         return redirect(url_for('predict_page'))
#     return render_template('record.html')

# @app.route('/predict')
# def predict_page():
#     audio_path = session.get('audio_path', None)
#     if audio_path:
#         prediction = predict(model, audio_path)
#         return render_template('predict.html', prediction=prediction)
#     return redirect(url_for('record'))

# @app.route('/chatbot')
# def chatbot():
#     return render_template('chatbot.html')

# if __name__ == '_main_':
#     app.run(debug=True)
