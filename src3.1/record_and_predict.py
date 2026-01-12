import torch
import torchaudio
import torch.nn as nn
from torchvision import models
import sounddevice as sd
from scipy.io.wavfile import write
import os

# 🎯 Your class labels
class_names = [
    "belly pain",
    "burping",
    "cold_hot",
    "discomfort",
    "hungry",
    "silence",
    "tired"
]

# 🧠 Load EfficientNet model
def load_model(model_path, num_classes=7):
    model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    return model

# 🧼 Convert audio to mel spectrogram for model
def preprocess_audio(audio_path, image_size=224):
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=128)
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(mel_spec + 1e-9)

    mel_spec = torch.nn.functional.interpolate(
        mel_spec.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False
    ).squeeze(0)

    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
    mel_spec_3ch = mel_spec.repeat(3, 1, 1).unsqueeze(0)

    return mel_spec_3ch

# 🤖 Predict class from audio
def predict(model, audio_path, device='cpu'):
    model.to(device)
    input_tensor = preprocess_audio(audio_path).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)

    return class_names[predicted_idx.item()]

# 🎙️ Record audio & save to test.wav
def record_audio(filename="test1.wav", duration=6, fs=16000):
    print("🎤 Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    print(f"✅ Done! Audio saved to: {filename}")

# 🚀 Run everything
def main():
    model_path = r"C:\Users\mayur\Desktop\Infrant Problem\models\best_model_pretrain_model.pth"
    audio_path = r"C:\Users\mayur\Desktop\Infrant Problem\test\test1.wav"

    record_audio(audio_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path)

    prediction = predict(model, audio_path, device)
    print(f"\n🔊 Predicted Class: **{prediction}**\n")

if __name__ == "__main__":
    main()

