import torch
import torchaudio
import torch.nn as nn
from torchvision import models

# List of class names in the order used during training
class_names = [
    "belly pain",
    "burping",
    "cold_hot",
    "discomfort",
    "hungry",
    "silence",
    "tired"
]

# Load your trained EfficientNet model
def load_model(model_path, num_classes=7):
    model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    return model

# Preprocess audio into a 3-channel Mel spectrogram (224x224)
def preprocess_audio(audio_path, image_size=224):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono (average if stereo)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Create Mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=128)
    mel_spec = mel_transform(waveform)

    # Apply log scaling
    mel_spec = torch.log(mel_spec + 1e-9)

    # Resize to (224, 224)
    mel_spec = torch.nn.functional.interpolate(mel_spec.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0)

    # Normalize between 0 and 1
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())

    # Convert to 3 channels by repeating
    mel_spec_3ch = mel_spec.repeat(3, 1, 1).unsqueeze(0)  # Shape: (1, 3, 224, 224)

    return mel_spec_3ch

# Predict the class of an audio file
def predict(model, audio_path, device='cpu'):
    model.to(device)
    input_tensor = preprocess_audio(audio_path).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)

    return class_names[predicted_idx.item()]

# Main runner
def main():
    audio_file = r"C:\Users\mayur\Desktop\Infrant Problem\test\sample3.wav"     # 🔁 Replace this
    model_path = r"C:\Users\mayur\Desktop\Infrant Problem\models\best_model_pretrain_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path)

    predicted = predict(model, audio_file, device=device)
    print(f"\n🔊 Predicted Class: **{predicted}**\n")

if __name__ == "__main__":
    main()
