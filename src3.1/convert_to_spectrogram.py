import os
import librosa
import numpy as np

# Paths
AUGMENTED_DIR = r"C:\Users\mayur\Desktop\Infrant Problem\newVtoA"
SPECTRO_DIR = r"C:\Users\mayur\Desktop\Infrant Problem\AUGMENTtoSPECTRO"

# Ensure output directory exists
os.makedirs(SPECTRO_DIR, exist_ok=True)

# Parameters
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512

# Process each class
for label in os.listdir(AUGMENTED_DIR):
    class_path = os.path.join(AUGMENTED_DIR, label)
    spectro_class_path = os.path.join(SPECTRO_DIR, label)
    os.makedirs(spectro_class_path, exist_ok=True)

    for file in os.listdir(class_path):
        if not file.endswith('.wav'):
            continue
        file_path = os.path.join(class_path, file)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Generate mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize to 0-1 range
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

        # Save as .npy
        npy_name = file.replace(".wav", ".npy")
        np.save(os.path.join(spectro_class_path, npy_name), mel_norm)

    print(f"✅ Converted class '{label}' to spectrograms")
