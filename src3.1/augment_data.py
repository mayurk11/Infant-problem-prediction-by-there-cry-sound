import os
import shutil
import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import random

# Define source and destination directories
SOURCE_DIR = r"C:\Users\mayur\Desktop\Infrant Problem\Baby Dataset"
DEST_DIR = r"C:\Users\mayur\Desktop\Infrant Problem\newVtoA"

# Clear the destination folder if exists
if os.path.exists(DEST_DIR):
    shutil.rmtree(DEST_DIR)
os.makedirs(DEST_DIR, exist_ok=True)

# Augmentation pipeline
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.2, max_shift=0.2, p=0.5)
])

# Target class balancing
TARGET_COUNT = 400  # Max samples per class

# Perform augmentation
for label in os.listdir(SOURCE_DIR):
    class_dir = os.path.join(SOURCE_DIR, label)
    dest_class_dir = os.path.join(DEST_DIR, label)
    os.makedirs(dest_class_dir, exist_ok=True)

    files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
    original_count = len(files)

    print(f"Processing class: {label} | Original: {original_count}")

    for i, file in enumerate(files):
        src_file = os.path.join(class_dir, file)
        y, sr = librosa.load(src_file, sr=None)

        # Save original to augmented dir
        sf.write(os.path.join(dest_class_dir, f"original_{i}.wav"), y, sr)

    # Create augmented samples until target reached
    aug_count = 0
    while original_count + aug_count < TARGET_COUNT:
        file = random.choice(files)
        src_file = os.path.join(class_dir, file)
        y, sr = librosa.load(src_file, sr=None)

        y_aug = augment(samples=y, sample_rate=sr)
        save_path = os.path.join(dest_class_dir, f"aug_{aug_count}.wav")
        sf.write(save_path, y_aug, sr)
        aug_count += 1

    print(f"✅ Done with class: {label} | Final count: {original_count + aug_count}")
