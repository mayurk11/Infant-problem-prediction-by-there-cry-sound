import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

INPUT_DIR = r"C:\Users\mayur\Desktop\Infrant Problem\AUGMENTtoSPECTRO"
OUTPUT_DIR = r"C:\Users\mayur\Desktop\Infrant Problem\spectrogram_png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in os.listdir(INPUT_DIR):
    class_input_dir = os.path.join(INPUT_DIR, label)
    class_output_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(class_output_dir, exist_ok=True)

    for fname in tqdm(os.listdir(class_input_dir), desc=f"Converting {label}"):
        if not fname.endswith(".npy"):
            continue
        arr = np.load(os.path.join(class_input_dir, fname))

        # Save as PNG
        plt.imsave(os.path.join(class_output_dir, fname.replace(".npy", ".png")), arr, cmap='viridis')
