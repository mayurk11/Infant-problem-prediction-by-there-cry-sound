import os
import shutil
import random
from tqdm import tqdm

def split_dataset(input_dir, output_dir, val_ratio=0.2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split in ['train', 'val']:
        split_path = os.path.join(output_dir, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)

    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for cls in tqdm(classes, desc="Splitting classes"):
        cls_path = os.path.join(input_dir, cls)
        images = os.listdir(cls_path)
        random.shuffle(images)

        val_count = int(len(images) * val_ratio)
        train_images = images[val_count:]
        val_images = images[:val_count]

        for split, split_images in zip(['train', 'val'], [train_images, val_images]):
            split_cls_path = os.path.join(output_dir, split, cls)
            os.makedirs(split_cls_path, exist_ok=True)

            for img in split_images:
                src = os.path.join(cls_path, img)
                dst = os.path.join(split_cls_path, img)
                shutil.copy(src, dst)

if __name__ == "__main__":
    input_dir = r"C:\Users\mayur\Desktop\Infrant Problem\spectrogram_png"
    output_dir = r"C:\Users\mayur\Desktop\Infrant Problem\data2"
    split_dataset(input_dir, output_dir)
