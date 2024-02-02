import argparse
import pandas as pd
import numpy as np
import re
import os
import shutil
import yaml
from tqdm import tqdm

np.random.seed(42)

# Example Usage
# python src/yolov8_datagen.py --source_dir datasets/archive --dest_dir datasets/custom-dataset --total_images 5000 --split 0.9 0.05 0.05

def get_input_args():
    """
    Get command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="YOLOv8 Dataset Generator")
    parser.add_argument("--source_dir", type=str, default='datasets/archive', help="Source directory path")
    parser.add_argument("--dest_dir", type=str, default='datasets/TD_YOLOv8', help="Destination directory path")
    parser.add_argument("--total_images", type=int, default=5000, help="Total number of images")
    parser.add_argument("--split", nargs='+', type=float, default=[0.9, 0.05, 0.05], help="Dataset split ratios (train, val, test)")
    return parser.parse_args()

class YOLOv8DatasetGenerator:
    def __init__(self, source_dir, dest_dir, total_images, split):
        """
        Initialize YOLOv8DatasetGenerator.

        Args:
            source_dir (str): Source directory path.
            dest_dir (str): Destination directory path.
            total_images (int): Total number of images.
            density (int): Density factor.
            split (list of float): Dataset split ratios (train, val, test).
        """
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.total_images = total_images
        self.split = split
        self.annots = None

    def make_dataset(self, row):
        """
        Create YOLOv8 dataset from a row of image data.

        Args:
            row (pd.Series): Row from the image dataframe.

        Returns:
            tuple: Tuple containing label file path and image file path.
        """
        img_id, img_w, img_h, img_set = row['id'], row['width'], row['height'], row['set']

        boxes = self.annots.query('image_id == @img_id')['bbox']
        boxes = boxes.apply(lambda x: [float(x[0]) / img_w + 0.5 * (float(x[2]) / img_w),
                                       float(x[1]) / img_h + 0.5 * (float(x[3]) / img_h),
                                       float(x[2]) / img_w, float(x[3]) / img_h])

        labels_dir = os.path.join(self.dest_dir, img_set, 'labels')
        images_dir = os.path.join(self.dest_dir, img_set, 'images')
        source_images_dir = os.path.join(self.source_dir, 'train_val_images', 'train_images')
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        label_path = os.path.join(labels_dir, img_id + '.txt')
        image_path = os.path.join(images_dir, img_id + '.jpg')
        source_image_path = os.path.join(source_images_dir, img_id + '.jpg')

        with open(label_path, 'w') as f:
            for box in boxes:
                if all(x >= 0 for x in box):
                    box_str = '0 ' + ' '.join(map(str, box))
                    f.write(box_str + '\n')

        shutil.copy(source_image_path, image_path)

        return label_path, image_path

    def create_dataset_config(self, dest_dir):
        """
        Create the YOLOv8 dataset configuration file.

        Args:
            dest_dir (str): Destination directory path.

        Returns:
            dict: YAML configuration dictionary.
        """
        config = {
            'train': 'train/images',
            'val': 'val/images',
            'nc': 1,
            'names': ['text'],
        }
        filename = os.path.join(dest_dir, 'data.yaml')

        with open(filename, 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

        return config

    def generate(self):
        """
        Filter source dataset and configure dataset splits. Generate YOLOv8 dataset and .yaml configuration.
        """
        imgs = pd.read_parquet(os.path.join(self.source_dir, 'img.parquet'))
        annots = pd.read_parquet(os.path.join(self.source_dir, 'annot.parquet'))

        annots['utf8_string'] = annots['utf8_string'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
        annots = annots[annots['utf8_string'].apply(lambda x: len(x) > 1)].reset_index(drop=True)

        selected_ids = np.random.choice(annots['image_id'].unique(), size=self.total_images, replace=False)

        annots = annots[annots['image_id'].isin(selected_ids)].reset_index(drop=True)
        imgs = imgs[imgs['id'].isin(selected_ids)].reset_index(drop=True)

        split_indices = np.cumsum(np.floor(np.array(self.split) * self.total_images)).astype(int)
        train_split, val_split, test_split, _ = np.split(selected_ids, split_indices)

        imgs.loc[imgs['id'].isin(train_split), 'set'] = 'train'
        imgs.loc[imgs['id'].isin(val_split), 'set'] = 'val'
        imgs.loc[imgs['id'].isin(test_split), 'set'] = 'test'

        annots['bbox'] = annots['bbox'].apply(lambda x: str(x)[1:-1].split())

        self.annots = annots

        print(f"Creating YOLOv8 dataset at {self.dest_dir}")
        for _, row in tqdm(imgs.iterrows(), total=len(imgs), desc="Processing images"):
            _ = self.make_dataset(row)

        print(f"Creating YOLOv8 dataset configuration at {self.dest_dir}")
        config = self.create_dataset_config(self.dest_dir)
        
def main():
    """
        Main function to generate YOLOv8 dataset using user input.
    """
    args = get_input_args()
    dataset_creator = YOLOv8DatasetGenerator(args.source_dir, args.dest_dir, args.total_images, args.split)
    dataset_creator.generate()
    

if __name__ == "__main__":
    main()
