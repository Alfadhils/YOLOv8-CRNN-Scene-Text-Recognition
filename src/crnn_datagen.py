import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import re
from PIL import Image
from tqdm import tqdm

def get_input_args():
    """
    Get command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="YOLOv8 Dataset Generator")
    parser.add_argument("--source_dir", type=str, default='../datasets/archive', help="Source directory path")
    parser.add_argument("--dest_dir", type=str, default='../datasets/cropped_text', help="Destination directory path")
    parser.add_argument("--total_images", type=int, default=10000, help="Total number of images")
    return parser.parse_args()

class TRDatasetGenerator:
    def __init__(self, source_dir, dest_dir, total_images):
        """
        Initialize TRDatasetGenerator.

        Args:
            source_dir (str): Source directory path.
            dest_dir (str): Destination directory path.
            total_images (int): Total number of images.
        """
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.total_images = total_images
        
    def make_dataset(self, row):
        """
        Create Text Recognition dataset from a row of image data.

        Args:
            row (pd.Series): Row from the image dataframe.

        Returns:
            dest_path: The image file destination path.
        """
        annot_id = row['id']
        img_id = row['image_id']
        x_min, y_min, bbox_w, bbox_h = row['bbox']
        x_max, y_max = x_min + bbox_w, y_min + bbox_h
        
        img_path = os.path.join(self.source_dir, 'train_val_images', 'train_images', img_id + '.jpg')
        img = Image.open(img_path)
        
        img_crop = img.crop((x_min, y_min, x_max, y_max))
        
        imgs_dir = os.path.join(self.dest_dir,'images')
        os.makedirs(imgs_dir, exist_ok=True)
        dest_path = os.path.join(imgs_dir, annot_id + '.jpg')
        
        img_crop.save(dest_path)
        
        return dest_path

    def generate(self):
        """
        Generate Text Recognition dataset.
        """
        annots = pd.read_parquet(os.path.join(self.source_dir, 'annot.parquet'))
        annots['utf8_string'] = annots['utf8_string'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
        annots = annots[annots['utf8_string'].apply(lambda x: len(x) > 1)].reset_index(drop=True)
        
        annots = annots.sample(self.total_images,random_state=42)
        annots['bbox'] = annots['bbox'].apply(lambda x: str(x)[1:-1].split())
        annots['bbox'] = annots['bbox'].apply(lambda x: [int(float(i)) for i in x])
        
        for _, row in tqdm(annots.iterrows(), total=self.total_images, desc="Processing images"):
            _ = self.make_dataset(row)
            
        annots[['id','utf8_string']].to_csv(os.path.join(self.dest_dir,'labels.csv'),index=False)

def main():
    """
        Main function to generate dataset using user input.
    """
    args = get_input_args()
    dataset_creator = TRDatasetGenerator(args.source_dir, args.dest_dir, args.total_images)
    dataset_creator.generate()
    

if __name__ == "__main__":
    main()