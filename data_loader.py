"""
Data Loader for Fakeddit Dataset.
Handles loading of text and images from CSV, downloading images from URLs,
and creating batches for model training/inference.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
import os
from config import config

class FakedditDataset(Dataset):
    def __init__(self, csv_path, image_dir=None, max_samples=None):
        """
        Args:
            csv_path: Path to Fakeddit CSV (train_100.csv)
            image_dir: Directory where images are downloaded (optional)
            max_samples: Limit samples (default: use all)
        """
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.head(max_samples)
        self.image_dir = image_dir
        self.has_image = 'hasImage' in self.df.columns
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text: Use 'text' (new format), fallback to 'clean_title' or 'title'
        if 'text' in row and pd.notna(row['text']):
            text = row['text']
        elif 'clean_title' in row and pd.notna(row['clean_title']):
            text = row['clean_title']
        else:
            text = row['title']
        
        # Label: Using 'label' column (if present) else fallback to 2_way_label
        if 'label' in row and not pd.isna(row['label']):
            label = int(row['label'])
        elif '2_way_label' in row and not pd.isna(row['2_way_label']):
            label = int(row['2_way_label'])
        else:
            label = -1
        
        # Image
        image = None
        if self.has_image and row['hasImage'] == True:
            try:
                if self.image_dir and os.path.exists(f"{self.image_dir}/{row['id']}.jpg"):
                    # Load from local
                    image = Image.open(f"{self.image_dir}/{row['id']}.jpg").convert('RGB')
                elif 'image_url' in row and pd.notna(row['image_url']):
                    # Download (careful: might be slow)
                    response = requests.get(row['image_url'], timeout=5)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
            except:
                image = Image.new('RGB', (224, 224), color='gray')  # Gray placeholder
        else:
            # Create placeholder for samples without images
            image = Image.new('RGB', (224, 224), color='gray')
        
        return {
            'text': str(text),
            'image': image,
            'label': label,
            'id': row['id']
        }

def collate_fn(batch):
    """Custom collate to handle images and text"""
    texts = [item['text'] for item in batch]
    images = [item['image'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    ids = [item['id'] for item in batch]
    
    return {
        'texts': texts,
        'images': images,
        'labels': labels,
        'ids': ids
    }

def get_dataloader(csv_path, batch_size=8, shuffle=True):
    dataset = FakedditDataset(csv_path)  # Uses all samples in CSV (99 pruned)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)