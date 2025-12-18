"""
Full Feature Extraction on Complete Dataset.
Processes all 99 preprocessed samples in batches and exports features.
"""
import torch
from data_loader import get_dataloader
from feature_extracter import FeatureExtractor
from config import config
import pandas as pd
import os
import numpy as np

print("\n" + "="*80)
print("FULL FEATURE EXTRACTION - ALL 99 SAMPLES")
print("="*80)

# Load data
csv_path = './data/fakeddit/train_100.csv'
if not os.path.exists(csv_path):
    print(f"ERROR: {csv_path} not found! Run preprocess.py first.")
    exit(1)

print(f"\n✓ Loading: {csv_path}")
df = pd.read_csv(csv_path)
print(f"✓ Total samples: {len(df)}")

# Initialize feature extractor
print("\nInitializing feature extractor...")
extractor = FeatureExtractor().to(config.DEVICE)
extractor.eval()

# Get data loader
dataloader = get_dataloader(csv_path, batch_size=config.BATCH_SIZE, shuffle=False)
print(f"✓ Data loader ready (batch_size={config.BATCH_SIZE})")
print(f"✓ Total batches: {len(dataloader)}")

# Extract features for all batches
all_features = {
    'text_pattern': [],
    'text_semantic': [],
    'image_pattern': [],
    'image_semantic': []
}
all_texts = []
all_ids = []
all_labels = []

print("\n" + "="*80)
print("EXTRACTING FEATURES FROM ALL BATCHES")
print("="*80)

batch_num = 0
total_samples = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
        batch_num += 1
        texts = batch['texts']
        images = batch['images']
        ids = batch['ids']
        labels = batch['labels']
        
        # Extract features
        features = extractor(texts, images)
        
        # Store features
        for key in all_features:
            all_features[key].append(features[key].cpu().numpy())
        
        all_texts.extend(texts)
        all_ids.extend(ids)
        all_labels.append(labels.cpu().numpy())
        
        total_samples += len(texts)
        print(f"  Batch {batch_num}: {len(texts):2d} samples | Total: {total_samples:3d}")

print(f"\n✓ Total samples processed: {total_samples}")

# Concatenate all batches
print("\nConcatenating features...")
for key in all_features:
    all_features[key] = np.vstack(all_features[key])

all_labels = np.hstack(all_labels).reshape(-1)

print(f"✓ Final shapes:")
for key, arr in all_features.items():
    print(f"  {key}: {arr.shape}")

# Create outputs directory
os.makedirs('./outputs/full_dataset', exist_ok=True)

print("\n" + "="*80)
print("EXPORTING FULL DATASET FEATURES")
print("="*80)

# Export individual feature streams
for stream_name, features_arr in all_features.items():
    df_features = pd.DataFrame(
        features_arr,
        columns=[f'dim_{i}' for i in range(512)]
    )
    df_features.insert(0, 'sample_id', range(len(df_features)))
    df_features.insert(1, 'id', all_ids)
    df_features.insert(2, 'label', all_labels)
    df_features.insert(3, 'text', all_texts)
    
    output_file = f'./outputs/full_dataset/features_{stream_name}_full.csv'
    df_features.to_csv(output_file, index=False)
    print(f"✓ {output_file}")
    print(f"  Shape: {df_features.shape}")

# Create summary file
print("\nCreating summary file...")
summary_df = pd.DataFrame({
    'sample_id': range(len(all_texts)),
    'id': all_ids,
    'label': all_labels,
    'text': all_texts,
})

for stream_name in ['text_pattern', 'text_semantic', 'image_pattern', 'image_semantic']:
    summary_df[f'{stream_name}_mean'] = all_features[stream_name].mean(axis=1)
    summary_df[f'{stream_name}_std'] = all_features[stream_name].std(axis=1)

output_file = './outputs/full_dataset/features_summary_full.csv'
summary_df.to_csv(output_file, index=False)
print(f"✓ {output_file}")
print(f"  Shape: {summary_df.shape}")

# Save as PyTorch tensor
print("\nSaving PyTorch tensor...")
features_dict = {
    'text_pattern': torch.tensor(all_features['text_pattern'], dtype=torch.float32),
    'text_semantic': torch.tensor(all_features['text_semantic'], dtype=torch.float32),
    'image_pattern': torch.tensor(all_features['image_pattern'], dtype=torch.float32),
    'image_semantic': torch.tensor(all_features['image_semantic'], dtype=torch.float32),
    'texts': all_texts,
    'ids': all_ids,
    'labels': torch.tensor(all_labels, dtype=torch.long)
}

output_file = './outputs/full_dataset/features_full.pt'
torch.save(features_dict, output_file)
print(f"✓ {output_file}")
print(f"  Total features: {len(all_texts)} samples × 2048-d")

print("\n" + "="*80)
print("✅ FULL DATASET FEATURES EXTRACTED & EXPORTED")
print("="*80)
print(f"\nSummary:")
print(f"  Total samples: {total_samples}")
print(f"  Features per sample: 2048-d (4 × 512)")
print(f"  Output location: ./outputs/full_dataset/")
print(f"\nFiles created:")
print(f"  1. features_text_pattern_full.csv")
print(f"  2. features_text_semantic_full.csv")
print(f"  3. features_image_pattern_full.csv")
print(f"  4. features_image_semantic_full.csv")
print(f"  5. features_summary_full.csv (quick view)")
print(f"  6. features_full.pt (PyTorch tensor)")
print("="*80 + "\n")
