"""
Data Preprocessing Pipeline.
Prunes dataset to 100 samples, removes nulls, cleans text, validates URLs,
and outputs a clean CSV ready for feature extraction.
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

def prune_and_preprocess():
    """
    1. Load Fakeddit train.csv
    2. Keep only 100 samples
    3. Select essential columns
    4. Clean and validate data
    5. Save to train_100.csv
    """
    
    print("=" * 60)
    print("STEP 1: PRUNING DATASET TO 100 SAMPLES")
    print("=" * 60)
    
    # Load full dataset
    csv_path = "./data/fakeddit/train.csv"
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Original shape: {df.shape}")
    
    # Keep first 100 samples
    df = df.head(100)
    print(f"After pruning to 100: {df.shape}")
    
    # Select only essential columns
    print("\nSelecting essential columns...")
    essential_cols = ['id', 'clean_title', 'image_url', '2_way_label']
    df = df[essential_cols].copy()
    print(f"Columns: {df.columns.tolist()}")
    
    # Rename for clarity
    df.columns = ['id', 'text', 'image_url', 'label']
    
    print("\n" + "=" * 60)
    print("STEP 2: DATA VALIDATION & CLEANING")
    print("=" * 60)
    
    # Remove rows with missing critical values
    print(f"Initial samples: {len(df)}")
    df = df.dropna(subset=['text', 'image_url', 'label'])
    print(f"After removing nulls: {len(df)}")
    
    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=['text', 'label'])
    print(f"After removing duplicates: {len(df)} (removed {initial_len - len(df)})")
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Text preprocessing
    print("\nCleaning text...")
    df['text'] = df['text'].str.strip()  # Remove leading/trailing whitespace
    df['text'] = df['text'].str.replace('\n', ' ')  # Remove newlines
    df['text'] = df['text'].str.replace('\r', ' ')  # Remove carriage returns
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
    
    # Validate URLs
    print("Validating image URLs...")
    df['has_valid_url'] = df['image_url'].str.startswith('http')
    invalid_urls = (~df['has_valid_url']).sum()
    if invalid_urls > 0:
        print(f"Warning: {invalid_urls} invalid URLs found (will use gray placeholders)")
    df = df.drop('has_valid_url', axis=1)
    
    # Label distribution
    print("\nLabel distribution:")
    print(f"  Real (0):  {(df['label'] == 0).sum()} samples")
    print(f"  Fake (1):  {(df['label'] == 1).sum()} samples")
    
    # Save pruned dataset
    print("\n" + "=" * 60)
    print("STEP 3: SAVING PRUNED DATASET")
    print("=" * 60)
    
    output_path = "./data/fakeddit/train_100.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved to: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Display sample
    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)
    print(df.head(3).to_string())
    
    return df

if __name__ == "__main__":
    df = prune_and_preprocess()
    print("\n✅ Preprocessing complete! Ready for feature extraction.")
