"""
Main Feature Extraction Pipeline.
Loads data, initializes feature extractors, extracts 4 feature streams,
and saves features to PyTorch tensor for downstream fusion/classification.
"""
import torch
from data_loader import get_dataloader
from feature_extracter import FeatureExtractor
from config import config
import os

def main():
    print("Starting Fake News Feature Extraction Pipeline")
    print(f"Using device: {config.DEVICE}")
    
    # 1. Create data loader (point to your Fakeddit CSV)
    # Use pruned 100-sample dataset
    csv_path = "./data/fakeddit/train_100.csv"
    if not os.path.exists(csv_path):
        # Fallback to full dataset
        csv_path = "./data/fakeddit/train.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        print("Please download Fakeddit from: https://github.com/entitize/Fakeddit")
        return
    
    print("Loading data...")
    dataloader = get_dataloader(csv_path, batch_size=config.BATCH_SIZE)
    
    # 2. Initialize feature extractor
    print("Initializing models...")
    extractor = FeatureExtractor().to(config.DEVICE)
    extractor.eval()  # Set to evaluation mode
    
    # 3. Test on a single batch
    print("\nTesting feature extraction on first batch...")
    batch = next(iter(dataloader))
    
    texts = batch['texts']
    images = batch['images']
    
    print(f"Batch size: {len(texts)}")
    print(f"Sample text: {texts[0][:50]}...")
    print(f"Sample image type: {type(images[0])}")
    
    # Extract features
    with torch.no_grad():
        features = extractor(texts, images)
    
    # 4. Print feature shapes (this is your proof of implementation)
    print("\n" + "="*50)
    print("FEATURE EXTRACTION SUCCESSFUL!")
    print("="*50)
    for name, tensor in features.items():
        print(f"{name}: shape {tensor.shape}")
    
    # 5. Save features for later use (optional)
    print("\nSaving sample features to 'features_sample.pt'...")
    torch.save({
        'features': features,
        'texts': texts,
        'labels': batch['labels']
    }, 'features_sample.pt')
    
    print("\nDone! You have successfully implemented the feature extraction module.")
    print("Next steps: Implement the Hierarchical Fusion module.")

if __name__ == "__main__":
    main()