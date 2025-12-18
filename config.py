"""
Configuration file for Fake News Detection project.
Defines model names, batch sizes, device settings, and feature dimensions.
"""
import torch

class Config:
    # Data
    DATA_PATH = "./data/fakeddit"
    BATCH_SIZE = 8  # Small due to large models
    NUM_WORKERS = 2
    
    # Feature dimensions
    FEATURE_DIM = 512  # d_k in paper
    
    # Models
    BERT_MODEL = "bert-base-uncased"
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    
    # Image preprocessing
    IMG_SIZE = 224  # Swin-T and CLIP expect 224x224
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
config = Config()