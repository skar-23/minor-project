"""
Train on the saved train split (69 samples) and evaluate on saved val split (14 samples).
Uses split indices saved in outputs/training/split_indices.json from previous runs.
"""
import json
from pathlib import Path
import torch
from fusion import FakeNewsDetectionModel
from train_model import Trainer
from config import config

PROJECT_ROOT = Path(__file__).parent
SPLIT_FILE = PROJECT_ROOT / 'outputs' / 'training' / 'split_indices.json'
MODEL_OUT = PROJECT_ROOT / 'outputs' / 'final' / 'train_on_train_eval_val_model.pt'
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

if not SPLIT_FILE.exists():
    raise SystemExit(f"Split indices not found: {SPLIT_FILE} — run a training run first to generate splits.")

with open(SPLIT_FILE, 'r') as f:
    splits = json.load(f)

train_idx = splits['train_indices']
val_idx = splits['val_indices']

print(f"Loaded splits — train: {len(train_idx)}, val: {len(val_idx)}")

# Load features
model = FakeNewsDetectionModel(feature_dim=512, num_heads=8, num_fusion_layers=2)
trainer = Trainer(model, device=config.DEVICE, learning_rate=1e-3, balance_strategy='class_weight')
features_dict, labels = trainer.load_features()

# Create datasets explicitly using provided indices
from train_model import FeatureDataset, collate_features, BATCH_SIZE

train_dataset = FeatureDataset(features_dict, labels, train_idx)
val_dataset = FeatureDataset(features_dict, labels, val_idx)

from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_features)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_features)

# Train longer on full train split
trainer.train(train_loader, val_loader, num_epochs=100, early_stopping_patience=10)

# Evaluate on val split
results = trainer.evaluate(val_loader)
print('\nEvaluation on val split:')
print(f"  Accuracy: {results['accuracy']:.4f}")
print(f"  Precision: {results['precision']:.4f}")
print(f"  Recall: {results['recall']:.4f}")
print(f"  F1: {results['f1']:.4f}")

# Save final model
torch.save(trainer.model.state_dict(), MODEL_OUT)
print(f"Saved model: {MODEL_OUT}")
