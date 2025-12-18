"""
Training pipeline for Fake News Detection Model.
Loads extracted features, trains hierarchical fusion + classifier.
Includes validation, early stopping, and model checkpointing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

from fusion import FakeNewsDetectionModel
from config import config

DEVICE = config.DEVICE
BATCH_SIZE = config.BATCH_SIZE
PROJECT_ROOT = Path(__file__).parent


class FeatureDataset(Dataset):
    """Dataset for pre-extracted features."""
    
    def __init__(self, features_dict, labels, indices):
        self.features_dict = features_dict
        self.labels = labels
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        features = {
            k: v[sample_idx] for k, v in self.features_dict.items()
        }
        label = self.labels[sample_idx]
        return features, label


def collate_features(batch):
    """Collate function for feature batches."""
    features_list, labels_list = zip(*batch)
    
    # Stack features for each stream
    batch_features = {}
    for key in features_list[0].keys():
        batch_features[key] = torch.stack([f[key] for f in features_list])
    
    batch_labels = torch.stack(labels_list)
    
    return batch_features, batch_labels


class Trainer:
    """Training orchestrator for fake news detection model."""
    
    def __init__(self, model, device=DEVICE, learning_rate=1e-3, weight_decay=1e-5, balance_strategy='none'):
        self.model = model.to(device)
        self.device = device
        self.balance_strategy = balance_strategy  # options: 'none','class_weight','oversample','focal'

        # default criterion; may be overridden after loading labels
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0
        self.best_model_state = None
        
    def load_features(self, feature_file='outputs/full_dataset/features_full.pt'):
        """Load pre-extracted features and labels."""
        feature_path = PROJECT_ROOT / feature_file
        
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        
        print(f"Loading features from {feature_path}...")
        data = torch.load(feature_path)
        
        # Extract feature streams and labels
        features_dict = {
            'text_pattern': data['text_pattern'].to(self.device),
            'text_semantic': data['text_semantic'].to(self.device),
            'image_pattern': data['image_pattern'].to(self.device),
            'image_semantic': data['image_semantic'].to(self.device),
        }
        labels = data['labels'].long().to(self.device)
        
        print(f"Loaded {len(labels)} samples")
        print(f"  - Real: {(labels == 1).sum().item()}")
        print(f"  - Fake: {(labels == 0).sum().item()}")

        # Configure balancing strategies based on label distribution
        # counts: tensor of class counts (index == label)
        counts = torch.bincount(labels.cpu())
        counts = counts.float()
        if counts.numel() < 2:
            counts = torch.cat([counts, torch.tensor([0.0])])

        if self.balance_strategy == 'class_weight':
            # Inverse frequency weighting (normalized)
            class_weights = 1.0 / (counts + 1e-8)
            class_weights = class_weights / class_weights.sum() * 2.0
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using class-weighted CrossEntropyLoss: {class_weights.cpu().tolist()}")

        elif self.balance_strategy == 'focal':
            # Focal loss implementation
            class FocalLoss(nn.Module):
                def __init__(self, gamma=2.0, weight=None):
                    super().__init__()
                    self.gamma = gamma
                    self.weight = weight
                def forward(self, input, target):
                    ce = nn.functional.cross_entropy(input, target, weight=self.weight, reduction='none')
                    pt = torch.exp(-ce)
                    loss = ((1 - pt) ** self.gamma) * ce
                    return loss.mean()

            weight = None
            if counts.sum() > 0:
                weight = (1.0 / (counts + 1e-8))
                weight = weight / weight.sum() * 2.0
                weight = weight.to(self.device)
            self.criterion = FocalLoss(weight=weight)
            print("Using FocalLoss for balancing")

        return features_dict, labels
    
    def create_dataloaders(self, features_dict, labels, train_split=0.7, val_split=0.15):
        """Create train/val/test dataloaders."""
        num_samples = labels.shape[0]
        num_train = int(num_samples * train_split)
        num_val = int(num_samples * val_split)
        num_test = num_samples - num_train - num_val
        
        print(f"\nSplitting dataset:")
        print(f"  Train: {num_train} samples")
        print(f"  Val:   {num_val} samples")
        print(f"  Test:  {num_test} samples")
        
        # Create indices
        indices = torch.randperm(num_samples).tolist()
        train_idx = indices[:num_train]
        val_idx = indices[num_train:num_train+num_val]
        test_idx = indices[num_train+num_val:]
        
        # Create datasets
        train_dataset = FeatureDataset(features_dict, labels, train_idx)
        val_dataset = FeatureDataset(features_dict, labels, val_idx)
        test_dataset = FeatureDataset(features_dict, labels, test_idx)
        
        # Optionally create an oversampling sampler for training
        train_sampler = None
        if self.balance_strategy == 'oversample':
            labels_cpu = labels.cpu()
            class_counts = torch.bincount(labels_cpu)
            class_counts = class_counts.float()
            class_weights = 1.0 / (class_counts + 1e-8)
            sample_weights = class_weights[labels_cpu]
            train_weights = sample_weights[train_idx]
            train_sampler = WeightedRandomSampler(weights=train_weights.tolist(), num_samples=len(train_weights), replacement=True)
            print("Using WeightedRandomSampler for oversampling minority class")

        # Create dataloaders
        if train_sampler is not None:
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_features)
        else:
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_features)

        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_features)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_features)
        
        return train_loader, val_loader, test_loader, (train_idx, val_idx, test_idx)
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_features, batch_labels in train_loader:
            # Forward pass
            logits, _, _ = self.model(batch_features)
            loss = self.criterion(logits, batch_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * batch_labels.shape[0]
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == batch_labels).sum().item()
            total_samples += batch_labels.shape[0]
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                logits, _, _ = self.model(batch_features)
                loss = self.criterion(logits, batch_labels)
                
                total_loss += loss.item() * batch_labels.shape[0]
                predictions = logits.argmax(dim=1)
                total_correct += (predictions == batch_labels).sum().item()
                total_samples += batch_labels.shape[0]
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def train(self, train_loader, val_loader, num_epochs=50, early_stopping_patience=5):
        """Train the model."""
        print(f"\nTraining on {self.device}...\n")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Checkpoint best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} [BEST]")
            else:
                patience_counter += 1
                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Restored best model with validation accuracy: {self.best_val_acc:.4f}")
    
    def evaluate(self, test_loader):
        """Evaluate on test set."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                logits, _, _ = self.model(batch_features)
                predictions = logits.argmax(dim=1)
                
                all_preds.append(predictions.cpu())
                all_labels.append(batch_labels.cpu())
                all_logits.append(logits.cpu())
        
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        logits = torch.cat(all_logits)
        
        # Metrics
        accuracy = (preds == labels).float().mean().item()
        
        # Per-class metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(labels.numpy(), preds.numpy(), zero_division=0)
        recall = recall_score(labels.numpy(), preds.numpy(), zero_division=0)
        f1 = f1_score(labels.numpy(), preds.numpy(), zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': preds,
            'labels': labels,
            'logits': logits
        }
    
    def save_results(self, eval_results, indices):
        """Save training results and evaluation metrics."""
        output_dir = PROJECT_ROOT / 'outputs' / 'training'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / 'fake_news_detection_model.pt'
        torch.save(self.model.state_dict(), model_path)
        print(f"\nModel saved: {model_path}")
        
        # Save training history
        history_df = pd.DataFrame(self.history)
        history_path = output_dir / 'training_history.csv'
        history_df.to_csv(history_path, index=False)
        print(f"Training history saved: {history_path}")
        
        # Save evaluation metrics
        metrics = {
            'accuracy': float(eval_results['accuracy']),
            'precision': float(eval_results['precision']),
            'recall': float(eval_results['recall']),
            'f1': float(eval_results['f1']),
            'best_val_acc': float(self.best_val_acc),
            'total_epochs': len(self.history['train_loss']),
            'timestamp': datetime.now().isoformat()
        }
        metrics_path = output_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved: {metrics_path}")
        
        # Save split indices
        indices_data = {
            'train_indices': indices[0],
            'val_indices': indices[1],
            'test_indices': indices[2]
        }
        indices_path = output_dir / 'split_indices.json'
        with open(indices_path, 'w') as f:
            json.dump({k: [int(x) for x in v] for k, v in indices_data.items()}, f, indent=2)
        print(f"Split indices saved: {indices_path}")
        
        return output_dir


def main():
    """Main training script."""
    print("=" * 60)
    print("FAKE NEWS DETECTION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize trainer
    model = FakeNewsDetectionModel(feature_dim=512, num_heads=8, num_fusion_layers=2)
    # Choose balancing strategy: 'none' | 'class_weight' | 'oversample' | 'focal'
    BALANCE_STRATEGY = 'class_weight'
    trainer = Trainer(model, device=DEVICE, learning_rate=1e-3, balance_strategy=BALANCE_STRATEGY)
    
    # Load features
    features_dict, labels = trainer.load_features()
    
    # Create dataloaders
    train_loader, val_loader, test_loader, indices = trainer.create_dataloaders(
        features_dict, labels, train_split=0.7, val_split=0.15
    )
    
    # Train
    trainer.train(train_loader, val_loader, num_epochs=50, early_stopping_patience=5)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    eval_results = trainer.evaluate(test_loader)
    
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {eval_results['accuracy']:.4f}")
    print(f"  Precision: {eval_results['precision']:.4f}")
    print(f"  Recall:    {eval_results['recall']:.4f}")
    print(f"  F1-Score:  {eval_results['f1']:.4f}")
    
    # Save results
    output_dir = trainer.save_results(eval_results, indices)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
