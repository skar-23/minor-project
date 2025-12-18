"""
Run a short sweep comparing balancing strategies and save summary metrics.
Saves:
 - outputs/final/balancing_sweep_results.csv
 - outputs/final/<strategy>_model.pt (best checkpoint per strategy)
"""
import torch
from pathlib import Path
import pandas as pd
from fusion import FakeNewsDetectionModel
from train_model import Trainer
from config import config

PROJECT_ROOT = Path(__file__).parent
OUT_DIR = PROJECT_ROOT / 'outputs' / 'final'
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = ['none', 'class_weight', 'oversample', 'focal']
RESULTS = []

for strat in STRATEGIES:
    print(f"\n=== Strategy: {strat} ===")
    model = FakeNewsDetectionModel(feature_dim=512, num_heads=8, num_fusion_layers=2)
    trainer = Trainer(model, device=config.DEVICE, learning_rate=1e-3, balance_strategy=strat)
    features_dict, labels = trainer.load_features()
    train_loader, val_loader, test_loader, indices = trainer.create_dataloaders(features_dict, labels, train_split=0.7, val_split=0.15)

    # Short training for comparison
    trainer.train(train_loader, val_loader, num_epochs=20, early_stopping_patience=4)

    # Evaluate
    eval_res = trainer.evaluate(test_loader)
    print(f"Strategy={strat} | Test Acc={eval_res['accuracy']:.4f} | F1={eval_res['f1']:.4f}")

    # Save model
    model_path = OUT_DIR / f"model_{strat}.pt"
    torch.save(trainer.model.state_dict(), model_path)

    RESULTS.append({
        'strategy': strat,
        'accuracy': float(eval_res['accuracy']),
        'precision': float(eval_res['precision']),
        'recall': float(eval_res['recall']),
        'f1': float(eval_res['f1'])
    })

# Save results
df = pd.DataFrame(RESULTS)
results_path = OUT_DIR / 'balancing_sweep_results.csv'
df.to_csv(results_path, index=False)
print(f"Saved sweep results: {results_path}")
print('Done.')
