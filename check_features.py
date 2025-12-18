import torch
from pathlib import Path

feature_file = Path("outputs/full_dataset/features_full.pt")
if feature_file.exists():
    data = torch.load(feature_file)
    print("Keys in feature file:", data.keys() if isinstance(data, dict) else "Not a dict")
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}: (dict with keys: {list(value.keys())})")
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.shape}")
            else:
                print(f"  {key}: {type(value)}")
    labels = data['labels'] if isinstance(data, dict) else None
    if labels is not None:
        unique, counts = labels.unique(return_counts=True)
        print("Label distribution:")
        for u, c in zip(unique.tolist(), counts.tolist()):
            print(f"  {u}: {c}")
else:
    print("Feature file not found!")
