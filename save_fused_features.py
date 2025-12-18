"""
Export fused features and attention weights using the trained model.
Saves:
 - outputs/full_dataset/fused_features.pt  (tensor [N,512])
 - outputs/full_dataset/attention_weights.pt (list/dict)
 - outputs/full_dataset/fused_features.csv (id + flattened vector)
"""
import torch
from pathlib import Path
import pandas as pd
from fusion import FakeNewsDetectionModel
from config import config

PROJECT_ROOT = Path(__file__).parent
FEATURE_FILE = PROJECT_ROOT / 'outputs' / 'full_dataset' / 'features_full.pt'
MODEL_FILE = PROJECT_ROOT / 'outputs' / 'training' / 'fake_news_detection_model.pt'
OUT_DIR = PROJECT_ROOT / 'outputs' / 'full_dataset'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = config.DEVICE

if not FEATURE_FILE.exists():
    raise SystemExit(f"Feature file not found: {FEATURE_FILE}")
if not MODEL_FILE.exists():
    raise SystemExit(f"Model checkpoint not found: {MODEL_FILE}")

print(f"Loading features from {FEATURE_FILE}")
data = torch.load(FEATURE_FILE)
features = {
    'text_pattern': data['text_pattern'],
    'text_semantic': data['text_semantic'],
    'image_pattern': data['image_pattern'],
    'image_semantic': data['image_semantic'],
}
ids = data.get('ids', None)
labels = data.get('labels', None)

# Instantiate model and load weights
model = FakeNewsDetectionModel(feature_dim=512, num_heads=8, num_fusion_layers=2)
state = torch.load(MODEL_FILE, map_location='cpu')
model.load_state_dict(state)
model.eval()
model.to(DEVICE)

# Run in batches
batch_size = 16
n = features['text_pattern'].shape[0]
fused_list = []
attention_list = []

with torch.no_grad():
    for i in range(0, n, batch_size):
        batch = {}
        for k in features:
            batch[k] = features[k][i:i+batch_size].to(DEVICE)
        logits, fused, attention = model(batch)
        fused_list.append(fused.cpu())
        attention_list.append(attention)

fused_all = torch.cat(fused_list, dim=0)
# Save fused features tensor
fused_path = OUT_DIR / 'fused_features.pt'
torch.save({'fused': fused_all, 'ids': ids, 'labels': labels}, fused_path)
print(f"Saved fused features: {fused_path}")

# Save attention weights (list may contain tensors on device)
att_path = OUT_DIR / 'attention_weights.pt'
torch.save(attention_list, att_path)
print(f"Saved attention weights: {att_path}")

# Also export CSV (ids + flattened vector)
csv_path = OUT_DIR / 'fused_features.csv'
rows = []
for idx in range(fused_all.shape[0]):
    vec = fused_all[idx].numpy().tolist()
    row = {'id': ids[idx] if ids is not None else idx}
    # store as semicolon-joined string to keep CSV readable
    row['fused'] = ' '.join([f"{x:.6f}" for x in vec])
    rows.append(row)

pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"Saved fused features CSV: {csv_path}")

print('Done.')
