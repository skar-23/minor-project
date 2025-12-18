# Multimodal Fake News Detection using Hierarchical Fusion

## Project Overview

A deep learning system for detecting fake news by fusing multimodal features (text + images) using a hierarchical attention mechanism.

---

## Directory Structure

```
├── config.py                 # Configuration (models, device, batch size)
├── data_loader.py            # Dataset class for Fakeddit data
├── feature_extracter.py      # 4-encoder feature extraction module
├── preprocess.py             # Data preprocessing script
├── train.py                  # Main training pipeline
├── features_sample.pt        # Extracted features (sample batch)
├── data/
│   └── fakeddit/
│       ├── train.csv         # Original 564K samples
│       └── train_100.csv     # Pruned 99 samples (cleaned)
└── fnd(1).pdf               # Project requirements
```

---

## Core Components

### 1. **Feature Extraction** (4 Streams)

- **Text Pattern** (BERT): Linguistic patterns, syntax, grammar
- **Text Semantic** (CLIP): Topic, meaning, context
- **Image Pattern** (Swin-T): Visual artifacts, textures
- **Image Semantic** (CLIP): Visual meaning, alignment

Each stream outputs 512-d vectors → **2048-d total per sample**

### 2. **Dataset**

- **Source**: Fakeddit (Reddit posts with images)
- **Size**: 99 samples (58 real, 41 fake)
- **Format**: Text + Image + Binary Label
- **File**: `data/fakeddit/train_100.csv`

---

## Usage

### Step 1: Preprocess Dataset

```bash
python preprocess.py
```

**Output:** `data/fakeddit/train_100.csv` (cleaned, 99 samples)

### Step 2: Extract Features

```bash
python train.py
```

**Output:** `features_sample.pt` (8-sample batch with 4 feature streams)

### Step 3: Results

Features are ready for the **Hierarchical Fusion Module** (next phase)

---

## Model Specifications

| Component      | Model             | Params | Output Dim |
| -------------- | ----------------- | ------ | ---------- |
| Text Pattern   | BERT-base-uncased | 110M   | 768 → 512  |
| Text Semantic  | CLIP (text)       | 305M   | 512        |
| Image Pattern  | Swin-T            | 28M    | 768 → 512  |
| Image Semantic | CLIP (image)      | 305M   | 512        |

**Total:** ~748M parameters (all frozen, only MLPs trained)

---

## Results

✅ **Feature Extraction Pipeline**: Fully functional  
✅ **Dataset**: Pruned & cleaned (99 samples)  
✅ **Features Extracted**: 4 streams per sample, saved to `features_sample.pt`

### Next Steps

1. Implement **Hierarchical Fusion Module** (attention-based fusion)
2. Build **Classification Head** (fake/real prediction)
3. Train end-to-end on full dataset

---

## Requirements

```
torch
torchvision
transformers
pillow
pandas
requests
```

Install via:

```bash
pip install torch torchvision transformers pillow pandas requests
```
