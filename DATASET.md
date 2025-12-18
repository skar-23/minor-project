# Dataset Information

## Fakeddit Dataset

The project uses the **Fakeddit Dataset** for multimodal fake news detection.

### Dataset Details

- **Name**: Fakeddit - A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection
- **Source**: Reddit Posts (subreddits: r/Cringetopia, r/Facepalm)
- **Total Samples**: 564,000 multimodal posts
- **Current Working Subset**: 99 samples (pruned for development)

### Dataset Format

Each sample contains:

- **ID**: Unique post identifier
- **Title**: Post text (raw and cleaned versions available)
- **Image**: Associated Reddit image (PNG/JPEG format)
- **Label**: Binary classification (0=Real, 1=Fake)

### Working Dataset Statistics

| Metric         | Value           |
| -------------- | --------------- |
| Total Samples  | 99              |
| Real Posts (0) | 58 (58.6%)      |
| Fake Posts (1) | 41 (41.4%)      |
| Data Quality   | 100% (no nulls) |
| Image URLs     | 100% valid      |

### How to Download

**Option 1: Download from Kaggle** (Recommended)

```bash
# Install kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d vanshikavmittal/fakeddit-dataset

# Extract
unzip fakeddit-dataset.zip -d data/
```

**Direct Link**: https://www.kaggle.com/datasets/vanshikavmittal/fakeddit-dataset

**Option 2: Original Repository**

- GitHub: https://github.com/entitize/Fakeddit

### File Structure After Download

```
data/
└── fakeddit/
    ├── train.csv              # Full training set (564K samples)
    ├── test.csv              # Test set
    ├── validate.csv          # Validation set
    └── images/               # Associated images (optional)
```

### Dataset License

Please refer to the original Fakeddit repository for licensing information.

### Citation

If you use this dataset, please cite:

```
@inproceedings{nakamura2019fakeddit,
  title={Fakeddit: A new multimodal benchmark dataset for fine-grained fake news detection},
  author={Nakamura, Kai and Levy, Sharon and Wang, William Yang},
  booktitle={Proceedings of the Eleventh ACM Conference on Web Search and Data Mining},
  pages={432--440},
  year={2020}
}
```

### Project Usage

The project works with:

1. **train.csv** - Main dataset file (required)
2. **raw_100.csv** - First 100 raw samples (included in repo)
3. **preprocessed_data.csv** - Cleaned 99 samples (included in repo)

For full feature extraction on all 564K samples, download the complete dataset from Kaggle.

### Notes

- The full `train.csv` (146.69 MB) exceeds GitHub's 100 MB limit, so it's not included in the repository
- For development, use the included `raw_100.csv` (100 samples) or `preprocessed_data.csv` (99 cleaned samples)
- For production training, download the full dataset from Kaggle

---

**Dataset Setup Instructions**:

1. Download dataset from Kaggle link above
2. Extract to `data/fakeddit/` folder
3. Run `preprocess.py` to clean data
4. Run `extract_full_features.py` to extract features
