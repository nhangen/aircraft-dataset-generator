# Utility Scripts

This directory contains utility scripts for dataset generation and processing.

## Scripts

### `batch_dataset_generator_40k_with_annotations.py`
**Purpose:** Generates large-scale aircraft datasets with PyVista rendering and COCO-style annotations.

**Features:**
- Batch processing to prevent memory leaks (subprocess isolation)
- Generates 40,000 images with oriented bounding boxes (OBB)
- Automatic train/val/test splitting (70%/20%/10%)
- Includes depth maps and COCO-style JSON annotations
- Supports F15, B52, and C130 aircraft models
- Progress tracking with state persistence

**Usage:**
```bash
python scripts/batch_dataset_generator_40k_with_annotations.py
```

**Configuration:**
- Edit `OUTPUT_DIR` to change dataset location
- Edit `BATCH_SIZE` to adjust images per batch (default: 50 scenes = 400 images)
- Edit split ratios in `target_train`, `target_val`, `target_test`

**Output Structure:**
```
dataset_name/
├── train/
│   ├── images/
│   ├── depth/
│   └── annotations/
├── val/
│   ├── images/
│   ├── depth/
│   └── annotations/
└── test/
    ├── images/
    ├── depth/
    └── annotations/
```

**Notes:**
- Successfully generated 40,000 image dataset on Sep 22, 2025
- Total generation time: ~2 hours
- Memory usage: Stable at ~5.6MB per main process
- No GPU memory leaks with subprocess isolation