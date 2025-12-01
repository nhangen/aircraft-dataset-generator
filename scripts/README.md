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

## Expanded Rotation Datasets

### Breaking the 120° Convergence Barrier

Traditional pose estimation models plateau at ~120° error due to constrained training data. The expanded rotation functionality breaks this barrier:

**Original Constraints (causing 120° plateau):**

- Pitch: ±30° (60° range)
- Roll: ±15° (30° range)
- Yaw: ±180° (360° range)

**Expanded Ranges (solution):**

```python
Dataset3D(
    pitch_range=(-90, 90),    # 3x expansion
    roll_range=(-180, 180),   # 12x expansion
    yaw_range=(-180, 180)     # Full coverage
)
```

**Results:** 30x larger pose space for breakthrough model performance

## Task-Specific Dataset Generation

Both 2D and 3D datasets support flexible task modes for targeted training:

### Task Mode Examples

**Classification Only:**

```python
from aircraft_toolkit import Dataset2D, Dataset3D

# Generate datasets for aircraft type detection
dataset_2d = Dataset2D(
    aircraft_types=['F15', 'B52', 'C130'],
    num_samples=5000,
    task_mode='classification'  # Only aircraft type labels
)

dataset_3d = Dataset3D(
    aircraft_types=['F15', 'B52', 'C130'],
    num_scenes=1000,
    task_mode='classification'  # Only aircraft type labels
)
```

**Pose Estimation Only:**

```python
# Generate datasets for 6DOF pose estimation
dataset_2d = Dataset2D(
    aircraft_types=['F15', 'B52', 'C130'],
    num_samples=5000,
    task_mode='pose',  # Pose + bounding boxes only
    pose_range={
        'pitch': (-90, 90),    # Expanded for better training
        'roll': (-180, 180),
        'yaw': (-180, 180)
    }
)

dataset_3d = Dataset3D(
    aircraft_types=['F15', 'B52', 'C130'],
    num_scenes=1000,
    task_mode='pose',  # Pose + camera params only
    include_oriented_bboxes=True,
    pitch_range=(-90, 90),     # Barrier-breaking ranges
    roll_range=(-180, 180)
)
```

**Multi-Task Learning:**

```python
# Generate complete annotations for joint training
dataset_both = Dataset3D(
    aircraft_types=['F15', 'B52', 'C130'],
    num_scenes=1000,
    task_mode='both',  # Classification + pose + all features
    include_oriented_bboxes=True
)
```

### `aircraft_40k_expanded_rotations` Dataset

- **40,000 images** with expanded rotation ranges
- **Generated:** Sep 25, 2025
- **Time:** 114 minutes (1.9 hours)
- **Purpose:** Break 120° convergence barrier in pose estimation
- **Validation:** 69.6% pitch violations, 92.0% roll violations (breaking old constraints)

**Notes:**

- Successfully generated 40,000 image dataset on Sep 22, 2025
- Total generation time: ~2 hours
- Memory usage: Stable at ~5.6MB per main process
- No GPU memory leaks with subprocess isolation
