# Aircraft Dataset Generator

A comprehensive toolkit for generating synthetic aircraft datasets for machine learning applications using **real 3D aircraft models** and PyVista rendering.

## Features

- **🛩️ Real Aircraft Models**: F-15 Eagle (50K vertices), B-52 Stratofortress (21K vertices), C-130 Hercules (97K vertices)
- **📊 Multiple Formats**: 2D silhouettes, 3D multi-view, baseline wireframes
- **📦 3D Bounding Boxes**: Oriented bounding boxes for pose estimation training
- **🎯 High Quality**: PyVista rendering with proper lighting, shading, and surfaces
- **🚀 Ready to Use**: 90+ sample images included (10 per aircraft × 9 categories)
- **📁 Flexible Output**: COCO, YOLO, Pascal VOC annotations
- **🔧 Extensible**: Add custom STL/OBJ/GLB aircraft models

## Quick Start

### Installation
```bash
# Install dependencies
pip install pyvista
pip install -e .

# Or use conda environment
conda env create -f environment.yml
conda activate aircraft-toolkit
pip install -e .
```

### Generate Datasets

**3D Multi-view (Recommended)**
```python
from aircraft_toolkit import Dataset3D

dataset = Dataset3D(
    aircraft_types=['F15', 'B52', 'C130'],
    num_scenes=100,
    views_per_scene=8,
    include_oriented_bboxes=True,  # Enable 3D bounding boxes
    image_size=(512, 512)
)
results = dataset.generate('output/aircraft_3d')
```

**2D Silhouettes**
```python
from aircraft_toolkit import Dataset2D

dataset = Dataset2D(
    aircraft_types=['F15', 'B52', 'C130'],
    num_samples=1000,
    image_size=(224, 224)
)
results = dataset.generate('output/aircraft_2d')
```

### View Sample Images
Check `sample_images/` for examples of all output types.

## Sample Images

The repository includes 90+ sample images organized by type:

```
sample_images/
├── 2d/                    # 2D silhouette samples
│   ├── F15/              # f15_2d_01.png → f15_2d_10.png
│   ├── B52/              # b52_2d_01.png → b52_2d_10.png
│   └── C130/             # c130_2d_01.png → c130_2d_10.png
└── 3d/                    # 3D rendered samples
    ├── baseline/          # Original wireframe samples
    │   ├── F15/          # f15_baseline_01.png → f15_baseline_10.png
    │   ├── B52/          # b52_baseline_01.png → b52_baseline_10.png
    │   └── C130/         # c130_baseline_01.png → c130_baseline_10.png
    └── pyvista/          # Real aircraft models
        ├── F15/          # f15_real_01.png → f15_real_10.png
        ├── B52/          # b52_real_01.png → b52_real_10.png
        ├── C130/         # c130_real_01.png → c130_real_10.png
        └── bounding_boxes/  # 3D bounding box samples
            ├── F15/      # f15_3d_bbox_01.png → f15_3d_bbox_10.png
            ├── B52/      # b52_3d_bbox_01.png → b52_3d_bbox_10.png
            └── C130/     # c130_3d_bbox_01.png → c130_3d_bbox_10.png
```

## Aircraft Models

**Real 3D Models Included:**
- **F-15 Eagle**: McDonnell Douglas F-15E Strike Eagle (50,637 vertices)
- **B-52 Stratofortress**: Boeing B-52 strategic bomber (21,392 vertices)
- **C-130 Hercules**: Lockheed C-130 transport aircraft (96,662 vertices)

**Model Sources:**
Models are automatically loaded from `models/aircraft/`:
- `f15.glb` - F-15E Strike Eagle
- `b52.glb` - B-52 Stratofortress
- `c130.obj` - C-130 Hercules

## 3D Bounding Boxes for Pose Estimation

Generate oriented 3D bounding boxes that follow aircraft orientation for pose estimation training:

```python
from aircraft_toolkit import Dataset3D

dataset = Dataset3D(
    aircraft_types=['F15', 'B52', 'C130'],
    num_scenes=100,
    include_oriented_bboxes=True,  # Enable 3D bounding boxes
    image_size=(512, 512)
)
results = dataset.generate('output/pose_estimation_data')
```

**Features:**
- ✅ **Oriented Bounding Boxes**: Boxes rotate with aircraft orientation
- ✅ **Full Coverage**: Encompasses entire aircraft (nose-to-tail, wing-to-wing)
- ✅ **Color-Coded Visualization**: Cyan bottom, green top, yellow vertical edges
- ✅ **Corner Labels**: Numbered 0-7 for debugging
- ✅ **Ground Clearance**: All Z coordinates > 0
- ✅ **Diverse Poses**: Pitch, yaw, roll variations for robust training

## Adding Custom Models

1. Download STL/OBJ/GLB aircraft models
2. Place in `models/aircraft/` with aircraft type names:
   ```
   models/aircraft/
   ├── f15.stl      # Will be used for F-15
   ├── b52.obj      # Will be used for B-52
   └── c130.glb     # Will be used for C-130
   ```
3. Models are automatically detected and used

**Model Sources:**
- [Printables.com](https://www.printables.com/search/models?q=military%20aircraft)
- [GrabCAD](https://grabcad.com/library?query=military%20aircraft)
- [Thingiverse](https://www.thingiverse.com/search?q=aircraft)

## Development

### Example Scripts
```bash
# Generate examples
python example_scripts/generate_examples.py --mode 3d

# Test specific aircraft
python example_scripts/generate_examples.py --mode test
```

### Provider Architecture
The system uses a modular provider architecture:
- **Basic Provider**: Wireframe fallback (14 vertices)
- **PyVista Provider**: Real 3D models (20K-97K vertices)

Priority: Custom models → Real models → Basic wireframes

## Output Structure

Generated datasets follow this structure:
```
output/
├── train/
│   ├── images/           # RGB images (PNG)
│   ├── depth/           # Depth maps (optional)
│   └── annotations.json # COCO format annotations
├── val/
└── test/
```

**Annotations include:**
- Aircraft pose (rotation, translation)
- Camera parameters
- Bounding boxes
- Aircraft type labels

## Requirements

**Core Dependencies:**
- Python 3.9+
- PyVista 0.40+
- NumPy, Pillow, OpenCV
- Trimesh, SciPy

**Optional:**
- Custom STL/OBJ/GLB aircraft models

## License

This toolkit is for educational and research purposes. Aircraft models are from public sources with appropriate attribution.
