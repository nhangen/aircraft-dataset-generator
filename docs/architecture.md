# Aircraft Data Generation Toolkit Architecture

## Repository Structure

```
aircraft-data-toolkit/
├── README.md                 # Main documentation
├── LICENSE                   # MIT license
├── setup.py                  # Package installation
├── requirements.txt          # Dependencies
├── example_scripts/          # Usage examples
│   ├── basic_2d_generation.py
│   ├── advanced_3d_generation.py
│   ├── custom_aircraft_model.py
│   ├── annotation_formats.py
│   └── batch_processing.py
├── aircraft_toolkit/         # Main package
│   ├── __init__.py
│   ├── core/                 # Core generation engines
│   │   ├── __init__.py
│   │   ├── dataset_2d.py     # 2D dataset generator
│   │   ├── dataset_3d.py     # 3D multi-view generator
│   │   ├── base.py          # Base classes and interfaces
│   │   └── validation.py     # Dataset validation
│   ├── models/              # Aircraft model definitions
│   │   ├── __init__.py
│   │   ├── military/         # Military aircraft
│   │   │   ├── __init__.py
│   │   │   ├── fighter.py    # F-15, F-16, F-22, etc.
│   │   │   ├── bomber.py     # B-52, B-1, B-2, etc.
│   │   │   └── transport.py  # C-130, C-17, KC-135, etc.
│   │   ├── civilian/         # Civilian aircraft
│   │   │   ├── __init__.py
│   │   │   ├── airliner.py   # 737, A320, 777, etc.
│   │   │   ├── general.py    # Cessna, Piper, etc.
│   │   │   └── helicopter.py # Bell, Robinson, etc.
│   │   ├── base_model.py     # Base aircraft model class
│   │   └── custom.py         # Custom aircraft builder
│   ├── rendering/           # Rendering engines
│   │   ├── __init__.py
│   │   ├── silhouette.py    # 2D silhouette rendering
│   │   ├── multiview.py     # 3D multi-view rendering
│   │   ├── depth.py         # Depth map generation
│   │   ├── camera.py        # Camera positioning system
│   │   └── lighting.py      # Lighting models
│   ├── annotations/         # Annotation systems
│   │   ├── __init__.py
│   │   ├── pose_2d.py       # 2D pose annotations
│   │   ├── pose_3d.py       # 3D pose + keypoints
│   │   ├── bounding_box.py  # 2D/3D bounding boxes
│   │   └── formats/         # Export formats
│   │       ├── __init__.py
│   │       ├── coco.py      # COCO format
│   │       ├── yolo.py      # YOLO format
│   │       ├── pascal_voc.py # Pascal VOC format
│   │       └── custom.py     # Custom format builder
│   ├── augmentation/        # Data augmentation
│   │   ├── __init__.py
│   │   ├── geometric.py     # Rotation, translation, scale
│   │   ├── photometric.py   # Brightness, contrast, noise
│   │   └── atmospheric.py   # Weather, visibility effects
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── geometry.py      # 3D geometry utilities
│       ├── visualization.py # Preview and debugging tools
│       ├── export.py        # Dataset export utilities
│       └── statistics.py    # Dataset statistics
├── tests/                   # Unit tests
│   ├── test_2d_generation.py
│   ├── test_3d_generation.py
│   ├── test_aircraft_models.py
│   ├── test_annotations.py
│   └── test_rendering.py
├── docs/                    # Documentation
│   ├── quickstart.md
│   ├── aircraft_models.md
│   ├── annotation_formats.md
│   ├── api_reference.md
│   └── tutorials/
│       ├── custom_aircraft.md
│       ├── advanced_rendering.md
│       └── batch_processing.md
└── data/                    # Sample data and templates
    ├── aircraft_templates/   # Aircraft shape templates
    ├── backgrounds/         # Background images
    └── example_scripts/     # Example datasets
```

## Core Components

### 1. Aircraft Model System
```python
from aircraft_toolkit.models.base_model import BaseAircraft

class BaseAircraft:
    def __init__(self, name: str, aircraft_type: str):
        self.name = name
        self.aircraft_type = aircraft_type
        self.components = {}  # wings, fuselage, tail, etc.
        
    def define_silhouette(self) -> List[Tuple[float, float]]:
        """Define 2D silhouette points"""
        pass
        
    def define_3d_keypoints(self) -> Dict[str, Tuple[float, float, float]]:
        """Define 3D keypoints (nose, wing tips, tail, etc.)"""
        pass
        
    def get_bounding_box_3d(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get 3D bounding box (min_point, max_point)"""
        pass

# Example implementation
class F15Fighter(BaseAircraft):
    def __init__(self):
        super().__init__("F-15 Eagle", "fighter")
        
    def define_silhouette(self):
        return [
            # Fuselage points (clockwise)
            (0.8, 0.0),    # Nose
            (0.6, 0.08),   # Upper nose
            (0.2, 0.12),   # Forward fuselage top
            (-0.4, 0.12),  # Rear fuselage top
            (-0.9, 0.0),   # Tail point
            (-0.4, -0.12), # Rear fuselage bottom
            (0.2, -0.12),  # Forward fuselage bottom
            (0.6, -0.08),  # Lower nose
        ]
        
    def define_3d_keypoints(self):
        return {
            'nose': (0.8, 0.0, 0.0),
            'left_wing_tip': (0.0, -0.6, -0.1),
            'right_wing_tip': (0.0, 0.6, -0.1),
            'left_tail_tip': (-0.8, -0.2, 0.3),
            'right_tail_tip': (-0.8, 0.2, 0.3),
            'tail_top': (-0.9, 0.0, 0.2),
            'left_engine': (-0.3, -0.15, -0.05),
            'right_engine': (-0.3, 0.15, -0.05)
        }
```

### 2. Dataset Generation API
```python
from aircraft_toolkit.core import Dataset2D, Dataset3D

class Dataset2D:
    def __init__(self,
                 aircraft_types: List[str],
                 num_samples: int,
                 image_size: Tuple[int, int] = (224, 224),
                 pose_range: Dict = None,
                 augmentations: List = None,
                 background_type: str = 'gradient'):
        pass
        
    def generate(self, 
                output_dir: str,
                split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                annotation_format: str = 'coco',
                num_workers: int = 4):
        """Generate dataset with specified parameters"""
        pass

class Dataset3D:
    def __init__(self,
                 aircraft_types: List[str],
                 num_scenes: int,
                 views_per_scene: int = 8,
                 camera_distance: Tuple[float, float] = (50, 200),
                 camera_height_range: Tuple[float, float] = (-30, 30),
                 include_depth_maps: bool = True,
                 include_surface_normals: bool = False):
        pass
        
    def generate(self,
                output_dir: str,
                split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                annotation_format: str = 'custom_3d',
                num_workers: int = 4):
        """Generate 3D multi-view dataset"""
        pass
```

### 3. Annotation Format System
```python
from aircraft_toolkit.annotations.formats import COCOExporter

class COCOExporter:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        
    def export(self, output_file: str):
        """Export to COCO format"""
        coco_data = {
            "info": {
                "description": "Aircraft Pose Dataset",
                "version": "1.0",
                "year": 2025
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "fighter", "supercategory": "aircraft"},
                {"id": 2, "name": "bomber", "supercategory": "aircraft"},
                {"id": 3, "name": "transport", "supercategory": "aircraft"}
            ]
        }
        # ... populate with dataset data
        
# Usage
from aircraft_toolkit.annotations.formats import COCOExporter, YOLOExporter

# Export to different formats
coco_exporter = COCOExporter('datasets/aircraft_2d')
coco_exporter.export('annotations.json')

yolo_exporter = YOLOExporter('datasets/aircraft_2d')
yolo_exporter.export('labels/')
```

### 4. Custom Aircraft Builder
```python
from aircraft_toolkit.models.custom import CustomAircraft

class CustomAircraft(BaseAircraft):
    def __init__(self, name: str):
        super().__init__(name, "custom")
        self.silhouette_points = []
        self.keypoints_3d = {}
        
    def add_fuselage(self, length: float, width: float, height: float):
        """Add fuselage component"""
        pass
        
    def add_wings(self, span: float, chord: float, sweep_angle: float = 0):
        """Add wing component"""
        pass
        
    def add_tail(self, height: float, area: float, vertical: bool = True):
        """Add tail component"""
        pass
        
    def build(self):
        """Finalize aircraft model"""
        self._generate_silhouette()
        self._generate_keypoints()

# Example usage
custom_aircraft = CustomAircraft("Custom Fighter")
custom_aircraft.add_fuselage(length=15, width=2, height=1.5)
custom_aircraft.add_wings(span=10, chord=3, sweep_angle=30)
custom_aircraft.add_tail(height=4, area=8)
custom_aircraft.build()
```

## Advanced Features

### 1. Atmospheric Effects
```python
from aircraft_toolkit.augmentation.atmospheric import WeatherEffects

weather = WeatherEffects()

# Add weather conditions
dataset_2d = Dataset2D(
    aircraft_types=['F15', 'B52'],
    num_samples=1000,
    augmentations=[
        weather.fog(density_range=(0.1, 0.5)),
        weather.rain(intensity_range=(0.2, 0.8)),
        weather.clouds(coverage_range=(0.3, 0.9))
    ]
)
```

### 2. Batch Processing
```python
from aircraft_toolkit.utils.batch import BatchProcessor

processor = BatchProcessor(num_workers=8)

# Generate multiple datasets in parallel
datasets = [
    ('military_aircraft', {'aircraft_types': ['F15', 'F16', 'B52'], 'num_samples': 5000}),
    ('civilian_aircraft', {'aircraft_types': ['Boeing737', 'A320'], 'num_samples': 3000}),
    ('helicopters', {'aircraft_types': ['Bell206', 'Robinson22'], 'num_samples': 2000})
]

processor.generate_datasets(datasets, output_base_dir='generated_datasets')
```

### 3. Dataset Validation and Statistics
```python
from aircraft_toolkit.utils.validation import DatasetValidator
from aircraft_toolkit.utils.statistics import DatasetStats

# Validate generated dataset
validator = DatasetValidator('generated_datasets/military_aircraft')
validation_report = validator.validate()
print(f"Dataset valid: {validation_report.is_valid}")
print(f"Issues found: {validation_report.issues}")

# Generate statistics
stats = DatasetStats('generated_datasets/military_aircraft')
print(f"Total images: {stats.total_images}")
print(f"Aircraft type distribution: {stats.aircraft_distribution}")
print(f"Pose angle coverage: {stats.pose_coverage}")
```

## Usage Examples

### 1. Quick Start - 2D Dataset
```python
from aircraft_toolkit import Dataset2D

# Generate basic 2D dataset
dataset = Dataset2D(
    aircraft_types=['F15', 'B52', 'C130'],
    num_samples=10000,
    image_size=(224, 224)
)

dataset.generate(
    output_dir='aircraft_2d_dataset',
    split_ratios=(0.7, 0.2, 0.1),
    annotation_format='coco'
)
```

### 2. Advanced 3D Dataset with Custom Settings
```python
from aircraft_toolkit import Dataset3D
from aircraft_toolkit.augmentation import GeometricAugmentation, PhotometricAugmentation

# Create augmentation pipeline
augmentations = [
    GeometricAugmentation.random_rotation(range_degrees=(-15, 15)),
    GeometricAugmentation.random_scale(range_factor=(0.8, 1.2)),
    PhotometricAugmentation.brightness(range_factor=(0.7, 1.3)),
    PhotometricAugmentation.contrast(range_factor=(0.8, 1.2))
]

# Generate 3D dataset
dataset_3d = Dataset3D(
    aircraft_types=['F15', 'F16', 'B52', 'C130'],
    num_scenes=2000,
    views_per_scene=12,  # More views for better coverage
    camera_distance=(30, 300),
    camera_height_range=(-45, 45),
    include_depth_maps=True,
    include_surface_normals=True
)

dataset_3d.generate(
    output_dir='aircraft_3d_multiview',
    split_ratios=(0.8, 0.1, 0.1),
    annotation_format='custom_3d',
    num_workers=8
)
```

### 3. Custom Aircraft and Export
```python
from aircraft_toolkit.models.custom import CustomAircraft
from aircraft_toolkit.annotations.formats import YOLOExporter

# Create custom aircraft
stealth_fighter = CustomAircraft("Stealth Fighter")
stealth_fighter.add_fuselage(length=18, width=3, height=2)
stealth_fighter.add_wings(span=12, chord=4, sweep_angle=45)
stealth_fighter.add_tail(height=3, area=6, vertical=True)
stealth_fighter.build()

# Generate dataset with custom aircraft
dataset = Dataset2D(
    aircraft_types=[stealth_fighter],  # Can mix built-in and custom
    num_samples=5000
)
dataset.generate('custom_stealth_dataset')

# Export to YOLO format
yolo_exporter = YOLOExporter('custom_stealth_dataset')
yolo_exporter.export('yolo_labels/')
```

This architecture provides a comprehensive, extensible toolkit for generating high-quality aircraft datasets for machine learning applications.