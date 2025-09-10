# Aircraft Data Generation Toolkit

A comprehensive toolkit for generating synthetic aircraft datasets for machine learning applications.

## Features

- **2D Aircraft Generation**: Silhouette-based pose estimation datasets
- **3D Multi-view Generation**: Multi-camera aircraft datasets with depth maps
- **Multiple Aircraft Types**: Military (F-15, B-52, C-130) and civilian aircraft
- **Flexible Annotations**: COCO, YOLO, Pascal VOC format support
- **Extensible Architecture**: Easy to add custom aircraft models

## Quick Start

```python
from aircraft_toolkit import Dataset2D

# Generate 2D dataset
dataset = Dataset2D(
    aircraft_types=['F15', 'B52', 'C130'],
    num_samples=10000,
    image_size=(224, 224)
)
dataset.generate('output/aircraft_2d')
```

## Installation

```bash
pip install aircraft-data-toolkit
```

## Repository Structure

See `docs/architecture.md` for detailed framework design.