# Aircraft Data Generation Toolkit

A comprehensive toolkit for generating synthetic aircraft datasets for machine learning applications with **modular 3D model providers** for high-quality, realistic aircraft geometry.

## 🆕 **TiGL Integration** - Professional-Grade Aircraft Models

This toolkit now supports **TiGL (TiGL Geometry Library)** for generating **NURBS-based, parametric aircraft models** with **100x-1000x** more detail than basic hand-coded meshes.

### Quality Comparison

| Provider | Vertices | Quality | Use Case |
|----------|----------|---------|----------|
| **Basic** | ~14 | Hand-coded approximation | Quick prototyping, TiGL unavailable |
| **TiGL** | 1,000-10,000+ | Professional NURBS geometry | Production ML training |

## Features

- **🚀 Modular Provider System**: Switch between Basic and TiGL providers seamlessly
- **🛩️ Professional Aircraft Models**: CPACS-based parametric F-15, B-52, C-130 models
- **🎯 Multiple Detail Levels**: Low/Medium/High quality settings for different use cases
- **🔄 Automatic Fallback**: Gracefully degrades to basic models if TiGL unavailable
- **2D Aircraft Generation**: Silhouette-based pose estimation datasets
- **3D Multi-view Generation**: Multi-camera aircraft datasets with depth maps
- **Flexible Annotations**: COCO, YOLO, Pascal VOC format support
- **🏗️ Extensible Architecture**: Easy to add custom aircraft models and providers

## Quick Start

### Basic Installation
```bash
pip install -e .
```

### TiGL Provider (Optional - for High-Quality Models)
```bash
conda install -c dlr-sc tigl3
```

### Generate High-Quality Dataset
```python
from aircraft_toolkit import Dataset3D

# Automatic provider selection (TiGL if available, Basic otherwise)
dataset = Dataset3D(
    aircraft_types=['F15', 'B52', 'C130'],
    num_scenes=1000,
    views_per_scene=8,
    image_size=(512, 512)
)
results = dataset.generate('output/aircraft_3d')
print(f"Generated with {dataset.provider_name} provider")
```

### Provider Selection
```python
from aircraft_toolkit.config import get_config_manager

# Force specific provider
config_mgr = get_config_manager()
config = config_mgr.get_config()
config.aircraft.model_provider = 'tigl'  # or 'basic'
config.aircraft.detail_level = 'high'    # 'low', 'medium', 'high'
config_mgr.save_config(config)

dataset = Dataset3D(aircraft_types=['F15'])
```

### Legacy 2D Generation (Unchanged)
```python
from aircraft_toolkit import Dataset2D

dataset = Dataset2D(
    aircraft_types=['F15', 'B52', 'C130'],
    num_samples=10000,
    image_size=(224, 224)
)
dataset.generate('output/aircraft_2d')
```

## 🏗️ Architecture

### Modular Provider System
```
aircraft_toolkit/
├── providers/           # Pluggable 3D model backends
│   ├── basic.py        # Hand-coded models (backward compatible)
│   ├── tigl_provider.py # TiGL CPACS-based models
│   └── base.py         # Abstract provider interface
├── config.py           # Configuration management
└── core/
    ├── dataset_2d.py   # 2D dataset generation
    └── dataset_3d.py   # 3D dataset generation (updated)
```

### Provider Capabilities

| Feature | Basic Provider | TiGL Provider |
|---------|----------------|---------------|
| Dependencies | None | TiGL + CPACS |
| Aircraft Detail | Low (~14 vertices) | High (1000+ vertices) |
| Surface Quality | Faceted | NURBS smooth |
| Parametric | No | Yes (CPACS files) |
| Detail Levels | 1 (low) | 3 (low/med/high) |
| Generation Speed | ~0.1ms | ~100ms-2s |

## 📊 Performance & Quality

### Mesh Quality Improvements
- **F-15**: 14 → 5,000+ vertices (**357x improvement**)
- **B-52**: 13 → 4,000+ vertices (**308x improvement**)
- **C-130**: 15 → 6,000+ vertices (**400x improvement**)

### Training Impact
- **Better Features**: Realistic aircraft proportions and details
- **Improved Generalization**: NURBS-based smooth surfaces
- **Configurable Quality**: Adjust detail level for computational budget

## 🚀 Examples

### Basic Usage (Backward Compatible)
```python
# Your existing code works unchanged!
from aircraft_toolkit import Dataset3D

dataset = Dataset3D(aircraft_types=['F15'])
dataset.generate('output/dataset')
```

### Advanced Configuration
```python
from aircraft_toolkit.providers import get_provider

# Direct provider usage
provider = get_provider('tigl')
mesh = provider.create_aircraft('F15', detail_level='high')
print(f"Generated {mesh.num_vertices} vertices, {mesh.num_faces} faces")
```

### Quality Comparison
```python
# Compare providers
basic_provider = get_provider('basic')
tigl_provider = get_provider('tigl')

basic_mesh = basic_provider.create_aircraft('F15')
tigl_mesh = tigl_provider.create_aircraft('F15', detail_level='medium')

print(f"Basic: {basic_mesh.num_vertices} vertices")
print(f"TiGL: {tigl_mesh.num_vertices} vertices")
print(f"Improvement: {tigl_mesh.num_vertices / basic_mesh.num_vertices:.0f}x")
```

## 📁 Repository Structure

```
aircraft-dataset-generator/
├── aircraft_toolkit/           # Main package
│   ├── providers/             # Modular provider system
│   ├── core/                  # Dataset generation
│   └── config.py              # Configuration management
├── examples/                  # Usage examples
│   ├── basic_3d_generation.py
│   ├── tigl_test_generation.py
│   └── integration_comparison.py
├── tests/                     # Test suite
├── samples/                   # Generated sample datasets
├── TIGL_INTEGRATION.md        # Complete TiGL guide
├── INTEGRATION_PLAN.md        # Implementation details
└── README.md                  # This file
```

## 📚 Documentation

- **[TiGL Integration Guide](TIGL_INTEGRATION.md)**: Complete setup and usage guide
- **[Integration Plan](INTEGRATION_PLAN.md)**: Technical implementation details
- **[Architecture Documentation](docs/architecture.md)**: Framework design
- **Examples**: See `examples/` directory for working code samples

## 🔧 Migration Guide

### From Previous Versions
Your existing code continues to work unchanged:

```python
# OLD CODE - STILL WORKS
dataset = Dataset3D(aircraft_types=['F15'])
dataset.generate('output')

# NEW FEATURES - OPTIONAL
# Automatically uses TiGL if available for better quality
```

### Upgrading to TiGL
1. **Install TiGL**: `conda install -c dlr-sc tigl3`
2. **Run existing code**: Automatically uses TiGL
3. **Configure quality**: Set detail level in config
4. **Enjoy**: 100x+ better mesh quality

## 🧪 Testing

```bash
# Run test suite
python -m pytest tests/ -v

# Generate test datasets
python examples/tigl_test_generation.py

# Compare providers
python examples/integration_comparison.py
```

## 🤝 Contributing

We welcome contributions! The modular provider system makes it easy to add new 3D model backends:

1. Implement `ModelProvider` interface
2. Register provider in `__init__.py`
3. Add tests and documentation
4. Submit PR

Potential providers: CadQuery, Open3D, FreeCAD, OpenSCAD

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- **TiGL Team**: For the excellent CPACS-based aircraft geometry library
- **OpenCASCADE**: For the underlying CAD kernel
- **Aerospace Community**: For CPACS standards and aircraft data