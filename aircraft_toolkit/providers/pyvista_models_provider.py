"""
PyVista-based provider using real 3D models for high-quality aircraft rendering.

This provider uses actual 3D model files (STL/OBJ) or PyVista's built-in models
instead of trying to procedurally generate aircraft geometry.
"""

import logging
import os
from pathlib import Path
from typing import List

import numpy as np

from .base import AircraftMesh, ModelProvider

try:
    import pyvista as pv
    from pyvista import examples

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

logger = logging.getLogger(__name__)


class PyVistaModelsProvider(ModelProvider):
    # Provider using real 3D models with PyVista.

    def __init__(self, config=None):
        # Initialize PyVista models provider.
        super().__init__(config)
        self.models_cache = {}

    def _initialize(self):
        # Initialize provider-specific resources.
        if not PYVISTA_AVAILABLE:
            raise ImportError(
                "PyVista is not available. Install with: pip install pyvista"
            )

        # Set configuration with defaults
        self.models_dir = self.config.get("models_dir", "models/aircraft")
        self.use_builtin = self.config.get("use_builtin", True)
        self.scaling_factor = self.config.get("scaling_factor", 1.0)
        self.center_models = self.config.get("center_models", True)

        # Performance optimization: supported extensions in priority order
        self.supported_extensions = [".glb", ".obj", ".stl", ".ply"]

        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)

    def get_supported_aircraft(self) -> List[str]:
        # Get list of supported aircraft types.
        aircraft = []

        # Built-in models (we'll use the airplane for all types with variations)
        if self.use_builtin:
            aircraft.extend(["F15", "B52", "C130"])

        # Check for custom model files (optimized scanning)
        models_path = Path(self.models_dir)
        if models_path.exists():
            for ext in self.supported_extensions:
                pattern = f"*{ext}"
                for model_file in models_path.glob(pattern):
                    name = model_file.stem.upper()
                    if name not in aircraft:
                        aircraft.append(name)

        return aircraft

    def is_available(self) -> bool:
        # Check if PyVista is available.
        return PYVISTA_AVAILABLE

    def create_aircraft(
        self, aircraft_type: str, detail_level: str = "medium", **kwargs
    ) -> AircraftMesh:
        """
        Load or create aircraft model.

        Args:
            aircraft_type: Type of aircraft
            detail_level: Not used for pre-made models
            **kwargs: Additional parameters

        Returns:
            AircraftMesh object
        """
        aircraft_type = aircraft_type.upper()

        # Check cache first
        if aircraft_type in self.models_cache:
            mesh = self.models_cache[aircraft_type].copy()
        else:
            mesh = self._load_aircraft_model(aircraft_type)
            self.models_cache[aircraft_type] = mesh.copy()

        # Apply any transformations specific to this aircraft type
        mesh = self._apply_aircraft_specific_transforms(mesh, aircraft_type)

        return self._pyvista_to_aircraft_mesh(mesh, aircraft_type)

    def _load_aircraft_model(self, aircraft_type: str):
        # Load aircraft model from file or use built-in.
        # First try to load custom model file
        custom_model = self._try_load_custom_model(aircraft_type)
        if custom_model is not None:
            return custom_model

        # Use built-in airplane model with variations
        if self.use_builtin and aircraft_type in ["F15", "B52", "C130"]:
            return self._create_builtin_variant(aircraft_type)

        raise ValueError(f"No model available for aircraft type: {aircraft_type}")

    def _try_load_custom_model(self, aircraft_type: str):
        # Try to load a custom model file.
        models_path = Path(self.models_dir)

        # Check for model files with this name (priority order for performance)
        for ext in self.supported_extensions:
            model_file = models_path / f"{aircraft_type.lower()}{ext}"
            if model_file.exists():
                try:
                    data = pv.read(str(model_file))
                    # Handle MultiBlock data (common with GLB files)
                    if hasattr(data, "combine"):
                        mesh = data.combine()
                    else:
                        mesh = data

                    # Convert to PolyData if needed
                    if hasattr(mesh, "extract_surface"):
                        mesh = mesh.extract_surface()
                    elif PYVISTA_AVAILABLE and not isinstance(mesh, pv.PolyData):
                        mesh = mesh.cast_to_polydata()
                    logger.info(f"Loaded custom model: {model_file}")
                    return mesh
                except Exception as e:
                    logger.error(f"Failed to load {model_file}: {e}")

        return None

    def _create_builtin_variant(self, aircraft_type: str):
        # Create a variant of the built-in airplane model.
        # Load the built-in airplane
        airplane = examples.load_airplane()

        # Apply transformations to create different aircraft "types"
        if aircraft_type == "F15":
            # Fighter jet - keep as is, maybe scale down slightly
            airplane = airplane.scale([0.8, 0.8, 0.8])

        elif aircraft_type == "B52":
            # Bomber - stretch the fuselage and wings
            airplane = airplane.scale([1.5, 1.8, 0.9])

        elif aircraft_type == "C130":
            # Transport - make it bulkier
            airplane = airplane.scale([1.2, 1.1, 1.3])

        return airplane

    def _apply_aircraft_specific_transforms(self, mesh, aircraft_type: str):
        # Apply any specific transformations for the aircraft type.
        # Rotate to standard orientation (nose pointing in +X direction)
        # PyVista's airplane is oriented differently, so we need to rotate it

        # First, center the model at origin
        if self.center_models:
            center = np.array(mesh.center)
            mesh = mesh.translate(-center, inplace=False)

        # Rotate to align with our coordinate system
        # The built-in airplane needs to be rotated to face the right direction
        mesh = mesh.rotate_x(90, inplace=False)  # Rotate to level flight
        mesh = mesh.rotate_z(180, inplace=False)  # Point nose forward

        # Apply scaling
        if self.scaling_factor != 1.0:
            mesh = mesh.scale(self.scaling_factor, inplace=False)

        return mesh

    def _pyvista_to_aircraft_mesh(self, pv_mesh, aircraft_type: str) -> AircraftMesh:
        # Convert PyVista mesh to AircraftMesh format.
        # Get vertices and faces
        vertices = np.array(pv_mesh.points)

        # Extract faces - handle different mesh types
        faces_raw = pv_mesh.faces
        faces = []

        if faces_raw is not None and len(faces_raw) > 0:
            i = 0
            while i < len(faces_raw):
                n_verts = faces_raw[i]
                if n_verts == 3:  # Triangle
                    faces.append(faces_raw[i + 1 : i + 4])
                elif n_verts == 4:  # Quad - split into triangles
                    quad = faces_raw[i + 1 : i + 5]
                    faces.append([quad[0], quad[1], quad[2]])
                    faces.append([quad[0], quad[2], quad[3]])
                i += n_verts + 1
            faces = np.array(faces)
        else:
            # Try to triangulate if no faces
            pv_mesh = pv_mesh.triangulate()
            faces_raw = pv_mesh.faces
            if faces_raw is not None:
                i = 0
                while i < len(faces_raw):
                    n_verts = faces_raw[i]
                    if n_verts == 3:
                        faces.append(faces_raw[i + 1 : i + 4])
                    i += n_verts + 1
                faces = np.array(faces) if faces else np.array([[0, 1, 2]])  # Fallback

        # Compute normals
        pv_mesh = pv_mesh.compute_normals(inplace=False)
        normals = (
            np.array(pv_mesh.point_data["Normals"])
            if "Normals" in pv_mesh.point_data
            else None
        )

        logger.info(
            f"Loaded {aircraft_type}: {len(vertices)} vertices, {len(faces)} faces"
        )

        return AircraftMesh(
            vertices=vertices,
            faces=faces,
            normals=normals,
            metadata={
                "provider": "pyvista_models",
                "aircraft_type": aircraft_type,
                "vertex_count": len(vertices),
                "face_count": len(faces),
                "source": (
                    "custom"
                    if aircraft_type in self.models_cache and len(vertices) > 2000
                    else "builtin"
                ),
            },
        )

    def download_sample_models(self):
        # Download sample aircraft models from online sources.
        logger.info("Downloading sample aircraft models...")

        # This would download actual aircraft models from repositories
        # For now, we'll just create a directory structure
        models_path = Path(self.models_dir)
        models_path.mkdir(parents=True, exist_ok=True)

        # Create a README explaining how to add models
        readme_path = models_path / "README.md"
        readme_content = """# Aircraft 3D Models

Place your aircraft 3D models in this directory.

## Supported Formats
- STL (.stl)
- OBJ (.obj)
- PLY (.ply)

## Naming Convention
Name your files after the aircraft type:
- f15.stl - F-15 Eagle
- b52.stl - B-52 Stratofortress
- c130.stl - C-130 Hercules

## Where to Find Models
1. **Free 3D Model Sites:**
   - Thingiverse: https://www.thingiverse.com/search?q=aircraft
   - TurboSquid (free section): https://www.turbosquid.com/Search/3D-Models/free/aircraft
   - Free3D: https://free3d.com/3d-models/aircraft

2. **Example Search Terms:**
   - "F-15 Eagle STL"
   - "B-52 bomber 3D model"
   - "C-130 Hercules STL free"

## Usage
Once you've added model files here, they will automatically be detected and used by the PyVistaModelsProvider.
"""
        readme_path.write_text(readme_content)
        logger.info(f"Created models directory at {models_path}")
        logger.info("Please add your STL/OBJ aircraft models to this directory")
