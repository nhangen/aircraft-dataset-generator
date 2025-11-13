"""
Base classes for aircraft model providers.

This module defines the abstract interface that all model providers must implement,
ensuring consistency and modularity across different 3D model generation backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AircraftMesh:
    """
    Standardized aircraft mesh data structure.

    Attributes:
        vertices: Nx3 array of vertex coordinates
        faces: Mx3 array of triangle face indices
        normals: Optional Nx3 array of vertex normals
        metadata: Additional metadata (aircraft type, dimensions, etc.)
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    metadata: dict = None

    def __post_init__(self):
        """Validate mesh data after initialization."""
        if self.metadata is None:
            self.metadata = {}

        # Ensure correct shapes
        assert (
            self.vertices.ndim == 2 and self.vertices.shape[1] == 3
        ), f"Vertices must be Nx3, got {self.vertices.shape}"
        assert (
            self.faces.ndim == 2 and self.faces.shape[1] == 3
        ), f"Faces must be Mx3, got {self.faces.shape}"

        if self.normals is not None:
            assert (
                self.normals.shape == self.vertices.shape
            ), f"Normals shape {self.normals.shape} must match vertices {self.vertices.shape}"

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        """Number of faces in the mesh."""
        return len(self.faces)

    def compute_normals(self):
        """Compute vertex normals if not present."""
        if self.normals is not None:
            return

        # Initialize normals
        normals = np.zeros_like(self.vertices)

        # Compute face normals and accumulate to vertices
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            face_normal = np.cross(v1 - v0, v2 - v0)
            normals[face] += face_normal

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        self.normals = np.where(norms > 0, normals / norms, normals)

    def center_and_scale(self, target_size: float = 10.0):
        """
        Center mesh at origin and scale to target size.

        Args:
            target_size: Target bounding box diagonal size
        """
        # Center at origin
        center = self.vertices.mean(axis=0)
        self.vertices -= center

        # Scale to target size
        bbox_size = self.vertices.max(axis=0) - self.vertices.min(axis=0)
        current_size = np.linalg.norm(bbox_size)
        if current_size > 0:
            scale = target_size / current_size
            self.vertices *= scale


class ModelProvider(ABC):
    """
    Abstract base class for aircraft model providers.

    All model providers must inherit from this class and implement
    the required methods for generating aircraft meshes.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the model provider.

        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config or {}
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """Initialize provider-specific resources."""
        pass

    @abstractmethod
    def get_supported_aircraft(self) -> list[str]:
        """
        Get list of supported aircraft types.

        Returns:
            List of aircraft type identifiers (e.g., ['F15', 'B52', 'C130'])
        """
        pass

    @abstractmethod
    def create_aircraft(
        self, aircraft_type: str, detail_level: str = "medium", **kwargs
    ) -> AircraftMesh:
        """
        Create an aircraft mesh.

        Args:
            aircraft_type: Aircraft type identifier
            detail_level: Level of detail ('low', 'medium', 'high')
            **kwargs: Additional provider-specific parameters

        Returns:
            AircraftMesh object containing the generated mesh

        Raises:
            ValueError: If aircraft type is not supported
            RuntimeError: If mesh generation fails
        """
        pass

    def validate_aircraft_type(self, aircraft_type: str):
        """
        Validate that an aircraft type is supported.

        Args:
            aircraft_type: Aircraft type to validate

        Raises:
            ValueError: If aircraft type is not supported
        """
        supported = self.get_supported_aircraft()
        if aircraft_type not in supported:
            raise ValueError(
                f"Aircraft type '{aircraft_type}' not supported. "
                f"Supported types: {', '.join(supported)}"
            )

    def get_provider_info(self) -> dict:
        """
        Get information about this provider.

        Returns:
            Dictionary containing provider name, version, capabilities, etc.
        """
        return {
            "name": self.__class__.__name__,
            "supported_aircraft": self.get_supported_aircraft(),
            "capabilities": self._get_capabilities(),
        }

    def _get_capabilities(self) -> dict:
        """
        Get provider capabilities.

        Override in subclasses to specify additional capabilities.

        Returns:
            Dictionary of capability flags
        """
        return {
            "parametric": False,
            "texture_support": False,
            "animation_support": False,
            "detail_levels": ["low", "medium", "high"],
        }

    def cleanup(self):
        """
        Clean up provider resources.

        Override in subclasses if cleanup is needed.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
