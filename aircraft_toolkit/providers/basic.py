"""Basic model provider using hand-coded meshes.

This module provides a fallback aircraft model provider that creates simple,
low-polygon 3D models using manually defined vertices and faces. This implementation
has no external dependencies and serves as both a backward compatibility layer
and a lightweight alternative when advanced 3D libraries are unavailable.

The basic provider creates wireframe-style aircraft models with minimal geometric
detail (10-15 vertices per aircraft) suitable for silhouette generation, basic
visualization, or resource-constrained environments.

Mesh Structure:
    Each aircraft is defined by:
    - Vertices: 3D points (x, y, z) defining key structural points
    - Faces: Triangle indices referencing vertex positions
    - Normals: Computed per-face for basic shading

Coordinate System:
    - X-axis: Forward/aft (positive = forward/nose)
    - Y-axis: Left/right (positive = right wing)
    - Z-axis: Up/down (positive = up/top)
    - Origin: Aircraft center of mass
"""

import numpy as np

from .base import AircraftMesh, ModelProvider


class BasicProvider(ModelProvider):
    """Basic aircraft model provider using hand-coded vertex meshes.

    This provider creates simple, low-polygon aircraft models using
    manually defined vertices and faces. While not photorealistic, these
    models are extremely fast to generate, have no external dependencies,
    and provide recognizable aircraft silhouettes.

    Performance:
        - Mesh generation: < 1ms per aircraft
        - Memory footprint: ~1KB per mesh
        - No external library dependencies

    Limitations:
        - Low geometric detail (10-15 vertices)
        - No texture support
        - Fixed detail level (always "low")
        - Simplified wing and tail geometries

    Use Cases:
        - Fallback when PyVista/trimesh unavailable
        - Resource-constrained environments
        - Silhouette-only generation
        - Rapid prototyping and testing
    """

    def _initialize(self):
        """Initialize the basic provider.

        Sets up the aircraft definition registry mapping aircraft type
        identifiers to their respective mesh creation functions.
        """
        self.aircraft_definitions = {
            "F15": self._create_f15,
            "B52": self._create_b52,
            "C130": self._create_c130,
        }

    def get_supported_aircraft(self) -> list[str]:
        """Get list of supported aircraft types.

        Returns:
            List of aircraft type identifiers that this provider can generate.
            Currently supports: ['F15', 'B52', 'C130'].
        """
        return list(self.aircraft_definitions.keys())

    def create_aircraft(
        self, aircraft_type: str, detail_level: str = "medium", **kwargs
    ) -> AircraftMesh:
        """Create a basic aircraft mesh.

        Generates a low-polygon 3D mesh for the specified aircraft type using
        hand-coded vertex and face definitions. The mesh is suitable for
        silhouette generation and basic visualization.

        Args:
            aircraft_type: Aircraft type identifier. Must be one of:
                - 'F15': F-15 Eagle fighter (14 vertices, 19 faces)
                - 'B52': B-52 Stratofortress bomber (13 vertices, 19 faces)
                - 'C130': C-130 Hercules transport (15 vertices, 22 faces)
            detail_level: Detail level specification. This parameter is ignored
                by the basic provider as it only supports low-detail models.
                Provided for API compatibility with other providers.
            **kwargs: Additional provider-specific parameters. All kwargs are
                ignored by the basic provider.

        Returns:
            AircraftMesh object containing:
                - vertices: NumPy array of 3D vertex coordinates (N x 3)
                - faces: NumPy array of triangle face indices (M x 3)
                - normals: Computed per-face normal vectors
                - metadata: Dict with aircraft_type, provider name, and detail level

        Raises:
            ValueError: If aircraft_type is not supported.

        Example:
            >>> provider = BasicProvider()
            >>> mesh = provider.create_aircraft('F15')
            >>> print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
            Vertices: 14, Faces: 19

        Note:
            The basic provider always returns low-detail meshes regardless of
            the detail_level parameter. For higher detail models, use
            PyVistaModelsProvider instead.
        """
        self.validate_aircraft_type(aircraft_type)

        # Get the appropriate mesh creation function for this aircraft
        create_func = self.aircraft_definitions[aircraft_type]
        vertices, faces = create_func()

        # Create mesh object with computed geometry
        mesh = AircraftMesh(
            vertices=vertices,
            faces=faces,
            metadata={
                "aircraft_type": aircraft_type,
                "provider": "basic",
                "detail_level": "low",  # Always low for basic provider
            },
        )

        # Compute face normals for basic shading
        mesh.compute_normals()

        return mesh

    def _create_f15(self) -> tuple:
        """Create simplified F-15 Eagle fighter mesh.

        Generates a low-polygon representation of an F-15 twin-engine tactical
        fighter aircraft. The mesh captures key identifying features: twin
        vertical tails, swept delta wings, and streamlined fuselage.

        Mesh Geometry:
            - Vertices: 14 points defining fuselage, wings, and tails
            - Faces: 19 triangles forming the aircraft surface
            - Features: Nose, cockpit, wings, twin vertical stabilizers

        Vertex Groups:
            - 0-4: Main fuselage centerline (nose to tail)
            - 5-7: Fuselage bottom profile
            - 8-11: Wing tips and trailing edges
            - 12-13: Twin vertical tails (characteristic F-15 feature)

        Returns:
            Tuple of (vertices, faces) where:
                - vertices: (14, 3) NumPy array of 3D coordinates
                - faces: (19, 3) NumPy array of triangle vertex indices

        Note:
            This is a simplified wireframe model. For photorealistic models
            with 50K+ vertices, use PyVistaModelsProvider with real 3D assets.
        """
        vertices = np.array(
            [
                # Main fuselage
                [4.0, 0.0, 0.0],  # 0: nose
                [1.5, 0.0, 0.5],  # 1: cockpit top
                [0.0, 0.0, 0.3],  # 2: wing root center
                [-2.0, 0.0, 0.3],  # 3: aft fuselage
                [-3.5, 0.0, 0.0],  # 4: tail
                # Fuselage bottom
                [1.5, 0.0, -0.5],  # 5: cockpit bottom
                [0.0, 0.0, -0.3],  # 6: wing root bottom
                [-2.0, 0.0, -0.3],  # 7: aft bottom
                # Wings
                [0.5, -3.0, 0.0],  # 8: left wing tip
                [0.5, 3.0, 0.0],  # 9: right wing tip
                [-0.8, -2.0, 0.0],  # 10: left wing trailing
                [-0.8, 2.0, 0.0],  # 11: right wing trailing
                # Twin tails
                [-2.5, -0.6, 1.2],  # 12: left vertical tail
                [-2.5, 0.6, 1.2],  # 13: right vertical tail
            ]
        )

        faces = np.array(
            [
                # Fuselage top
                [0, 1, 2],
                [2, 3, 4],
                # Fuselage bottom
                [0, 5, 6],
                [6, 7, 4],
                # Fuselage sides
                [0, 1, 5],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 4, 7],
                # Wings
                [2, 8, 10],
                [2, 10, 6],
                [2, 9, 11],
                [2, 11, 6],
                [8, 9, 2],
                [10, 11, 6],
                # Twin tails
                [3, 12, 4],
                [3, 13, 12],
                [4, 12, 13],
            ]
        )

        return vertices, faces

    def _create_b52(self) -> tuple:
        """Create simplified B-52 Stratofortress bomber mesh.

        Generates a low-polygon representation of a B-52 long-range strategic
        bomber. The mesh emphasizes the B-52's distinctive features: extreme
        wingspan with swept-back wings and elongated fuselage.

        Mesh Geometry:
            - Vertices: 13 points defining fuselage, wings, and tail
            - Faces: 19 triangles forming the aircraft surface
            - Features: Long nose, swept wings (5.0 unit span), vertical tail

        Vertex Groups:
            - 0-4: Main fuselage centerline (extended length for bomber)
            - 5-7: Fuselage bottom profile
            - 8-11: Wide swept wing geometry (bomber characteristic)
            - 12: Single vertical tail stabilizer

        Returns:
            Tuple of (vertices, faces) where:
                - vertices: (13, 3) NumPy array of 3D coordinates
                - faces: (19, 3) NumPy array of triangle vertex indices

        Note:
            The swept wing geometry (negative X for trailing edge) accurately
            represents the B-52's distinctive wing design. Wing span is
            proportionally larger than F-15 to reflect bomber characteristics.
        """
        vertices = np.array(
            [
                # Main fuselage
                [6.0, 0.0, 0.0],  # 0: nose
                [3.0, 0.0, 0.4],  # 1: forward fuselage
                [0.0, 0.0, 0.3],  # 2: wing root
                [-3.0, 0.0, 0.3],  # 3: aft fuselage
                [-5.5, 0.0, 0.0],  # 4: tail
                # Fuselage bottom
                [3.0, 0.0, -0.4],  # 5: forward bottom
                [0.0, 0.0, -0.3],  # 6: wing root bottom
                [-3.0, 0.0, -0.3],  # 7: aft bottom
                # Long swept wings
                [1.0, -5.0, 0.1],  # 8: left wing tip
                [1.0, 5.0, 0.1],  # 9: right wing tip
                [-1.5, -4.0, 0.0],  # 10: left wing trailing
                [-1.5, 4.0, 0.0],  # 11: right wing trailing
                # Vertical tail
                [-4.0, 0.0, 1.5],  # 12: tail top
            ]
        )

        faces = np.array(
            [
                # Fuselage top
                [0, 1, 2],
                [2, 3, 4],
                # Fuselage bottom
                [0, 5, 6],
                [6, 7, 4],
                # Fuselage sides
                [0, 1, 5],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 4, 7],
                # Wings (swept)
                [2, 8, 10],
                [2, 10, 6],
                [2, 9, 11],
                [2, 11, 6],
                [8, 9, 2],
                [10, 11, 6],
                # Vertical tail
                [3, 12, 4],
                [4, 12, 7],
                [3, 7, 12],
            ]
        )

        return vertices, faces

    def _create_c130(self) -> tuple:
        """Create simplified C-130 Hercules transport mesh.

        Generates a low-polygon representation of a C-130 tactical military
        transport aircraft. The mesh captures the C-130's distinctive features:
        high-mounted wings, large cargo fuselage, and T-tail configuration.

        Mesh Geometry:
            - Vertices: 15 points defining fuselage, wings, and T-tail
            - Faces: 22 triangles forming the aircraft surface
            - Features: Cargo belly, high wings, T-tail (horizontal on vertical)

        Vertex Groups:
            - 0-4: Main fuselage centerline
            - 5-7: Large cargo belly (lower than fighter/bomber)
            - 8-11: High-mounted wings (Z=1.2, above fuselage)
            - 12-14: T-tail configuration (characteristic transport feature)

        Returns:
            Tuple of (vertices, faces) where:
                - vertices: (15, 3) NumPy array of 3D coordinates
                - faces: (22, 3) NumPy array of triangle vertex indices

        Note:
            The high wing position (Z=1.2) and large cargo belly (Z=-1.0)
            are characteristic of military transport aircraft, providing
            ground clearance for cargo operations. The T-tail places the
            horizontal stabilizer above the fuselage wake.
        """
        vertices = np.array(
            [
                # Main fuselage
                [4.5, 0.0, -0.2],  # 0: nose
                [2.0, 0.0, 0.2],  # 1: cockpit
                [0.0, 0.0, 0.3],  # 2: cargo area
                [-2.5, 0.0, 0.2],  # 3: aft fuselage
                [-4.0, 0.0, 0.0],  # 4: tail
                # Large cargo belly
                [2.0, 0.0, -0.8],  # 5: cockpit bottom
                [0.0, 0.0, -1.0],  # 6: cargo belly
                [-2.5, 0.0, -0.8],  # 7: aft belly
                # High wings (above fuselage)
                [1.0, -4.5, 1.2],  # 8: left wing tip
                [1.0, 4.5, 1.2],  # 9: right wing tip
                [-1.0, -3.5, 0.9],  # 10: left wing trailing
                [-1.0, 3.5, 0.9],  # 11: right wing trailing
                # T-tail
                [-3.5, 0.0, 2.0],  # 12: vertical tail top
                [-3.5, -1.8, 2.0],  # 13: left horizontal tail
                [-3.5, 1.8, 2.0],  # 14: right horizontal tail
            ]
        )

        faces = np.array(
            [
                # Fuselage top
                [0, 1, 2],
                [2, 3, 4],
                # Large cargo belly
                [0, 5, 6],
                [6, 7, 4],
                # Fuselage sides
                [0, 1, 5],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 4, 7],
                # High wings
                [2, 8, 10],
                [2, 9, 11],
                [8, 9, 2],
                [10, 11, 6],
                # Wing undersides
                [6, 10, 8],
                [6, 11, 9],
                [6, 8, 9],
                [6, 9, 11],
                # T-tail vertical
                [3, 12, 4],
                [4, 12, 7],
                [3, 7, 12],
                # T-tail horizontal
                [12, 13, 14],
            ]
        )

        return vertices, faces

    def _get_capabilities(self) -> dict:
        """Get basic provider capabilities and limitations.

        Returns a dictionary describing what features and functionality this
        provider supports. Used by the provider registry for capability
        discovery and provider selection.

        Returns:
            Dictionary with capability flags and limits:
                - parametric: Whether models can be parameterized (False)
                - texture_support: Whether textures are supported (False)
                - animation_support: Whether animation is supported (False)
                - detail_levels: List of available detail levels (['low'])
                - max_vertices: Approximate maximum vertex count per model (20)
                - external_dependencies: Whether external libs required (False)

        Note:
            The basic provider is intentionally limited to ensure it works
            in all environments without external dependencies. For advanced
            features, use PyVistaModelsProvider or HeadlessProvider.
        """
        return {
            "parametric": False,  # No parametric model support
            "texture_support": False,  # No texture/material support
            "animation_support": False,  # No animation/rigging support
            "detail_levels": ["low"],  # Only low detail available
            "max_vertices": 20,  # Approximate maximum vertices per model
            "external_dependencies": False,  # Pure Python, no external libs
        }
