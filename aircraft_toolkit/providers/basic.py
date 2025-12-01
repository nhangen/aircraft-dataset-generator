"""
Basic model provider using hand-coded meshes.

This is the original implementation, preserved for backward compatibility
and as a fallback when advanced libraries are not available.
"""

import numpy as np

from .base import AircraftMesh, ModelProvider


class BasicProvider(ModelProvider):
    """
    Basic aircraft model provider using hand-coded vertex meshes.

    This provider creates simple, low-polygon aircraft models using
    manually defined vertices and faces. While not realistic, these
    models are fast to generate and have no external dependencies.
    """

    def _initialize(self):
        """Initialize the basic provider."""
        self.aircraft_definitions = {
            "F15": self._create_f15,
            "B52": self._create_b52,
            "C130": self._create_c130,
        }

    def get_supported_aircraft(self) -> list[str]:
        """Get list of supported aircraft types."""
        return list(self.aircraft_definitions.keys())

    def create_aircraft(
        self, aircraft_type: str, detail_level: str = "medium", **kwargs
    ) -> AircraftMesh:
        """
        Create a basic aircraft mesh.

        Args:
            aircraft_type: Aircraft type ('F15', 'B52', or 'C130')
            detail_level: Ignored for basic provider
            **kwargs: Additional parameters (ignored)

        Returns:
            AircraftMesh object
        """
        self.validate_aircraft_type(aircraft_type)

        # Get creation function
        create_func = self.aircraft_definitions[aircraft_type]
        vertices, faces = create_func()

        # Create mesh object
        mesh = AircraftMesh(
            vertices=vertices,
            faces=faces,
            metadata={
                "aircraft_type": aircraft_type,
                "provider": "basic",
                "detail_level": "low",  # Always low for basic provider
            },
        )

        # Compute normals
        mesh.compute_normals()

        return mesh

    def _create_f15(self) -> tuple:
        """Create simple F-15 fighter mesh."""
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
        """Create simple B-52 bomber mesh."""
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
        """Create simple C-130 transport mesh."""
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
        """Get basic provider capabilities."""
        return {
            "parametric": False,
            "texture_support": False,
            "animation_support": False,
            "detail_levels": ["low"],  # Only low detail available
            "max_vertices": 20,  # Approximate maximum
            "external_dependencies": False,
        }
