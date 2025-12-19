"""
Headless 3D provider that renders without OpenGL/GPU dependencies.
Uses pure Python rasterization for Docker environments.
"""

from typing import List

import numpy as np
from PIL import Image, ImageDraw

from .base import AircraftMesh, ModelProvider


class HeadlessProvider(ModelProvider):
    """Pure computational 3D provider for headless environments."""

    def __init__(self, config=None):
        """Initialize headless provider."""
        super().__init__(config)
        self.name = "headless"

    def _initialize(self):
        """Initialize provider-specific resources."""
        pass

    def get_supported_aircraft(self) -> List[str]:
        """Return list of supported aircraft types."""
        return ['F15', 'B52', 'C130']

    def create_aircraft(self, aircraft_type: str, detail_level: str = 'medium', **kwargs) -> AircraftMesh:
        """Create aircraft mesh for headless rendering."""
        aircraft_type = aircraft_type.upper()

        if aircraft_type == 'F15':
            return self._create_f15_mesh()
        if aircraft_type == 'B52':
            return self._create_b52_mesh()
        if aircraft_type == 'C130':
            return self._create_c130_mesh()
        raise ValueError(f"Unsupported aircraft type: {aircraft_type}")

    def _create_f15_mesh(self) -> AircraftMesh:
        """Create F-15 mesh optimized for headless rendering."""
        # F-15 Eagle - Twin-engine fighter
        vertices = np.array([
            # Fuselage (nose to tail)
            [0.0, 0.0, 0.0],    # Nose
            [8.0, 0.0, 0.0],    # Center
            [16.0, 0.0, 1.0],   # Tail

            # Wings (swept back)
            [6.0, -8.0, 0.0],   # Left wing tip
            [6.0, 8.0, 0.0],    # Right wing tip
            [10.0, -4.0, 0.0],  # Left wing root
            [10.0, 4.0, 0.0],   # Right wing root

            # Vertical stabilizers
            [14.0, -2.0, 4.0],  # Left tail
            [14.0, 2.0, 4.0],   # Right tail
            [14.0, 0.0, 6.0],   # Top tail

            # Engine intakes
            [5.0, -2.0, -1.0],  # Left intake
            [5.0, 2.0, -1.0],   # Right intake
        ])

        faces = np.array([
            # Fuselage triangles
            [0, 1, 4], [0, 1, 3], [1, 2, 7], [1, 2, 8],
            # Wing triangles
            [3, 5, 1], [4, 6, 1], [5, 7, 2], [6, 8, 2],
            # Tail triangles
            [7, 8, 9], [2, 7, 9], [2, 8, 9],
            # Engine triangles
            [10, 11, 1], [0, 10, 1], [0, 11, 1],
        ])

        return AircraftMesh(
            vertices=vertices,
            faces=faces,
            metadata={'length': 16.0, 'wingspan': 16.0, 'height': 6.0}
        )

    def _create_b52_mesh(self) -> AircraftMesh:
        """Create B-52 mesh optimized for headless rendering."""
        # B-52 Stratofortress - Strategic bomber with long wings
        vertices = np.array([
            # Fuselage
            [0.0, 0.0, 0.0],    # Nose
            [10.0, 0.0, 0.0],   # Center
            [24.0, 0.0, 2.0],   # Tail

            # Long straight wings
            [8.0, -20.0, -1.0], # Left wing tip
            [8.0, 20.0, -1.0],  # Right wing tip
            [8.0, -8.0, 0.0],   # Left wing root
            [8.0, 8.0, 0.0],    # Right wing root

            # Vertical tail
            [22.0, 0.0, 8.0],   # Top tail
            [20.0, -2.0, 2.0],  # Left tail
            [20.0, 2.0, 2.0],   # Right tail

            # Engine pods (simplified)
            [6.0, -15.0, -2.0], # Left outboard engine
            [6.0, -5.0, -2.0],  # Left inboard engine
            [6.0, 5.0, -2.0],   # Right inboard engine
            [6.0, 15.0, -2.0],  # Right outboard engine
        ])

        faces = np.array([
            # Fuselage
            [0, 1, 5], [0, 1, 6], [1, 2, 8], [1, 2, 9],
            # Wings
            [3, 5, 1], [4, 6, 1], [3, 10, 5], [4, 13, 6],
            # Tail
            [2, 7, 8], [2, 7, 9], [8, 9, 7],
            # Engines
            [10, 11, 3], [12, 13, 4], [5, 11, 1], [6, 12, 1],
        ])

        return AircraftMesh(
            vertices=vertices,
            faces=faces,
            metadata={'length': 24.0, 'wingspan': 40.0, 'height': 8.0}
        )

    def _create_c130_mesh(self) -> AircraftMesh:
        """Create C-130 mesh optimized for headless rendering."""
        # C-130 Hercules - Transport aircraft with high wings
        vertices = np.array([
            # Fuselage
            [0.0, 0.0, 0.0],    # Nose
            [8.0, 0.0, 0.0],    # Center
            [20.0, 0.0, 1.0],   # Tail

            # High-mounted wings
            [8.0, -16.0, 4.0],  # Left wing tip
            [8.0, 16.0, 4.0],   # Right wing tip
            [8.0, -6.0, 3.0],   # Left wing root
            [8.0, 6.0, 3.0],    # Right wing root

            # T-tail
            [18.0, 0.0, 8.0],   # Top of tail
            [16.0, -4.0, 8.0],  # Left horizontal stabilizer
            [16.0, 4.0, 8.0],   # Right horizontal stabilizer

            # Propeller engines
            [6.0, -12.0, 3.0],  # Left outboard prop
            [6.0, -4.0, 3.0],   # Left inboard prop
            [6.0, 4.0, 3.0],    # Right inboard prop
            [6.0, 12.0, 3.0],   # Right outboard prop

            # Landing gear (simplified)
            [4.0, 0.0, -2.0],   # Nose gear
            [10.0, -2.0, -1.0], # Left main gear
            [10.0, 2.0, -1.0],  # Right main gear
        ])

        faces = np.array([
            # Fuselage
            [0, 1, 5], [0, 1, 6], [1, 2, 5], [1, 2, 6],
            # Wings
            [3, 5, 1], [4, 6, 1], [3, 10, 5], [4, 13, 6],
            # T-tail
            [2, 7, 8], [2, 7, 9], [7, 8, 9],
            # Props
            [10, 11, 5], [12, 13, 6], [1, 11, 5], [1, 12, 6],
            # Landing gear
            [0, 14, 1], [15, 16, 1],
        ])

        return AircraftMesh(
            vertices=vertices,
            faces=faces,
            metadata={'length': 20.0, 'wingspan': 32.0, 'height': 8.0}
        )

    def render_view(self, mesh: AircraftMesh, **kwargs) -> Image.Image:
        """Render view using pure Python rasterization."""
        image_size = kwargs.get('image_size', (512, 512))
        aircraft_pose = kwargs.get('aircraft_pose', {})
        camera = kwargs.get('camera')

        # Create blank image
        image = Image.new('RGB', image_size, color=(135, 206, 235))  # Sky blue
        draw = ImageDraw.Draw(image)

        # Apply aircraft transformations
        transformed_vertices = self._apply_transformations(mesh.vertices, aircraft_pose, camera)

        # Project to 2D
        projected_vertices = []
        for vertex in transformed_vertices:
            # Simple perspective projection
            if vertex[2] > 0.1:  # Avoid division by zero
                focal_length = image_size[0] * 0.8  # More appropriate focal length
                x_2d = int((vertex[0] * focal_length) / vertex[2] + image_size[0] / 2)
                y_2d = int(-(vertex[1] * focal_length) / vertex[2] + image_size[1] / 2)
                projected_vertices.append((x_2d, y_2d))
            else:
                projected_vertices.append((image_size[0]//2, image_size[1]//2))

        # Draw filled polygons
        for face in mesh.faces:
            if len(face) >= 3:
                face_points = [projected_vertices[i] for i in face if i < len(projected_vertices)]
                if len(face_points) >= 3:
                    try:
                        draw.polygon(face_points, fill=(100, 100, 100), outline=(50, 50, 50))
                    except Exception:
                        pass  # Skip invalid polygons

        return image

    def _apply_transformations(self, vertices, aircraft_pose, camera):
        """Apply aircraft pose and camera transformations."""
        import math

        # Apply aircraft pose transformation
        if aircraft_pose:
            rotation = aircraft_pose.get('rotation', {'pitch': 0, 'yaw': 0, 'roll': 0})
            translation = aircraft_pose.get('position', [0, 0, 0])

            # Convert rotations to transformation matrix
            pitch = math.radians(rotation['pitch'])
            yaw = math.radians(rotation['yaw'])
            roll = math.radians(rotation['roll'])

            # Rotation matrices
            R_x = np.array([
                [1, 0, 0],
                [0, math.cos(pitch), -math.sin(pitch)],
                [0, math.sin(pitch), math.cos(pitch)]
            ])
            R_y = np.array([
                [math.cos(yaw), 0, math.sin(yaw)],
                [0, 1, 0],
                [-math.sin(yaw), 0, math.cos(yaw)]
            ])
            R_z = np.array([
                [math.cos(roll), -math.sin(roll), 0],
                [math.sin(roll), math.cos(roll), 0],
                [0, 0, 1]
            ])

            # Combined rotation matrix
            R = R_z @ R_y @ R_x

            # Transform vertices
            transformed_vertices = vertices @ R.T
            transformed_vertices += np.array(translation)
        else:
            transformed_vertices = vertices.copy()

        # Apply camera transformation
        if camera:
            # Transform to camera space
            world_vertices = np.column_stack([transformed_vertices, np.ones(len(transformed_vertices))])
            camera_vertices = world_vertices @ camera.view_matrix.T
            transformed_vertices = camera_vertices[:, :3]

        return transformed_vertices
