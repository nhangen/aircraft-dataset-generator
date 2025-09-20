"""3D Aircraft Dataset Generation with Multi-View Rendering"""

import os
import json
import math
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageDraw
from tqdm import tqdm

# Import new provider system
from ..providers import get_provider
from ..config import get_config

logger = logging.getLogger(__name__)


class Camera:
    """3D camera for multi-view rendering"""

    def __init__(self, position: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 1, 0])):
        self.position = position
        self.target = target
        self.up = up
        self.view_matrix = self._compute_view_matrix()

    def _compute_view_matrix(self) -> np.ndarray:
        """Compute view matrix for camera (right-handed coordinate system)"""
        # Forward vector (from camera to target)
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)

        # Right vector
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)

        # Up vector (recalculate to ensure orthogonality)
        up = np.cross(right, forward)

        # View matrix (transforms world coordinates to camera coordinates)
        view_matrix = np.array([
            [right[0], right[1], right[2], -np.dot(right, self.position)],
            [up[0], up[1], up[2], -np.dot(up, self.position)],
            [forward[0], forward[1], forward[2], -np.dot(forward, self.position)],  # Positive Z forward
            [0, 0, 0, 1]
        ])
        return view_matrix


# Legacy aircraft classes removed - now using provider system
# See aircraft_toolkit.providers.basic for backward compatibility


class Dataset3D:
    """Generate 3D multi-view aircraft datasets with proper rendering"""

    def __init__(self,
                 aircraft_types: List[str],
                 num_scenes: int,
                 views_per_scene: int = 8,
                 camera_distance: Tuple[float, float] = (8, 12),
                 camera_height_range: Tuple[float, float] = (-5, 10),
                 include_depth_maps: bool = True,
                 include_surface_normals: bool = False,
                 image_size: Tuple[int, int] = (512, 512)):
        """
        Initialize 3D dataset generator

        Args:
            aircraft_types: List of aircraft type names ['F15', 'B52', 'C130']
            num_scenes: Number of aircraft scenes to generate
            views_per_scene: Number of camera views per scene
            camera_distance: Range of camera distances from aircraft
            camera_height_range: Range of camera heights
            include_depth_maps: Whether to generate depth maps
            include_surface_normals: Whether to generate surface normals
            image_size: Output image size (width, height)
        """
        self.aircraft_types = aircraft_types
        self.num_scenes = num_scenes
        self.views_per_scene = views_per_scene
        self.camera_distance = camera_distance
        self.camera_height_range = camera_height_range
        self.include_depth_maps = include_depth_maps
        self.include_surface_normals = include_surface_normals
        self.image_size = image_size

        # Initialize configuration and providers
        self.config = get_config()
        self.provider_name = self.config.get_preferred_provider()
        self.model_provider = get_provider(self.provider_name, self.config.providers.get(self.provider_name, {}).config)

        # Initialize 3D aircraft models using provider system
        self.aircraft_models = self._load_aircraft_models()

        logger.info(f"3D Dataset Generator initialized with {self.provider_name} provider")
        logger.info(f"Generating {num_scenes} scenes × {views_per_scene} views = {num_scenes * views_per_scene} total images")
        print(f"🛩️  3D Dataset Generator initialized with {self.provider_name} provider")
        print(f"📊 {num_scenes} scenes × {views_per_scene} views = {num_scenes * views_per_scene} total images")

    def _load_aircraft_models(self) -> Dict:
        """Load 3D aircraft model definitions using provider system"""
        models = {}

        # Validate aircraft types with provider
        supported_aircraft = self.model_provider.get_supported_aircraft()

        for aircraft_type in self.aircraft_types:
            aircraft_key = aircraft_type.upper()
            if aircraft_key not in supported_aircraft:
                logger.warning(f"Aircraft type {aircraft_key} not supported by {self.provider_name} provider")
                continue

            try:
                # Generate aircraft mesh using provider
                mesh = self.model_provider.create_aircraft(
                    aircraft_key,
                    detail_level=self.config.aircraft.detail_level
                )
                models[aircraft_key] = mesh
                logger.info(f"Loaded {aircraft_key}: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            except Exception as e:
                logger.error(f"Failed to load {aircraft_key}: {e}")
                continue

        if not models:
            raise RuntimeError(f"No aircraft models could be loaded with {self.provider_name} provider")

        return models

    def generate(self,
                 output_dir: str,
                 split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                 annotation_format: str = 'custom_3d',
                 num_workers: int = 1) -> Dict:
        """
        Generate 3D multi-view aircraft dataset

        Args:
            output_dir: Directory to save dataset
            split_ratios: Train/val/test split ratios
            annotation_format: Annotation format ('custom_3d')
            num_workers: Number of parallel workers

        Returns:
            Dataset generation results
        """
        print(f"🛩️  Generating 3D Multi-View Aircraft Dataset")
        print(f"📊 {self.num_scenes} scenes, {len(self.aircraft_types)} aircraft types")

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
            if self.include_depth_maps:
                os.makedirs(os.path.join(output_dir, split, 'depth'), exist_ok=True)

        # Calculate split sizes
        train_size = int(self.num_scenes * split_ratios[0])
        val_size = int(self.num_scenes * split_ratios[1])
        test_size = self.num_scenes - train_size - val_size

        # Generate datasets
        train_annotations = self._generate_split('train', train_size, output_dir)
        val_annotations = self._generate_split('val', val_size, output_dir)
        test_annotations = self._generate_split('test', test_size, output_dir)

        # Save annotations
        self._save_annotations(train_annotations, output_dir, 'train')
        self._save_annotations(val_annotations, output_dir, 'val')
        self._save_annotations(test_annotations, output_dir, 'test')

        total_images = self.num_scenes * self.views_per_scene

        return {
            'total_scenes': self.num_scenes,
            'views_per_scene': self.views_per_scene,
            'total_images': total_images,
            'train_scenes': train_size,
            'val_scenes': val_size,
            'test_scenes': test_size,
            'aircraft_types': self.aircraft_types,
            'output_dir': output_dir
        }

    def _generate_split(self, split_name: str, num_scenes: int, output_dir: str) -> List[Dict]:
        """Generate scenes for a specific split"""
        annotations = []

        for scene_idx in tqdm(range(num_scenes), desc=f"Generating {split_name}"):
            # Random aircraft type and pose
            available_types = list(self.aircraft_models.keys())
            aircraft_type = np.random.choice(available_types)
            aircraft_mesh = self.aircraft_models[aircraft_type]

            # Random aircraft pose
            aircraft_pose = self._generate_random_aircraft_pose()

            # Generate multiple views for this scene
            scene_cameras = self._generate_camera_positions()

            for view_idx, camera in enumerate(scene_cameras):
                # Render the view
                image, depth_map = self._render_view(aircraft_mesh, aircraft_pose, camera)

                # Save images
                image_filename = f"{split_name}_{scene_idx:06d}_{view_idx:02d}.png"
                image_path = os.path.join(output_dir, split_name, 'images', image_filename)
                image.save(image_path)

                depth_path = None
                if self.include_depth_maps and depth_map is not None:
                    depth_filename = f"{split_name}_{scene_idx:06d}_{view_idx:02d}_depth.png"
                    depth_path = os.path.join(output_dir, split_name, 'depth', depth_filename)
                    depth_map.save(depth_path)

                # Create annotation
                annotation = {
                    'scene_id': scene_idx,
                    'view_id': view_idx,
                    'image_path': image_path,
                    'depth_path': depth_path,
                    'aircraft_type': aircraft_type,
                    'aircraft_pose': aircraft_pose,
                    'camera_position': camera.position.tolist(),
                    'camera_target': camera.target.tolist(),
                    'image_size': self.image_size
                }
                annotations.append(annotation)

        return annotations

    def _generate_random_aircraft_pose(self) -> Dict:
        """Generate random 6DOF aircraft pose"""
        return {
            'position': [0.0, 0.0, 0.0],  # Aircraft at origin
            'rotation': {
                'pitch': np.random.uniform(-30, 30),
                'yaw': np.random.uniform(-180, 180),
                'roll': np.random.uniform(-15, 15)
            }
        }

    def _generate_camera_positions(self) -> List[Camera]:
        """Generate camera positions around the aircraft"""
        cameras = []

        for i in range(self.views_per_scene):
            # Circular camera positions around aircraft
            angle = (2 * math.pi * i) / self.views_per_scene
            distance = np.random.uniform(*self.camera_distance)
            height = np.random.uniform(*self.camera_height_range)

            # Camera position
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            z = height

            position = np.array([x, y, z])
            target = np.array([0.0, 0.0, 0.0])  # Look at aircraft origin

            camera = Camera(position, target)
            cameras.append(camera)

        return cameras

    def _render_view(self, aircraft_mesh, aircraft_pose: Dict, camera: Camera) -> Tuple[Image.Image, Optional[Image.Image]]:
        """Render aircraft from camera viewpoint"""
        # Create rendered image
        image = Image.new('RGB', self.image_size, color=(135, 206, 235))  # Sky blue
        draw = ImageDraw.Draw(image)

        # Transform aircraft vertices to camera space
        transformed_vertices = self._transform_vertices(aircraft_mesh.vertices, aircraft_pose, camera)

        # Project 3D points to 2D screen coordinates
        projected_points = self._project_to_screen(transformed_vertices)

        # Render faces
        self._render_faces(draw, aircraft_mesh.faces, projected_points, transformed_vertices)

        # Generate depth map if requested
        depth_map = None
        if self.include_depth_maps:
            depth_map = self._generate_depth_map(transformed_vertices, aircraft_mesh.faces)

        return image, depth_map

    def _transform_vertices(self, vertices: np.ndarray, aircraft_pose: Dict, camera: Camera) -> np.ndarray:
        """Transform aircraft vertices to camera coordinate system"""
        # Apply aircraft rotation
        rotation = aircraft_pose['rotation']

        # Simple rotation matrices (pitch, yaw, roll)
        pitch = math.radians(rotation['pitch'])
        yaw = math.radians(rotation['yaw'])
        roll = math.radians(rotation['roll'])

        # Rotation around X (pitch)
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch), math.cos(pitch)]
        ])

        # Rotation around Y (yaw)
        R_y = np.array([
            [math.cos(yaw), 0, math.sin(yaw)],
            [0, 1, 0],
            [-math.sin(yaw), 0, math.cos(yaw)]
        ])

        # Rotation around Z (roll)
        R_z = np.array([
            [math.cos(roll), -math.sin(roll), 0],
            [math.sin(roll), math.cos(roll), 0],
            [0, 0, 1]
        ])

        # Combined rotation
        R = R_z @ R_y @ R_x

        # Apply rotation to vertices
        rotated_vertices = vertices @ R.T

        # Transform to homogeneous coordinates
        homogeneous_vertices = np.hstack([rotated_vertices, np.ones((rotated_vertices.shape[0], 1))])

        # Apply camera view matrix
        camera_vertices = homogeneous_vertices @ camera.view_matrix.T

        return camera_vertices[:, :3]  # Return 3D coordinates

    def _project_to_screen(self, vertices_3d: np.ndarray) -> List[Tuple[int, int]]:
        """Project 3D vertices to 2D screen coordinates"""
        projected = []

        # Simple perspective projection
        focal_length = 200  # Focal length for perspective - reduced to prevent clipping

        for vertex in vertices_3d:
            if vertex[2] > 0.1:  # Avoid division by zero
                # Perspective projection
                x_proj = (vertex[0] * focal_length) / vertex[2]
                y_proj = (vertex[1] * focal_length) / vertex[2]

                # Convert to screen coordinates
                screen_x = int(self.image_size[0] / 2 + x_proj)
                screen_y = int(self.image_size[1] / 2 - y_proj)  # Flip Y

                projected.append((screen_x, screen_y))
            else:
                projected.append((0, 0))  # Behind camera

        return projected

    def _render_faces(self, draw: ImageDraw.Draw, faces: List, projected_points: List[Tuple[int, int]], vertices_3d: np.ndarray):
        """Render aircraft faces as polygons with depth-based shading"""
        # Sort faces by depth for proper rendering
        face_depths = []
        for face in faces:
            if len(face) >= 3:
                depths = [vertices_3d[i][2] for i in face if i < len(vertices_3d)]
                if depths:
                    avg_depth = np.mean(depths)
                    face_depths.append((avg_depth, face))

        # Sort by depth (farthest first)
        face_depths.sort(key=lambda x: x[0], reverse=True)

        for depth, face in face_depths:
            if len(face) >= 3:
                face_points = [projected_points[i] for i in face if i < len(projected_points)]

                if len(face_points) >= 3:
                    # Check if points form a valid shape
                    valid_points = []
                    for x, y in face_points:
                        if 0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]:
                            valid_points.append((x, y))

                    if len(valid_points) >= 3:
                        try:
                            # Depth-based shading - much darker
                            shade = max(20, min(80, int(40 + depth * 4)))
                            fill_color = (shade, shade, shade)
                            outline_color = (max(10, shade - 15), max(10, shade - 15), max(10, shade - 15))

                            draw.polygon(valid_points, fill=fill_color, outline=outline_color, width=1)
                        except:
                            # If polygon fails, just skip this face
                            pass

    def _generate_depth_map(self, vertices_3d: np.ndarray, faces: List) -> Image.Image:
        """Generate depth map for the rendered view"""
        # Create depth image (simplified)
        depth_array = np.full(self.image_size[::-1], 255, dtype=np.uint8)  # Far = white

        # For each face, compute average depth and fill polygon
        for face in faces:
            if len(face) >= 3:
                # Get face vertex depths
                face_depths = [vertices_3d[i][2] for i in face if i < len(vertices_3d)]
                if face_depths:
                    avg_depth = np.mean(face_depths)
                    # Convert depth to grayscale (closer = darker)
                    depth_value = max(0, min(255, int(255 - (avg_depth * 10))))

                    # This is a simplified depth map - a real implementation would
                    # use proper rasterization

        return Image.fromarray(depth_array, mode='L')

    def _save_annotations(self, annotations: List[Dict], output_dir: str, split_name: str):
        """Save annotations in JSON format"""
        output_file = os.path.join(output_dir, f"{split_name}_3d_annotations.json")
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)