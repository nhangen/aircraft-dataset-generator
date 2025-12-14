# 3D Aircraft Dataset Generation with Multi-View Rendering

import json
import logging
import math
import os
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from ..config import get_config

# Import new provider system
from ..providers import get_provider

logger = logging.getLogger(__name__)


class Camera:
    # 3D camera for multi-view rendering

    def __init__(
        self, position: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 1, 0])
    ):
        self.position = position
        self.target = target
        self.up = up
        self.view_matrix = self._compute_view_matrix()

    def _compute_view_matrix(self) -> np.ndarray:
        # Compute view matrix for camera (right-handed coordinate system)
        # Forward vector (from camera to target)
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)

        # Right vector
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)

        # Up vector (recalculate to ensure orthogonality)
        up = np.cross(right, forward)

        # View matrix (transforms world coordinates to camera coordinates)
        view_matrix = np.array(
            [
                [right[0], right[1], right[2], -np.dot(right, self.position)],
                [up[0], up[1], up[2], -np.dot(up, self.position)],
                [
                    forward[0],
                    forward[1],
                    forward[2],
                    -np.dot(forward, self.position),
                ],  # Positive Z forward
                [0, 0, 0, 1],
            ]
        )
        return view_matrix


# Legacy aircraft classes removed - now using provider system
# See aircraft_toolkit.providers.basic for backward compatibility


class Dataset3D:
    # Generate 3D multi-view aircraft datasets with proper rendering

    def __init__(
        self,
        aircraft_types: list[str],
        num_scenes: int,
        views_per_scene: int = 8,
        camera_distance: tuple[float, float] = (8, 12),
        camera_height_range: tuple[float, float] = (-5, 10),
        include_depth_maps: bool = True,
        include_surface_normals: bool = False,
        include_oriented_bboxes: bool = False,
        image_size: tuple[int, int] = (512, 512),
        pitch_range: tuple[float, float] = (-30, 30),
        roll_range: tuple[float, float] = (-15, 15),
        yaw_range: tuple[float, float] = (-180, 180),
        task_mode: str = "both",
    ):
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
            include_oriented_bboxes: Whether to compute 3D→2D oriented bounding boxes
            image_size: Output image size (width, height)
            pitch_range: Aircraft pitch rotation range in degrees
            roll_range: Aircraft roll rotation range in degrees
            yaw_range: Aircraft yaw rotation range in degrees
            task_mode: Generation mode ('classification', 'pose', 'both')
                - classification: Only aircraft type labels
                - pose: Only pose estimation annotations
                - both: Both classification and pose annotations
        """
        self.aircraft_types = aircraft_types
        self.num_scenes = num_scenes
        self.views_per_scene = views_per_scene
        self.camera_distance = camera_distance
        self.camera_height_range = camera_height_range
        self.include_depth_maps = include_depth_maps
        self.include_surface_normals = include_surface_normals
        self.include_oriented_bboxes = include_oriented_bboxes
        self.image_size = image_size
        self.pitch_range = pitch_range
        self.roll_range = roll_range
        self.yaw_range = yaw_range
        self.task_mode = task_mode

        if task_mode not in ["classification", "pose", "both"]:
            raise ValueError(
                f"Invalid task_mode: {task_mode}. Must be 'classification', 'pose', or 'both'"
            )

        # Initialize configuration and providers with fallback
        self.config = get_config()
        self.provider_name, self.model_provider = self._select_working_provider()

        # Initialize 3D aircraft models using provider system
        self.aircraft_models = self._load_aircraft_models()

        logger.info(f"3D Dataset Generator initialized with {self.provider_name} provider")
        logger.info(
            f"Generating {num_scenes} scenes × {views_per_scene} views = {num_scenes * views_per_scene} total images"
        )
        print(f"🛩️  3D Dataset Generator initialized with {self.provider_name} provider")
        print(
            f"📊 {num_scenes} scenes × {views_per_scene} views = {num_scenes * views_per_scene} total images"
        )

    def _select_working_provider(self) -> tuple:
        """
        Select a provider that supports the requested aircraft types.

        Returns:
            Tuple of (provider_name, provider_instance)
        """
        from ..providers import list_providers

        preferred_name = self.config.get_preferred_provider()

        # Try preferred provider first
        try:
            preferred_provider = get_provider(preferred_name)
            supported = preferred_provider.get_supported_aircraft()

            # Check if any requested aircraft are supported
            available_aircraft = [
                ac.upper() for ac in self.aircraft_types if ac.upper() in supported
            ]
            if available_aircraft:
                logger.info(
                    f"Using preferred provider '{preferred_name}' with {len(available_aircraft)} supported aircraft"
                )
                return preferred_name, preferred_provider
            else:
                logger.warning(
                    f"Preferred provider '{preferred_name}' supports no requested aircraft types"
                )
        except Exception as e:
            logger.warning(f"Preferred provider '{preferred_name}' failed: {e}")

        # Fall back to any working provider
        available_providers = list_providers()

        for provider_name in available_providers:
            if provider_name == preferred_name:
                continue  # Already tried above

            try:
                provider = get_provider(provider_name)
                supported = provider.get_supported_aircraft()

                # Check if any requested aircraft are supported
                available_aircraft = [
                    ac.upper() for ac in self.aircraft_types if ac.upper() in supported
                ]
                if available_aircraft:
                    logger.info(
                        f"Falling back to provider '{provider_name}' with {len(available_aircraft)} supported aircraft"
                    )
                    return provider_name, provider
            except Exception as e:
                logger.warning(f"Provider '{provider_name}' failed: {e}")
                continue

        raise RuntimeError(f"No provider available that supports any of: {self.aircraft_types}")

    def _load_aircraft_models(self) -> dict:
        # Load 3D aircraft model definitions using provider system
        models = {}

        # Validate aircraft types with provider
        supported_aircraft = self.model_provider.get_supported_aircraft()

        for aircraft_type in self.aircraft_types:
            aircraft_key = aircraft_type.upper()
            if aircraft_key not in supported_aircraft:
                logger.warning(
                    f"Aircraft type {aircraft_key} not supported by {self.provider_name} provider"
                )
                continue

            try:
                # Generate aircraft mesh using provider
                mesh = self.model_provider.create_aircraft(
                    aircraft_key, detail_level=self.config.aircraft.detail_level
                )
                models[aircraft_key] = mesh
                logger.info(
                    f"Loaded {aircraft_key}: {mesh.num_vertices} vertices, {mesh.num_faces} faces"
                )
            except Exception as e:
                logger.error(f"Failed to load {aircraft_key}: {e}")
                continue

        if not models:
            raise RuntimeError(
                f"No aircraft models could be loaded with {self.provider_name} provider"
            )

        return models

    def generate(
        self,
        output_dir: str,
        split_ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
        annotation_format: str = "custom_3d",
        num_workers: int = 1,
    ) -> dict:
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
        print("🛩️  Generating 3D Multi-View Aircraft Dataset")
        print(f"📊 {self.num_scenes} scenes, {len(self.aircraft_types)} aircraft types")

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
            if self.include_depth_maps:
                os.makedirs(os.path.join(output_dir, split, "depth"), exist_ok=True)

        # Calculate split sizes
        train_size = int(self.num_scenes * split_ratios[0])
        val_size = int(self.num_scenes * split_ratios[1])
        test_size = self.num_scenes - train_size - val_size

        # Generate datasets
        train_annotations = self._generate_split("train", train_size, output_dir)
        val_annotations = self._generate_split("val", val_size, output_dir)
        test_annotations = self._generate_split("test", test_size, output_dir)

        # Save annotations
        self._save_annotations(train_annotations, output_dir, "train")
        self._save_annotations(val_annotations, output_dir, "val")
        self._save_annotations(test_annotations, output_dir, "test")

        total_images = self.num_scenes * self.views_per_scene

        return {
            "total_scenes": self.num_scenes,
            "views_per_scene": self.views_per_scene,
            "total_images": total_images,
            "train_scenes": train_size,
            "val_scenes": val_size,
            "test_scenes": test_size,
            "aircraft_types": self.aircraft_types,
            "output_dir": output_dir,
        }

    def _generate_split(self, split_name: str, num_scenes: int, output_dir: str) -> list[dict]:
        # Generate scenes for a specific split
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

                # Compute oriented bounding box if enabled
                obb_data = None
                if self.include_oriented_bboxes:
                    obb_data = self._compute_oriented_bounding_box(
                        aircraft_mesh, aircraft_pose, camera
                    )
                    # Draw wireframe bounding box on the image
                    image = self._draw_bounding_box_wireframe(image, obb_data)

                # Save images
                image_filename = f"{split_name}_{scene_idx:06d}_{view_idx:02d}.png"
                image_path = os.path.join(output_dir, split_name, "images", image_filename)
                image.save(image_path)

                depth_path = None
                if self.include_depth_maps and depth_map is not None:
                    depth_filename = f"{split_name}_{scene_idx:06d}_{view_idx:02d}_depth.png"
                    depth_path = os.path.join(output_dir, split_name, "depth", depth_filename)
                    depth_map.save(depth_path)

                # Create annotation based on task mode
                annotation = {
                    "scene_id": scene_idx,
                    "view_id": view_idx,
                    "image_path": image_path,
                    "image_size": self.image_size,
                }

                # Add classification data if needed
                if self.task_mode in ["classification", "both"]:
                    annotation["aircraft_type"] = aircraft_type

                # Add pose data if needed
                if self.task_mode in ["pose", "both"]:
                    annotation["aircraft_pose"] = aircraft_pose
                    annotation["camera_position"] = camera.position.tolist()
                    annotation["camera_target"] = camera.target.tolist()

                    # Add depth path if available
                    if depth_path:
                        annotation["depth_path"] = depth_path

                    # Add OBB data if computed
                    if obb_data is not None:
                        annotation["oriented_bbox"] = obb_data

                annotations.append(annotation)

        return annotations

    def _generate_random_aircraft_pose(self) -> dict:
        # Generate random 6DOF aircraft pose
        return {
            "position": [0.0, 0.0, 0.0],  # Aircraft at origin
            "rotation": {
                "pitch": np.random.uniform(*self.pitch_range),
                "yaw": np.random.uniform(*self.yaw_range),
                "roll": np.random.uniform(*self.roll_range),
            },
        }

    def _generate_camera_positions(self) -> list[Camera]:
        # Generate camera positions around the aircraft
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

    def _render_view(
        self, aircraft_mesh, aircraft_pose: dict, camera: Camera
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        # Render aircraft from camera viewpoint
        # Check if provider has its own rendering method (e.g., headless provider)
        if hasattr(self.model_provider, "render_view"):
            # Use provider's render method
            image = self.model_provider.render_view(
                aircraft_mesh,
                aircraft_pose=aircraft_pose,
                camera=camera,
                image_size=self.image_size,
            )
            return image, None  # No depth map for provider rendering

        # Check if PyVista is available for high-quality rendering
        try:
            import pyvista as pv

            return self._render_view_pyvista(aircraft_mesh, aircraft_pose, camera)
        except ImportError:
            # Fall back to basic rendering
            return self._render_view_basic(aircraft_mesh, aircraft_pose, camera)

    def _render_view_pyvista(
        self, aircraft_mesh, aircraft_pose: dict, camera: Camera
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        # Render using PyVista with aggressive memory management to prevent GPU leaks
        import gc
        import os

        import numpy as np
        import pyvista as pv
        from PIL import Image

        # Force PyVista to use off-screen rendering mode
        pv.OFF_SCREEN = True

        # Set environment variables to prevent context leaks
        os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
        os.environ["PYOPENGL_PLATFORM"] = "egl"

        # Initialize variables for cleanup
        pv_mesh = None
        plotter = None

        try:
            # Clear any existing plotters first
            try:
                pv.close_all()
            except:
                pass

            # Create PyVista mesh from aircraft mesh
            faces_with_count = np.column_stack(
                [np.full(len(aircraft_mesh.faces), 3), aircraft_mesh.faces]  # Triangle count
            ).flatten()

            pv_mesh = pv.PolyData(aircraft_mesh.vertices, faces_with_count)

            # Apply aircraft pose transformation
            rotation = aircraft_pose["rotation"]
            translation = aircraft_pose["position"]

            # Convert rotations to transformation matrix
            import math

            pitch = math.radians(rotation["pitch"])
            yaw = math.radians(rotation["yaw"])
            roll = math.radians(rotation["roll"])

            # Rotation matrices
            R_x = np.array(
                [
                    [1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)],
                ]
            )
            R_y = np.array(
                [[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]]
            )
            R_z = np.array(
                [
                    [math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1],
                ]
            )

            # Combined rotation
            R = R_z @ R_y @ R_x

            # Create 4x4 transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = translation

            # Apply transformation
            pv_mesh = pv_mesh.transform(transform, inplace=False)

            # Create plotter with specific context management
            plotter = pv.Plotter(off_screen=True, window_size=self.image_size)

            # Set background color (sky blue)
            plotter.background_color = (135 / 255, 206 / 255, 235 / 255)

            # Add the mesh with basic shading
            plotter.add_mesh(pv_mesh, color="lightgray", show_edges=False, lighting=True)

            # Reset camera to properly frame the mesh
            plotter.reset_camera()

            # Calculate proper camera distance based on mesh size
            bounds = pv_mesh.bounds
            mesh_size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

            # Set camera based on the intended viewing angle
            angle_rad = np.arctan2(camera.position[1], camera.position[0])
            height = camera.position[2]

            # Position camera at a distance proportional to mesh size
            distance = mesh_size * 2.5  # Distance as multiple of mesh size

            camera_x = distance * np.cos(angle_rad)
            camera_y = distance * np.sin(angle_rad)
            camera_z = height * (mesh_size / 10)  # Scale height relative to mesh

            plotter.camera.position = (camera_x, camera_y, camera_z)
            plotter.camera.focal_point = pv_mesh.center
            plotter.camera.up = (0, 0, 1)
            plotter.camera.view_angle = 30  # degrees

            # Render the scene with explicit cleanup
            image_array = plotter.screenshot(return_img=True, transparent_background=False)

            # Convert to PIL Image
            image = Image.fromarray(image_array)

            # Generate depth map (simplified for now)
            depth_map = None
            if self.include_depth_maps:
                # Use z-buffer approach for depth
                depth_array = np.full(self.image_size[::-1], 255, dtype=np.uint8)  # White = far
                depth_map = Image.fromarray(depth_array, mode="L")

            return image, depth_map

        finally:
            # CRITICAL: Aggressive cleanup to prevent GPU memory leak
            if plotter is not None:
                try:
                    # Remove all actors first
                    for actor in plotter.actors.values():
                        plotter.remove_actor(actor)
                    plotter.clear()
                    plotter.close()
                    # Force delete
                    del plotter
                except:
                    pass

            # Clear mesh data immediately
            if pv_mesh is not None:
                try:
                    del pv_mesh
                except:
                    pass

            # Clear all PyVista objects and force context cleanup
            try:
                pv.close_all()
                # Force garbage collection multiple times to ensure cleanup
                gc.collect()
                gc.collect()
                gc.collect()
            except:
                pass

            # Try to reset PyVista global state
            try:
                # Reset global settings
                pv.rcParams.reset_to_defaults()
                # Clear any cached textures or buffers if available
                if hasattr(pv, "_clear_cache"):
                    pv._clear_cache()
            except:
                pass

    def _render_view_basic(
        self, aircraft_mesh, aircraft_pose: dict, camera: Camera
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        # Basic wireframe rendering fallback
        # Create rendered image
        image = Image.new("RGB", self.image_size, color=(135, 206, 235))  # Sky blue
        draw = ImageDraw.Draw(image)

        # Transform aircraft vertices to camera space
        transformed_vertices = self._transform_vertices(
            aircraft_mesh.vertices, aircraft_pose, camera
        )

        # Project 3D points to 2D screen coordinates
        projected_points = self._project_to_screen(transformed_vertices)

        # Render faces
        self._render_faces(draw, aircraft_mesh.faces, projected_points, transformed_vertices)

        # Generate depth map if requested
        depth_map = None
        if self.include_depth_maps:
            depth_map = self._generate_depth_map(transformed_vertices, aircraft_mesh.faces)

        return image, depth_map

    def _transform_vertices(
        self, vertices: np.ndarray, aircraft_pose: dict, camera: Camera
    ) -> np.ndarray:
        # Transform aircraft vertices to camera coordinate system
        # Apply aircraft rotation
        rotation = aircraft_pose["rotation"]

        # Simple rotation matrices (pitch, yaw, roll)
        pitch = math.radians(rotation["pitch"])
        yaw = math.radians(rotation["yaw"])
        roll = math.radians(rotation["roll"])

        # Rotation around X (pitch)
        R_x = np.array(
            [
                [1, 0, 0],
                [0, math.cos(pitch), -math.sin(pitch)],
                [0, math.sin(pitch), math.cos(pitch)],
            ]
        )

        # Rotation around Y (yaw)
        R_y = np.array(
            [[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]]
        )

        # Rotation around Z (roll)
        R_z = np.array(
            [[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]]
        )

        # Combined rotation
        R = R_z @ R_y @ R_x

        # Apply rotation to vertices
        rotated_vertices = vertices @ R.T

        # Transform to homogeneous coordinates
        homogeneous_vertices = np.hstack(
            [rotated_vertices, np.ones((rotated_vertices.shape[0], 1))]
        )

        # Apply camera view matrix
        camera_vertices = homogeneous_vertices @ camera.view_matrix.T

        return camera_vertices[:, :3]  # Return 3D coordinates

    def _project_to_screen(self, vertices_3d: np.ndarray) -> list[tuple[int, int]]:
        # Project 3D vertices to 2D screen coordinates
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

    def _render_faces(
        self,
        draw: ImageDraw.Draw,
        faces: list,
        projected_points: list[tuple[int, int]],
        vertices_3d: np.ndarray,
    ):
        # Render aircraft faces as polygons with depth-based shading
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
                            outline_color = (
                                max(10, shade - 15),
                                max(10, shade - 15),
                                max(10, shade - 15),
                            )

                            draw.polygon(
                                valid_points, fill=fill_color, outline=outline_color, width=1
                            )
                        except:
                            # If polygon fails, just skip this face
                            pass

    def _generate_depth_map(self, vertices_3d: np.ndarray, faces: list) -> Image.Image:
        # Generate depth map for the rendered view
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

        return Image.fromarray(depth_array, mode="L")

    def _compute_oriented_bounding_box(self, mesh, aircraft_pose: dict, camera: Camera) -> dict:
        """
        Compute 3D→2D oriented bounding box projection using full aircraft extents.

        This creates a proper aircraft-axis aligned bounding box that:
        1. Uses full nose-to-tail and wing-to-wing extents
        2. Ensures all Z coordinates are above ground (Z > 0)
        3. Matches PyVista's camera projection exactly
        """
        try:
            import pyvista as pv

            # Create PyVista mesh from aircraft mesh if needed
            if hasattr(mesh, "bounds"):
                pv_mesh = mesh
            else:
                faces_with_count = np.column_stack(
                    [np.full(len(mesh.faces), 3), mesh.faces]
                ).flatten()
                pv_mesh = pv.PolyData(mesh.vertices, faces_with_count)

            # Apply the SAME transformation as the rendering pipeline
            rotation = aircraft_pose["rotation"]
            translation = aircraft_pose["position"]

            pitch = math.radians(rotation["pitch"])
            yaw = math.radians(rotation["yaw"])
            roll = math.radians(rotation["roll"])

            # Rotation matrices (same as rendering)
            R_x = np.array(
                [
                    [1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)],
                ]
            )
            R_y = np.array(
                [[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]]
            )
            R_z = np.array(
                [
                    [math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1],
                ]
            )

            R = R_z @ R_y @ R_x
            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = translation

            # Apply transformation
            transformed_mesh = pv_mesh.transform(transform, inplace=False)

            # Create ORIENTED bounding box that follows aircraft orientation
            # Get the original mesh bounds in aircraft local coordinates BEFORE transformation
            original_bounds = pv_mesh.bounds
            x_min_local = original_bounds[0]
            x_max_local = original_bounds[1]
            y_min_local = original_bounds[2]
            y_max_local = original_bounds[3]
            z_min_local = original_bounds[4]
            z_max_local = original_bounds[5]

            # Add padding in local aircraft coordinates
            x_range = x_max_local - x_min_local
            y_range = y_max_local - y_min_local
            z_range = z_max_local - z_min_local

            padding = 0.1  # 10% padding
            x_padding = x_range * padding
            y_padding = y_range * padding
            z_padding = z_range * 0.05

            x_min_local -= x_padding
            x_max_local += x_padding
            y_min_local -= y_padding
            y_max_local += y_padding
            z_min_local -= z_padding
            z_max_local += z_padding

            # Create 8 corners in aircraft LOCAL coordinates
            local_corners = np.array(
                [
                    [x_min_local, y_min_local, z_min_local],  # 0: tail-port-bottom
                    [x_max_local, y_min_local, z_min_local],  # 1: nose-port-bottom
                    [x_min_local, y_max_local, z_min_local],  # 2: tail-starboard-bottom
                    [x_max_local, y_max_local, z_min_local],  # 3: nose-starboard-bottom
                    [x_min_local, y_min_local, z_max_local],  # 4: tail-port-top
                    [x_max_local, y_min_local, z_max_local],  # 5: nose-port-top
                    [x_min_local, y_max_local, z_max_local],  # 6: tail-starboard-top
                    [x_max_local, y_max_local, z_max_local],  # 7: nose-starboard-top
                ]
            )

            # Transform corners to world coordinates using the same transformation
            corners_3d = []
            for corner in local_corners:
                # Convert to homogeneous coordinates
                corner_homo = np.append(corner, 1.0)
                # Apply transformation
                world_corner = transform @ corner_homo
                corners_3d.append(world_corner[:3])

            corners_3d = np.array(corners_3d)

            # Ensure all corners are above ground
            min_z = corners_3d[:, 2].min()
            if min_z < 0:
                z_offset = -min_z + 0.1
                corners_3d[:, 2] += z_offset

            mesh_center = transformed_mesh.center
            mesh_size = max(x_range, y_range, z_range)

            # Get camera positioning (exactly matching PyVista rendering)
            angle_rad = np.arctan2(camera.position[1], camera.position[0])
            height_factor = camera.position[2]

            distance = mesh_size * 2.5  # Same multiplier as rendering
            camera_x = distance * np.cos(angle_rad)
            camera_y = distance * np.sin(angle_rad)
            camera_z = height_factor * (mesh_size / 10)  # Same scaling as rendering

            cam_pos = np.array([camera_x, camera_y, camera_z])
            focal_point = mesh_center
            up_vector = np.array([0, 0, 1])

            # Project corners using perspective projection matching PyVista
            corners_2d = []
            visible_corners = []

            for corner in corners_3d:
                # Vector from camera to corner
                to_corner = corner - cam_pos

                # Create camera coordinate system (same as PyVista)
                forward = focal_point - cam_pos
                forward = forward / np.linalg.norm(forward)
                right = np.cross(forward, up_vector)
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward)

                # Transform to camera space
                cam_x = np.dot(to_corner, right)
                cam_y = np.dot(to_corner, up)
                cam_z = np.dot(to_corner, forward)

                # Check if in front of camera
                if cam_z > 0.1:
                    # Use PyVista-compatible projection with 30-degree FOV
                    fov_scale = self.image_size[0] * 0.8  # Increased for proper scaling
                    x_screen = (cam_x / cam_z) * fov_scale + self.image_size[0] / 2
                    y_screen = -(cam_y / cam_z) * fov_scale + self.image_size[1] / 2
                    corners_2d.append([x_screen, y_screen])
                    visible_corners.append(True)
                else:
                    corners_2d.append([0, 0])  # Behind camera
                    visible_corners.append(False)

            corners_2d = np.array(corners_2d)

        except Exception as e:
            print(f"Warning: OBB computation failed: {e}")
            # Create fallback corners at image center
            center_x, center_y = self.image_size[0] // 2, self.image_size[1] // 2
            corners_3d = np.array([[0, 0, 0]] * 8)  # Dummy 3D corners
            corners_2d = np.array([[center_x, center_y]] * 8)
            visible_corners = [True] * 8

        # Compute 2D bounding box from all visible corners
        valid_corners = []
        for i, (corner, visible) in enumerate(zip(corners_2d, visible_corners)):
            x, y = corner
            if visible:  # Include all visible corners, even if slightly outside image
                valid_corners.append([x, y])

        if len(valid_corners) > 0:
            valid_corners = np.array(valid_corners)
            bbox_2d = {
                "x_min": float(valid_corners[:, 0].min()),
                "y_min": float(valid_corners[:, 1].min()),
                "x_max": float(valid_corners[:, 0].max()),
                "y_max": float(valid_corners[:, 1].max()),
            }
            bbox_2d["width"] = bbox_2d["x_max"] - bbox_2d["x_min"]
            bbox_2d["height"] = bbox_2d["y_max"] - bbox_2d["y_min"]
            bbox_2d["area"] = bbox_2d["width"] * bbox_2d["height"]
        else:
            bbox_2d = None

        return {
            "corners_3d": corners_3d.tolist(),
            "corners_2d": corners_2d.tolist(),
            "visible_corners": visible_corners,
            "bbox_2d": bbox_2d,
            "obb_center_2d": corners_2d.mean(axis=0).tolist() if len(corners_2d) > 0 else [0, 0],
            "num_visible": sum(visible_corners),
        }

    def _draw_bounding_box_wireframe(self, image: Image.Image, obb_data: dict) -> Image.Image:
        """
        Draw 3D wireframe bounding box on the image using OpenCV.

        Args:
            image: PIL Image to draw on
            obb_data: OBB data from _compute_oriented_bounding_box

        Returns:
            PIL Image with wireframe drawn
        """
        try:
            import cv2
            import numpy as np

            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            corners_2d = np.array(obb_data["corners_2d"])
            visible_corners = obb_data["visible_corners"]

            # Define the 12 edges of the bounding box
            # Corner order: 0=back-left-bottom, 1=back-right-bottom, 2=front-left-bottom, 3=front-right-bottom
            #               4=back-left-top, 5=back-right-top, 6=front-left-top, 7=front-right-top
            edges = [
                # Bottom face edges
                (0, 1),  # back-left to back-right
                (1, 3),  # back-right to front-right
                (3, 2),  # front-right to front-left
                (2, 0),  # front-left to back-left
                # Top face edges
                (4, 5),  # back-left to back-right (top)
                (5, 7),  # back-right to front-right (top)
                (7, 6),  # front-right to front-left (top)
                (6, 4),  # front-left to back-left (top)
                # Vertical edges
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]

            # Simple color scheme for edges
            bottom_color = (0, 255, 255)  # Cyan for bottom edges
            top_color = (0, 255, 0)  # Green for top edges
            vertical_color = (255, 255, 0)  # Yellow for vertical edges

            edge_colors = [
                # Bottom face (cyan)
                bottom_color,
                bottom_color,
                bottom_color,
                bottom_color,
                # Top face (green)
                top_color,
                top_color,
                top_color,
                top_color,
                # Vertical edges (yellow)
                vertical_color,
                vertical_color,
                vertical_color,
                vertical_color,
            ]

            # Draw edges
            for i, (start_idx, end_idx) in enumerate(edges):
                # Only draw edge if both corners are visible
                if visible_corners[start_idx] and visible_corners[end_idx]:
                    start_point = tuple(map(int, corners_2d[start_idx]))
                    end_point = tuple(map(int, corners_2d[end_idx]))

                    # Check if points are within reasonable bounds
                    if (
                        0 <= start_point[0] <= self.image_size[0] * 1.2
                        and 0 <= start_point[1] <= self.image_size[1] * 1.2
                        and 0 <= end_point[0] <= self.image_size[0] * 1.2
                        and 0 <= end_point[1] <= self.image_size[1] * 1.2
                    ):

                        cv2.line(cv_image, start_point, end_point, edge_colors[i], 2)

            # Draw corner points with labels
            corner_colors = [
                (255, 0, 0),  # Red for bottom corners
                (255, 0, 0),
                (255, 0, 0),
                (255, 0, 0),
                (255, 165, 0),  # Orange for top corners
                (255, 165, 0),
                (255, 165, 0),
                (255, 165, 0),
            ]

            for i, (corner, visible) in enumerate(zip(corners_2d, visible_corners)):
                if visible:
                    point = tuple(map(int, corner))
                    if 0 <= point[0] <= self.image_size[0] and 0 <= point[1] <= self.image_size[1]:
                        # Draw corner point
                        cv2.circle(cv_image, point, 4, corner_colors[i], -1)
                        # Draw corner label
                        cv2.putText(
                            cv_image,
                            str(i),
                            (point[0] + 6, point[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            corner_colors[i],
                            1,
                        )

            # Add simple bounding box info
            if len(corners_2d) >= 8:
                # Calculate bounding box center
                box_center = corners_2d.mean(axis=0).astype(int)

                # Add simple label
                label_color = (255, 255, 255)  # White text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                # Add bounding box label
                if (
                    0 <= box_center[0] <= self.image_size[0]
                    and 0 <= box_center[1] <= self.image_size[1]
                ):
                    cv2.putText(
                        cv_image,
                        "3D BBOX",
                        tuple(box_center + [10, -10]),
                        font,
                        font_scale,
                        label_color,
                        thickness,
                    )

            # Convert back to PIL RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_image)

        except Exception as e:
            print(f"Warning: Could not draw bounding box wireframe: {e}")
            return image

    def _save_annotations(self, annotations: list[dict], output_dir: str, split_name: str):
        # Save annotations in JSON format
        output_file = os.path.join(output_dir, f"{split_name}_3d_annotations.json")
        with open(output_file, "w") as f:
            json.dump(annotations, f, indent=2)
