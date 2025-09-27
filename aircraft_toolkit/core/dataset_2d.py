"""2D Aircraft Dataset Generation"""

import os
import json
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

from ..models.military import F15Fighter, B52Bomber, C130Transport


class Dataset2D:
    """Generate 2D aircraft silhouette datasets with pose annotations"""
    
    def __init__(self,
                 aircraft_types: List[str],
                 num_samples: int,
                 image_size: Tuple[int, int] = (224, 224),
                 pose_range: Optional[Dict] = None,
                 background_type: str = 'gradient',
                 task_mode: str = 'both'):
        """
        Initialize 2D dataset generator

        Args:
            aircraft_types: List of aircraft type names ['F15', 'B52', 'C130']
            num_samples: Total number of images to generate
            image_size: Output image dimensions (width, height)
            pose_range: Dict with pose ranges, e.g. {'pitch': (-45, 45)}
            background_type: Background style ('gradient', 'solid', 'sky')
            task_mode: Generation mode ('classification', 'pose', 'both')
                - classification: Only aircraft type labels
                - pose: Only pose estimation annotations
                - both: Both classification and pose annotations
        """
        self.aircraft_types = aircraft_types
        self.num_samples = num_samples
        self.image_size = image_size
        self.background_type = background_type
        self.task_mode = task_mode

        if task_mode not in ['classification', 'pose', 'both']:
            raise ValueError(f"Invalid task_mode: {task_mode}. Must be 'classification', 'pose', or 'both'")
        
        # Default pose ranges
        default_ranges = {
            'pitch': (-45, 45),
            'yaw': (-180, 180),
            'roll': (-30, 30),
            'x': (-0.2, 0.2),
            'y': (-0.2, 0.2),
            'z': (0.8, 1.2)
        }

        if pose_range:
            # Merge user ranges with defaults
            self.pose_range = default_ranges.copy()
            self.pose_range.update(pose_range)
        else:
            self.pose_range = default_ranges
        
        # Initialize aircraft models
        self.aircraft_models = self._load_aircraft_models()
    
    def _load_aircraft_models(self) -> Dict:
        """Load aircraft model definitions"""
        models = {}
        for aircraft_type in self.aircraft_types:
            if aircraft_type.upper() == 'F15':
                models['F15'] = F15Fighter()
            elif aircraft_type.upper() == 'B52':
                models['B52'] = B52Bomber()
            elif aircraft_type.upper() == 'C130':
                models['C130'] = C130Transport()
            else:
                raise ValueError(f"Unknown aircraft type: {aircraft_type}")
        return models
    
    def generate(self,
                 output_dir: str,
                 split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                 annotation_format: str = 'coco',
                 num_workers: int = 1) -> Dict:
        """
        Generate the 2D aircraft dataset
        
        Args:
            output_dir: Directory to save generated dataset
            split_ratios: Train/val/test split ratios
            annotation_format: Annotation format ('coco', 'yolo', 'custom')
            num_workers: Number of parallel workers (not implemented yet)
            
        Returns:
            Dict with dataset statistics and file paths
        """
        print(f"🛩️  Generating 2D Aircraft Dataset")
        print(f"📊 {self.num_samples} samples, {len(self.aircraft_types)} aircraft types")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        
        # Calculate split sizes
        train_size = int(self.num_samples * split_ratios[0])
        val_size = int(self.num_samples * split_ratios[1])
        test_size = self.num_samples - train_size - val_size
        
        # Generate datasets
        train_annotations = self._generate_split('train', train_size, output_dir)
        val_annotations = self._generate_split('val', val_size, output_dir)
        test_annotations = self._generate_split('test', test_size, output_dir)
        
        # Save annotations
        self._save_annotations(train_annotations, output_dir, 'train', annotation_format)
        self._save_annotations(val_annotations, output_dir, 'val', annotation_format)
        self._save_annotations(test_annotations, output_dir, 'test', annotation_format)
        
        return {
            'total_samples': self.num_samples,
            'train_samples': train_size,
            'val_samples': val_size,
            'test_samples': test_size,
            'aircraft_types': self.aircraft_types,
            'output_dir': output_dir
        }
    
    def _generate_split(self, split_name: str, num_samples: int, output_dir: str) -> List[Dict]:
        """Generate samples for a specific split"""
        annotations = []
        
        for i in tqdm(range(num_samples), desc=f"Generating {split_name}"):
            # Random aircraft type
            aircraft_type = np.random.choice(self.aircraft_types)
            aircraft_model = self.aircraft_models[aircraft_type]
            
            # Random pose
            pose = self._generate_random_pose()
            
            # Generate image with bounding box
            image, bbox_data = self._render_aircraft(aircraft_model, pose)

            # Save image
            image_filename = f"{split_name}_{i:06d}.png"
            image_path = os.path.join(output_dir, split_name, 'images', image_filename)
            image.save(image_path)

            # Create annotation based on task mode
            annotation = {
                'scene_id': i,
                'view_id': 0,  # 2D images only have one view
                'image_path': image_path,
                'image_size': list(self.image_size)
            }

            # Add classification data if needed
            if self.task_mode in ['classification', 'both']:
                annotation['aircraft_type'] = aircraft_type

            # Add pose data if needed
            if self.task_mode in ['pose', 'both']:
                annotation['aircraft_pose'] = {
                    'position': [pose['x'], pose['y'], pose['z']],
                    'rotation': {
                        'pitch': pose['pitch'],
                        'yaw': pose['yaw'],
                        'roll': pose['roll']
                    }
                }
                annotation['camera_position'] = [0.0, 0.0, 5.0]  # Default 2D camera position
                annotation['camera_target'] = [0.0, 0.0, 0.0]

                # Add bounding box data for pose estimation
                if bbox_data:
                    annotation['bbox_2d'] = bbox_data['bbox_2d']
                    annotation['bbox_center'] = bbox_data['center']
                    annotation['bbox_size'] = [bbox_data['width'], bbox_data['height']]

            annotations.append(annotation)
        
        return annotations
    
    def _generate_random_pose(self) -> Dict:
        """Generate random 6DOF pose within specified ranges"""
        return {
            'pitch': np.random.uniform(*self.pose_range['pitch']),
            'yaw': np.random.uniform(*self.pose_range['yaw']),
            'roll': np.random.uniform(*self.pose_range['roll']),
            'x': np.random.uniform(*self.pose_range['x']),
            'y': np.random.uniform(*self.pose_range['y']),
            'z': np.random.uniform(*self.pose_range['z'])
        }
    
    def _render_aircraft(self, aircraft_model, pose: Dict) -> Tuple[Image.Image, Dict]:
        """Render aircraft silhouette with given pose and compute bounding box"""
        # Create image with gradient background
        image = Image.new('RGB', self.image_size, color=(135, 206, 235))  # Sky blue
        draw = ImageDraw.Draw(image)

        # Get aircraft silhouette points
        silhouette_points = aircraft_model.get_silhouette_for_pose(pose)

        # Scale and center points
        scaled_points = self._scale_points_to_image(silhouette_points)

        # Compute bounding box from silhouette points
        bbox_data = {}
        if len(scaled_points) > 2:
            # Draw aircraft silhouette
            draw.polygon(scaled_points, fill=(64, 64, 64), outline=(32, 32, 32))

            # Compute 2D bounding box
            xs = [p[0] for p in scaled_points]
            ys = [p[1] for p in scaled_points]
            bbox_data = {
                'bbox_2d': [min(xs), min(ys), max(xs), max(ys)],  # [x_min, y_min, x_max, y_max]
                'center': [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2],
                'width': max(xs) - min(xs),
                'height': max(ys) - min(ys),
                'silhouette_points': scaled_points  # Keep the actual silhouette for advanced pose estimation
            }

        return image, bbox_data
    
    def _scale_points_to_image(self, points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """Scale normalized aircraft points to image coordinates"""
        scaled = []
        center_x, center_y = self.image_size[0] // 2, self.image_size[1] // 2
        scale = min(self.image_size) * 0.3  # Aircraft takes 30% of image
        
        for x, y in points:
            img_x = int(center_x + x * scale)
            img_y = int(center_y + y * scale)
            scaled.append((img_x, img_y))
        
        return scaled
    
    def _save_annotations(self, annotations: List[Dict], output_dir: str, 
                         split_name: str, format_type: str):
        """Save annotations in specified format"""
        if format_type == 'coco':
            self._save_coco_format(annotations, output_dir, split_name)
        elif format_type == 'custom':
            self._save_custom_format(annotations, output_dir, split_name)
        else:
            raise ValueError(f"Unsupported annotation format: {format_type}")
    
    def _save_custom_format(self, annotations: List[Dict], output_dir: str, split_name: str):
        """Save in custom JSON format (now matches 3D format)"""
        output_file = os.path.join(output_dir, f"{split_name}_annotations.json")
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
    
    def _save_coco_format(self, annotations: List[Dict], output_dir: str, split_name: str):
        """Save in COCO format with task mode support"""
        coco_data = {
            "info": {"description": f"Aircraft Dataset - {self.task_mode} mode"},
            "categories": [
                {"id": 1, "name": "F15"},
                {"id": 2, "name": "B52"},
                {"id": 3, "name": "C130"}
            ],
            "images": [],
            "annotations": []
        }

        annotation_id = 1
        for ann in annotations:
            # Add image info
            coco_data["images"].append({
                "id": ann["scene_id"],
                "file_name": os.path.basename(ann["image_path"]),
                "width": ann["image_size"][0],
                "height": ann["image_size"][1]
            })

            # Add annotation based on task mode
            coco_annotation = {
                "id": annotation_id,
                "image_id": ann["scene_id"],
                "iscrowd": 0
            }

            # Add classification data if in classification or both mode
            if self.task_mode in ['classification', 'both'] and 'aircraft_type' in ann:
                category_map = {"F15": 1, "B52": 2, "C130": 3}
                coco_annotation["category_id"] = category_map.get(ann["aircraft_type"], 1)

            # Add bounding box and pose data if in pose or both mode
            if self.task_mode in ['pose', 'both']:
                if 'bbox_2d' in ann:
                    # COCO bbox format: [x, y, width, height]
                    bbox = ann['bbox_2d']  # [x_min, y_min, x_max, y_max]
                    coco_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    coco_annotation["bbox"] = coco_bbox
                    coco_annotation["area"] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                # Add pose data as custom fields
                if 'aircraft_pose' in ann:
                    coco_annotation["pose"] = ann["aircraft_pose"]
                if 'camera_position' in ann:
                    coco_annotation["camera_position"] = ann["camera_position"]
                if 'camera_target' in ann:
                    coco_annotation["camera_target"] = ann["camera_target"]

            coco_data["annotations"].append(coco_annotation)
            annotation_id += 1

        output_file = os.path.join(output_dir, f"{split_name}_coco.json")
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)