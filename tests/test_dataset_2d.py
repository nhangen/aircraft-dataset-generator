# Unit tests for 2D dataset generation

import json
import os
import shutil
import tempfile
import unittest

from PIL import Image

from aircraft_toolkit.core.dataset_2d import Dataset2D


class TestDataset2D(unittest.TestCase):
    # Test 2D dataset generation functionality

    def setUp(self):
        # Set up test fixtures
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = Dataset2D(
            aircraft_types=["F15", "B52"],
            num_samples=10,  # Small number for testing
            image_size=(64, 64),  # Small images for speed
        )

    def tearDown(self):
        # Clean up test fixtures
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        # Test dataset can be initialized with valid parameters
        self.assertEqual(self.dataset.aircraft_types, ["F15", "B52"])
        self.assertEqual(self.dataset.num_samples, 10)
        self.assertEqual(self.dataset.image_size, (64, 64))
        self.assertIn("F15", self.dataset.aircraft_models)
        self.assertIn("B52", self.dataset.aircraft_models)

    def test_invalid_aircraft_type(self):
        # Test that invalid aircraft types raise errors
        with self.assertRaises(ValueError):
            Dataset2D(aircraft_types=["INVALID_AIRCRAFT"], num_samples=1)

    def test_pose_generation(self):
        # Test random pose generation
        pose = self.dataset._generate_random_pose()

        # Check that all required pose components are present
        required_keys = ["pitch", "yaw", "roll", "x", "y", "z"]
        for key in required_keys:
            self.assertIn(key, pose)
            self.assertIsInstance(pose[key], (int, float))

        # Check that values are within expected ranges
        self.assertGreaterEqual(pose["pitch"], -45)
        self.assertLessEqual(pose["pitch"], 45)
        self.assertGreaterEqual(pose["yaw"], -180)
        self.assertLessEqual(pose["yaw"], 180)

    def test_point_scaling(self):
        # Test scaling of aircraft points to image coordinates
        points = [(0.0, 0.0), (0.5, 0.5), (-0.5, -0.5)]
        scaled = self.dataset._scale_points_to_image(points)

        self.assertEqual(len(scaled), len(points))

        # Check that scaled points are integers within image bounds
        for x, y in scaled:
            self.assertIsInstance(x, int)
            self.assertIsInstance(y, int)
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, self.dataset.image_size[0])
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, self.dataset.image_size[1])

    def test_aircraft_rendering(self):
        # Test aircraft rendering produces valid images
        aircraft_model = self.dataset.aircraft_models["F15"]
        pose = self.dataset._generate_random_pose()

        image, bbox_data = self.dataset._render_aircraft(aircraft_model, pose)

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, self.dataset.image_size)
        self.assertEqual(image.mode, "RGB")

        # Test bounding box data
        self.assertIsInstance(bbox_data, dict)
        if bbox_data:  # Only check if aircraft was rendered
            self.assertIn("bbox_2d", bbox_data)
            self.assertIn("center", bbox_data)
            self.assertIn("width", bbox_data)
            self.assertIn("height", bbox_data)

    def test_small_dataset_generation(self):
        # Test generation of a small complete dataset
        # Generate very small dataset for testing
        small_dataset = Dataset2D(aircraft_types=["F15"], num_samples=3, image_size=(32, 32))

        _results = small_dataset.generate(
            output_dir=self.temp_dir,
            split_ratios=(0.6, 0.4, 0.0),  # No test split
            annotation_format="custom",
        )

        # Check results structure
        self.assertEqual(_results["total_samples"], 3)
        self.assertEqual(_results["train_samples"], 1)  # 60% of 3 = 1
        self.assertEqual(_results["val_samples"], 1)  # 40% of 3 = 1
        self.assertEqual(_results["test_samples"], 1)  # Remainder

        # Check that directories were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "train", "images")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "val", "images")))

        # Check that annotation files were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "train_annotations.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "val_annotations.json")))

    def test_annotation_format_custom(self):
        # Test custom annotation format generation
        small_dataset = Dataset2D(aircraft_types=["F15"], num_samples=2, image_size=(32, 32))

        _results = small_dataset.generate(
            output_dir=self.temp_dir,
            split_ratios=(1.0, 0.0, 0.0),  # Only train split
            annotation_format="custom",
        )

        # Load and validate annotations
        ann_file = os.path.join(self.temp_dir, "train_annotations.json")
        with open(ann_file) as f:
            annotations = json.load(f)

        self.assertEqual(len(annotations), 2)

        for ann in annotations:
            # Check required fields (unified format)
            self.assertIn("scene_id", ann)
            self.assertIn("view_id", ann)
            self.assertIn("image_path", ann)
            self.assertIn("aircraft_type", ann)
            self.assertIn("aircraft_pose", ann)
            self.assertIn("camera_position", ann)
            self.assertIn("camera_target", ann)
            self.assertIn("image_size", ann)

            # Check unified pose structure
            aircraft_pose = ann["aircraft_pose"]
            self.assertIn("position", aircraft_pose)
            self.assertIn("rotation", aircraft_pose)

            # Check rotation components
            rotation = aircraft_pose["rotation"]
            required_rotation_keys = ["pitch", "yaw", "roll"]
            for key in required_rotation_keys:
                self.assertIn(key, rotation)

            # Check position is list of 3 coordinates
            self.assertEqual(len(aircraft_pose["position"]), 3)

    def test_annotation_format_coco(self):
        # Test COCO annotation format generation
        small_dataset = Dataset2D(aircraft_types=["F15"], num_samples=2, image_size=(32, 32))

        results = small_dataset.generate(
            output_dir=self.temp_dir, split_ratios=(1.0, 0.0, 0.0), annotation_format="coco"
        )

        # Load and validate COCO annotations
        coco_file = os.path.join(self.temp_dir, "train_coco.json")
        with open(coco_file) as f:
            coco_data = json.load(f)

        # Check COCO structure
        self.assertIn("info", coco_data)
        self.assertIn("categories", coco_data)
        self.assertIn("images", coco_data)
        self.assertIn("annotations", coco_data)

        self.assertEqual(len(coco_data["images"]), 2)

    def test_multiple_aircraft_types(self):
        # Test dataset generation with multiple aircraft types
        multi_dataset = Dataset2D(
            aircraft_types=["F15", "B52", "C130"], num_samples=6, image_size=(32, 32)
        )

        _results = multi_dataset.generate(
            output_dir=self.temp_dir, split_ratios=(1.0, 0.0, 0.0), annotation_format="custom"
        )

        # Load annotations and check aircraft type distribution
        ann_file = os.path.join(self.temp_dir, "train_annotations.json")
        with open(ann_file) as f:
            annotations = json.load(f)

        aircraft_types_found = set(ann["aircraft_type"] for ann in annotations)

        # Should have at least one of each type (probabilistically)
        # Note: This test might occasionally fail due to randomness
        # In practice, with 6 samples and 3 types, we should see variety
        self.assertGreater(len(aircraft_types_found), 1)


if __name__ == "__main__":
    unittest.main()
