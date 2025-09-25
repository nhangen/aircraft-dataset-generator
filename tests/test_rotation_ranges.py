#!/usr/bin/env python3
"""
Unit tests for rotation range functionality in Dataset3D.
Tests the fix for the 120° convergence barrier.
"""

import unittest
import numpy as np
import os
import json
import tempfile
import shutil
from aircraft_toolkit.core.dataset_3d import Dataset3D
from aircraft_toolkit.config import get_config


class TestRotationRanges(unittest.TestCase):
    """Test rotation range parameters in Dataset3D"""

    def setUp(self):
        """Set up test configuration"""
        self.config = get_config()
        # Use basic provider for faster tests
        self.config.aircraft.model_provider = 'basic'
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_rotation_ranges(self):
        """Test that default rotation ranges match original constraints"""
        dataset = Dataset3D(
            aircraft_types=['F15'],
            num_scenes=5,
            views_per_scene=2
        )

        # Generate small dataset to test rotation ranges
        dataset.generate(self.temp_dir, split_ratios=(1.0, 0.0, 0.0))

        # Check annotations
        annotation_file = os.path.join(self.temp_dir, 'train_3d_annotations.json')
        self.assertTrue(os.path.exists(annotation_file))

        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        rotations = []
        for ann in annotations:
            pose = ann['aircraft_pose']['rotation']
            rotations.append([pose['pitch'], pose['yaw'], pose['roll']])

        rotations = np.array(rotations)

        # Test original constraints (within tolerance for randomness)
        self.assertGreaterEqual(rotations[:, 0].min(), -30.1)  # pitch >= -30
        self.assertLessEqual(rotations[:, 0].max(), 30.1)      # pitch <= 30
        self.assertGreaterEqual(rotations[:, 1].min(), -180.1) # yaw >= -180
        self.assertLessEqual(rotations[:, 1].max(), 180.1)     # yaw <= 180
        self.assertGreaterEqual(rotations[:, 2].min(), -15.1)  # roll >= -15
        self.assertLessEqual(rotations[:, 2].max(), 15.1)      # roll <= 15

    def test_custom_rotation_ranges(self):
        """Test custom rotation ranges (the fix for 120° barrier)"""
        dataset = Dataset3D(
            aircraft_types=['F15'],
            num_scenes=10,  # More samples for better range testing
            views_per_scene=2,
            pitch_range=(-90, 90),    # Full pitch
            roll_range=(-180, 180),   # Full roll
            yaw_range=(-90, 90)       # Custom yaw
        )

        dataset.generate(self.temp_dir, split_ratios=(1.0, 0.0, 0.0))

        # Check annotations
        annotation_file = os.path.join(self.temp_dir, 'train_3d_annotations.json')
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        rotations = []
        for ann in annotations:
            pose = ann['aircraft_pose']['rotation']
            rotations.append([pose['pitch'], pose['yaw'], pose['roll']])

        rotations = np.array(rotations)

        # Test expanded constraints
        self.assertGreaterEqual(rotations[:, 0].min(), -90.1)  # pitch >= -90
        self.assertLessEqual(rotations[:, 0].max(), 90.1)      # pitch <= 90
        self.assertGreaterEqual(rotations[:, 1].min(), -90.1)  # yaw >= -90
        self.assertLessEqual(rotations[:, 1].max(), 90.1)      # yaw <= 90
        self.assertGreaterEqual(rotations[:, 2].min(), -180.1) # roll >= -180
        self.assertLessEqual(rotations[:, 2].max(), 180.1)     # roll <= 180

        # Verify we're actually getting expanded ranges (not just within them)
        pitch_range = rotations[:, 0].max() - rotations[:, 0].min()
        roll_range = rotations[:, 2].max() - rotations[:, 2].min()

        # Should be much larger than original constraints
        self.assertGreater(pitch_range, 60)   # Much larger than original ±30° = 60° range
        self.assertGreater(roll_range, 60)    # Much larger than original ±15° = 30° range

    def test_rotation_range_validation(self):
        """Test that rotation ranges are properly stored and used"""
        pitch_range = (-45, 45)
        roll_range = (-30, 30)
        yaw_range = (-120, 120)

        dataset = Dataset3D(
            aircraft_types=['F15'],
            num_scenes=1,
            views_per_scene=1,
            pitch_range=pitch_range,
            roll_range=roll_range,
            yaw_range=yaw_range
        )

        # Test that ranges are stored correctly
        self.assertEqual(dataset.pitch_range, pitch_range)
        self.assertEqual(dataset.roll_range, roll_range)
        self.assertEqual(dataset.yaw_range, yaw_range)

    def test_extreme_rotation_ranges(self):
        """Test extreme rotation ranges for edge cases"""
        dataset = Dataset3D(
            aircraft_types=['F15'],
            num_scenes=3,
            views_per_scene=1,
            pitch_range=(-89, 89),    # Near gimbal lock
            roll_range=(-179, 179),   # Near full rotation
            yaw_range=(0, 360)        # Full rotation
        )

        # Should not crash
        dataset.generate(self.temp_dir, split_ratios=(1.0, 0.0, 0.0))

        annotation_file = os.path.join(self.temp_dir, 'train_3d_annotations.json')
        self.assertTrue(os.path.exists(annotation_file))

    def test_barrier_breaking_ranges(self):
        """Test the specific ranges designed to break the 120° barrier"""
        # These are the exact ranges from the solution
        dataset = Dataset3D(
            aircraft_types=['F15', 'B52', 'C130'],
            num_scenes=30,  # More samples for better statistical coverage
            views_per_scene=2,
            pitch_range=(-90, 90),    # 3x expansion from ±30°
            roll_range=(-180, 180),   # 12x expansion from ±15°
            yaw_range=(-180, 180)     # Full coverage (unchanged)
        )

        dataset.generate(self.temp_dir, split_ratios=(1.0, 0.0, 0.0))

        annotation_file = os.path.join(self.temp_dir, 'train_3d_annotations.json')
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        # Should have all aircraft types
        aircraft_types = set(ann['aircraft_type'] for ann in annotations)
        self.assertIn('F15', aircraft_types)
        self.assertIn('B52', aircraft_types)
        self.assertIn('C130', aircraft_types)

        # Check that we have significantly expanded rotation coverage
        rotations = []
        for ann in annotations:
            pose = ann['aircraft_pose']['rotation']
            rotations.append([pose['pitch'], pose['yaw'], pose['roll']])

        rotations = np.array(rotations)

        # Test that rotations can go beyond original constraints
        pitch_range = rotations[:, 0].max() - rotations[:, 0].min()
        roll_range = rotations[:, 2].max() - rotations[:, 2].min()

        # Original constraints: pitch ±30° = 60°, roll ±15° = 30°
        original_pitch_range = 60
        original_roll_range = 30

        # Test that we're actually using the expanded ranges
        # Should be able to exceed original constraints
        self.assertTrue(
            rotations[:, 0].max() > 30 or rotations[:, 0].min() < -30,
            "Pitch should exceed original ±30° constraint"
        )
        self.assertTrue(
            rotations[:, 2].max() > 15 or rotations[:, 2].min() < -15,
            "Roll should exceed original ±15° constraint"
        )

        # Should cover more range than original (allowing for statistical variation)
        self.assertGreater(pitch_range, 0.8 * original_pitch_range,
                          f"Pitch range {pitch_range:.1f}° should use most of expanded range")
        self.assertGreater(roll_range, 1.5 * original_roll_range,
                          f"Roll range {roll_range:.1f}° should significantly exceed original")

        print(f"✅ Barrier-breaking test passed:")
        print(f"   Pitch range: {pitch_range:.1f}° (original: {original_pitch_range}°)")
        print(f"   Roll range: {roll_range:.1f}° (original: {original_roll_range}°)")
        print(f"   Pitch bounds: {rotations[:, 0].min():.1f}° to {rotations[:, 0].max():.1f}°")
        print(f"   Roll bounds: {rotations[:, 2].min():.1f}° to {rotations[:, 2].max():.1f}°")


class TestRotationDistribution(unittest.TestCase):
    """Test that rotations are uniformly distributed within ranges"""

    def setUp(self):
        self.config = get_config()
        self.config.aircraft.model_provider = 'basic'
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_uniform_distribution(self):
        """Test that rotations are uniformly distributed"""
        dataset = Dataset3D(
            aircraft_types=['F15'],
            num_scenes=50,  # More samples for statistical testing
            views_per_scene=1,
            pitch_range=(-60, 60),
            roll_range=(-90, 90),
            yaw_range=(-180, 180)
        )

        dataset.generate(self.temp_dir, split_ratios=(1.0, 0.0, 0.0))

        annotation_file = os.path.join(self.temp_dir, 'train_3d_annotations.json')
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        rotations = []
        for ann in annotations:
            pose = ann['aircraft_pose']['rotation']
            rotations.append([pose['pitch'], pose['yaw'], pose['roll']])

        rotations = np.array(rotations)

        # Test that rotations cover the full range reasonably well
        # Not a perfect uniform test, but checks that we're using the full range
        pitch_range = rotations[:, 0].max() - rotations[:, 0].min()
        roll_range = rotations[:, 2].max() - rotations[:, 2].min()

        # Should cover at least 80% of the available range
        self.assertGreater(pitch_range, 0.8 * 120)  # 80% of (-60, 60) = 96°
        self.assertGreater(roll_range, 0.8 * 180)   # 80% of (-90, 90) = 144°


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)