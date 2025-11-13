"""Unit tests for aircraft model definitions"""

import unittest

from aircraft_toolkit.models.military import B52Bomber, BaseAircraft, C130Transport, F15Fighter


class TestBaseAircraft(unittest.TestCase):
    """Test base aircraft functionality"""

    def test_base_aircraft_initialization(self):
        """Test base aircraft can be initialized"""
        aircraft = BaseAircraft("Test Aircraft", "test")
        self.assertEqual(aircraft.name, "Test Aircraft")
        self.assertEqual(aircraft.aircraft_type, "test")
        self.assertEqual(aircraft.silhouette_points, [])

    def test_get_silhouette_for_pose(self):
        """Test silhouette generation with pose"""
        aircraft = BaseAircraft("Test", "test")
        aircraft.silhouette_points = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]

        pose = {"pitch": 0, "yaw": 0, "roll": 0, "x": 0, "y": 0, "z": 1}
        silhouette = aircraft.get_silhouette_for_pose(pose)

        self.assertEqual(len(silhouette), 3)
        self.assertIsInstance(silhouette, list)


class TestMilitaryAircraft(unittest.TestCase):
    """Test military aircraft models"""

    def test_f15_fighter_initialization(self):
        """Test F-15 fighter model"""
        f15 = F15Fighter()
        self.assertEqual(f15.name, "F-15 Eagle")
        self.assertEqual(f15.aircraft_type, "fighter")
        self.assertGreater(len(f15.silhouette_points), 5)

        # Check that silhouette points are tuples of floats
        for point in f15.silhouette_points:
            self.assertIsInstance(point, tuple)
            self.assertEqual(len(point), 2)
            self.assertIsInstance(point[0], (int, float))
            self.assertIsInstance(point[1], (int, float))

    def test_b52_bomber_initialization(self):
        """Test B-52 bomber model"""
        b52 = B52Bomber()
        self.assertEqual(b52.name, "B-52 Stratofortress")
        self.assertEqual(b52.aircraft_type, "bomber")
        self.assertGreater(len(b52.silhouette_points), 5)

    def test_c130_transport_initialization(self):
        """Test C-130 transport model"""
        c130 = C130Transport()
        self.assertEqual(c130.name, "C-130 Hercules")
        self.assertEqual(c130.aircraft_type, "transport")
        self.assertGreater(len(c130.silhouette_points), 5)

    def test_aircraft_silhouette_differences(self):
        """Test that different aircraft have different silhouettes"""
        f15 = F15Fighter()
        b52 = B52Bomber()
        c130 = C130Transport()

        # Silhouettes should be different
        self.assertNotEqual(f15.silhouette_points, b52.silhouette_points)
        self.assertNotEqual(f15.silhouette_points, c130.silhouette_points)
        self.assertNotEqual(b52.silhouette_points, c130.silhouette_points)

    def test_silhouette_bounds(self):
        """Test that silhouette points are within reasonable bounds"""
        aircraft_models = [F15Fighter(), B52Bomber(), C130Transport()]

        for aircraft in aircraft_models:
            for x, y in aircraft.silhouette_points:
                # Points should be roughly within [-1, 1] range
                self.assertGreaterEqual(x, -1.5)
                self.assertLessEqual(x, 1.5)
                self.assertGreaterEqual(y, -1.5)
                self.assertLessEqual(y, 1.5)


if __name__ == "__main__":
    unittest.main()
