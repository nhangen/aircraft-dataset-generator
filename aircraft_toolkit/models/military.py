"""Military aircraft model definitions"""

import math
from typing import List, Tuple, Dict


class BaseAircraft:
    """Base class for aircraft models"""
    
    def __init__(self, name: str, aircraft_type: str):
        self.name = name
        self.aircraft_type = aircraft_type
        self.silhouette_points = []
    
    def get_silhouette_for_pose(self, pose: Dict) -> List[Tuple[float, float]]:
        """Get aircraft silhouette points for given pose"""
        # Simplified: return base silhouette (rotation logic would go here)
        return self.silhouette_points


class F15Fighter(BaseAircraft):
    """F-15 Eagle fighter aircraft"""
    
    def __init__(self):
        super().__init__("F-15 Eagle", "fighter")
        self.silhouette_points = [
            # Fuselage outline (simplified)
            (0.8, 0.0),    # Nose
            (0.6, 0.08),   # Upper nose
            (0.2, 0.12),   # Forward fuselage top
            (-0.4, 0.12),  # Rear fuselage top
            (-0.8, 0.06),  # Tail top
            (-0.9, 0.0),   # Tail point
            (-0.8, -0.06), # Tail bottom
            (-0.4, -0.12), # Rear fuselage bottom
            (0.2, -0.12),  # Forward fuselage bottom
            (0.6, -0.08),  # Lower nose
        ]


class B52Bomber(BaseAircraft):
    """B-52 Stratofortress bomber aircraft"""
    
    def __init__(self):
        super().__init__("B-52 Stratofortress", "bomber")
        self.silhouette_points = [
            # Long, swept-wing bomber silhouette
            (0.9, 0.0),    # Nose
            (0.7, 0.06),   # Upper nose
            (0.0, 0.08),   # Wing root top
            (-0.6, 0.06),  # Rear fuselage top
            (-0.8, 0.04),  # Tail top
            (-0.9, 0.0),   # Tail point
            (-0.8, -0.04), # Tail bottom
            (-0.6, -0.06), # Rear fuselage bottom
            (0.0, -0.08),  # Wing root bottom
            (0.7, -0.06),  # Lower nose
        ]


class C130Transport(BaseAircraft):
    """C-130 Hercules transport aircraft"""
    
    def __init__(self):
        super().__init__("C-130 Hercules", "transport")
        self.silhouette_points = [
            # High-wing transport silhouette
            (0.7, 0.0),    # Nose
            (0.5, 0.08),   # Upper nose
            (0.1, 0.15),   # High wing position
            (-0.3, 0.15),  # Wing trailing edge
            (-0.5, 0.12),  # Rear fuselage top
            (-0.7, 0.06),  # Tail top
            (-0.8, 0.0),   # Tail point
            (-0.7, -0.06), # Tail bottom
            (-0.5, -0.12), # Rear fuselage bottom
            (0.1, -0.12),  # Wing root bottom
            (0.5, -0.08),  # Lower nose
        ]