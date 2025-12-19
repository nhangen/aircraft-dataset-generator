# Military aircraft model definitions

from typing import Dict, List, Tuple


class BaseAircraft:
    # Base class for aircraft models

    def __init__(self, name: str, aircraft_type: str):
        self.name = name
        self.aircraft_type = aircraft_type
        self.silhouette_points = []

    def get_silhouette_for_pose(self, pose: Dict) -> List[Tuple[float, float]]:
        # Get aircraft silhouette points for given pose
        import math

        # Get pose parameters
        pitch = math.radians(pose.get("pitch", 0))
        yaw = math.radians(pose.get("yaw", 0))
        roll = math.radians(pose.get("roll", 0))
        scale_x = pose.get("z", 1.0)  # Use z as scale factor

        transformed_points = []

        for x, y in self.silhouette_points:
            # Apply basic 2D rotation (primarily yaw)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)

            # Rotate point
            rotated_x = x * cos_yaw - y * sin_yaw
            rotated_y = x * sin_yaw + y * cos_yaw

            # Apply pitch scaling (affects apparent height)
            pitch_scale = abs(math.cos(pitch))
            rotated_y *= pitch_scale

            # Apply roll effect (skew the silhouette)
            roll_offset = rotated_y * math.sin(roll) * 0.3
            rotated_x += roll_offset

            # Apply scale
            rotated_x *= scale_x
            rotated_y *= scale_x

            transformed_points.append((rotated_x, rotated_y))

        return transformed_points


class F15Fighter(BaseAircraft):
    # F-15 Eagle fighter aircraft

    def __init__(self):
        super().__init__("F-15 Eagle", "fighter")
        self.silhouette_points = [
            # Detailed F-15 silhouette with wings and twin tails
            # Nose
            (1.0, 0.0),
            (0.9, 0.03),
            (0.8, 0.05),
            # Cockpit and forward fuselage
            (0.6, 0.08),
            (0.4, 0.10),
            (0.2, 0.12),
            # Wing leading edge
            (0.1, 0.35),
            (-0.1, 0.45),
            (-0.3, 0.42),
            # Wing trailing edge
            (-0.4, 0.25),
            (-0.5, 0.15),
            # Twin tail (right)
            (-0.7, 0.12),
            (-0.8, 0.25),
            (-0.85, 0.22),
            (-0.9, 0.08),
            # Rear fuselage
            (-0.95, 0.0),
            # Twin tail (left) - mirror
            (-0.9, -0.08),
            (-0.85, -0.22),
            (-0.8, -0.25),
            (-0.7, -0.12),
            # Wing trailing edge (left)
            (-0.5, -0.15),
            (-0.4, -0.25),
            # Wing leading edge (left)
            (-0.3, -0.42),
            (-0.1, -0.45),
            (0.1, -0.35),
            # Forward fuselage (left)
            (0.2, -0.12),
            (0.4, -0.10),
            (0.6, -0.08),
            # Nose (left)
            (0.8, -0.05),
            (0.9, -0.03),
        ]


class B52Bomber(BaseAircraft):
    # B-52 Stratofortress bomber aircraft

    def __init__(self):
        super().__init__("B-52 Stratofortress", "bomber")
        self.silhouette_points = [
            # Long B-52 bomber with swept wings and distinctive shape
            # Nose
            (1.0, 0.0),
            (0.9, 0.02),
            (0.8, 0.04),
            # Long fuselage
            (0.6, 0.06),
            (0.4, 0.07),
            (0.2, 0.08),
            (0.0, 0.08),
            # Swept wing (right side)
            (-0.1, 0.25),
            (-0.3, 0.55),
            (-0.5, 0.65),
            (-0.7, 0.60),
            # Wing trailing edge
            (-0.6, 0.35),
            (-0.5, 0.12),
            # Rear fuselage and tail
            (-0.7, 0.08),
            (-0.8, 0.06),
            (-0.85, 0.15),
            (-0.9, 0.12),
            (-0.95, 0.0),
            # Mirror for left side
            (-0.9, -0.12),
            (-0.85, -0.15),
            (-0.8, -0.06),
            (-0.7, -0.08),
            # Wing trailing edge (left)
            (-0.5, -0.12),
            (-0.6, -0.35),
            # Swept wing (left side)
            (-0.7, -0.60),
            (-0.5, -0.65),
            (-0.3, -0.55),
            (-0.1, -0.25),
            # Fuselage (left)
            (0.0, -0.08),
            (0.2, -0.08),
            (0.4, -0.07),
            (0.6, -0.06),
            # Nose (left)
            (0.8, -0.04),
            (0.9, -0.02),
        ]


class C130Transport(BaseAircraft):
    # C-130 Hercules transport aircraft

    def __init__(self):
        super().__init__("C-130 Hercules", "transport")
        self.silhouette_points = [
            # High-wing C-130 transport with distinctive shape
            # Nose
            (0.9, 0.0),
            (0.8, 0.02),
            (0.7, 0.04),
            # Forward fuselage
            (0.5, 0.06),
            (0.3, 0.08),
            (0.1, 0.09),
            # High wing (right side)
            (0.2, 0.25),
            (0.0, 0.40),
            (-0.2, 0.45),
            (-0.4, 0.42),
            (-0.5, 0.35),
            # Wing trailing edge
            (-0.3, 0.15),
            (-0.2, 0.10),
            # Rear fuselage
            (-0.4, 0.09),
            (-0.6, 0.08),
            # T-tail vertical
            (-0.7, 0.06),
            (-0.75, 0.25),
            (-0.8, 0.22),
            (-0.85, 0.0),
            # T-tail horizontal
            (-0.82, 0.0),
            (-0.8, -0.22),
            (-0.75, -0.25),
            (-0.7, -0.06),
            # Rear fuselage (left)
            (-0.6, -0.08),
            (-0.4, -0.09),
            # Wing trailing edge (left)
            (-0.2, -0.10),
            (-0.3, -0.15),
            # High wing (left side)
            (-0.5, -0.35),
            (-0.4, -0.42),
            (-0.2, -0.45),
            (0.0, -0.40),
            (0.2, -0.25),
            # Forward fuselage (left)
            (0.1, -0.09),
            (0.3, -0.08),
            (0.5, -0.06),
            # Nose (left)
            (0.7, -0.04),
            (0.8, -0.02),
        ]
