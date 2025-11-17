"""Military aircraft model definitions.

This module provides 2D silhouette representations of military aircraft for
dataset generation. Each aircraft model contains a set of 2D points that define
its silhouette profile, which can be transformed based on pose parameters.

Supported Aircraft:
    - F-15 Eagle: Twin-engine fighter with distinctive twin tails
    - B-52 Stratofortress: Strategic bomber with swept wings
    - C-130 Hercules: High-wing transport aircraft with T-tail

Coordinate System:
    - X-axis: Forward/aft (1.0 = nose, -1.0 = tail)
    - Y-axis: Left/right (positive = right, negative = left)
    - Origin: Aircraft center of mass
    - Units: Normalized [-1.0, 1.0] range
"""


class BaseAircraft:
    """Base class for all aircraft models.

    Provides common functionality for aircraft silhouette generation and
    pose-based transformations. Each aircraft is defined by a set of 2D
    silhouette points that outline its shape.

    Attributes:
        name: Human-readable aircraft name (e.g., "F-15 Eagle").
        aircraft_type: Aircraft category (e.g., "fighter", "bomber", "transport").
        silhouette_points: List of (x, y) tuples defining the aircraft outline
            in normalized coordinates. Points should form a closed polygon
            representing the top-down view silhouette.
    """

    def __init__(self, name: str, aircraft_type: str):
        """Initialize aircraft model.

        Args:
            name: Human-readable aircraft name.
            aircraft_type: Aircraft category identifier.
        """
        self.name = name
        self.aircraft_type = aircraft_type
        self.silhouette_points = []

    def get_silhouette_for_pose(self, pose: dict) -> list[tuple[float, float]]:
        """Transform aircraft silhouette based on pose parameters.

        Applies 3D rotation transformations (pitch, yaw, roll) to the 2D
        silhouette points to simulate aircraft orientation. The transformation
        uses approximate 2D projections of 3D rotations for efficient rendering.

        Transformation order:
            1. Yaw rotation (heading change)
            2. Pitch scaling (nose up/down affects apparent height)
            3. Roll effect (banking creates lateral skew)
            4. Scale application (distance from camera)

        Args:
            pose: Dictionary containing pose parameters:
                - pitch: Nose up/down angle in degrees [-90, 90].
                  Positive = nose up. Affects apparent vertical size.
                - yaw: Left/right heading in degrees [-180, 180].
                  Positive = nose right. Primary rotation axis.
                - roll: Banking angle in degrees [-180, 180].
                  Positive = right wing down. Creates lateral skew.
                - z: Scale factor for distance simulation [0.0, inf].
                  Default is 1.0. Larger values = closer to camera.

        Returns:
            List of transformed (x, y) coordinate tuples representing the
            aircraft silhouette in the new pose. Points maintain the same
            order as the original silhouette_points.

        Example:
            >>> aircraft = F15Fighter()
            >>> pose = {'pitch': 15, 'yaw': 45, 'roll': 0, 'z': 1.0}
            >>> points = aircraft.get_silhouette_for_pose(pose)
            >>> len(points) == len(aircraft.silhouette_points)
            True

        Note:
            This uses approximate 2D projections rather than full 3D transformations
            for performance. Results are visually accurate for typical viewing angles
            but may not be geometrically precise for extreme orientations.
        """
        import math

        # Extract and convert pose parameters to radians
        pitch = math.radians(pose.get("pitch", 0))
        yaw = math.radians(pose.get("yaw", 0))
        roll = math.radians(pose.get("roll", 0))
        scale_x = pose.get("z", 1.0)  # Z-distance used as scale factor

        transformed_points = []

        for x, y in self.silhouette_points:
            # Apply yaw rotation (primary heading rotation)
            # Standard 2D rotation matrix: [cos -sin; sin cos]
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)

            # Rotate point around origin
            rotated_x = x * cos_yaw - y * sin_yaw
            rotated_y = x * sin_yaw + y * cos_yaw

            # Apply pitch effect (nose up/down)
            # Pitch affects apparent height through foreshortening
            # Use cosine scaling: 0° = full height, ±90° = minimal height
            pitch_scale = abs(math.cos(pitch))
            rotated_y *= pitch_scale

            # Apply roll effect (banking)
            # Roll creates lateral skew - right wing down shifts right side left
            # Scale factor 0.3 provides subtle but visible roll indication
            roll_offset = rotated_y * math.sin(roll) * 0.3
            rotated_x += roll_offset

            # Apply distance scaling
            # Larger z values make aircraft appear closer/larger
            rotated_x *= scale_x
            rotated_y *= scale_x

            transformed_points.append((rotated_x, rotated_y))

        return transformed_points


class F15Fighter(BaseAircraft):
    """F-15 Eagle twin-engine fighter aircraft.

    The F-15 is a tactical fighter aircraft with distinctive twin vertical
    tails and large delta wings. The silhouette captures its swept-wing design
    and dual tail configuration.

    Characteristics:
        - Twin vertical stabilizers (tails)
        - Swept delta wings
        - Twin-engine exhaust
        - Pointed nose cone
        - Wide stance for maneuverability

    Silhouette Points: 28 points defining the aircraft outline in top-down view.
    """

    def __init__(self):
        """Initialize F-15 Eagle fighter model."""
        super().__init__("F-15 Eagle", "fighter")
        # Silhouette points form a closed polygon outlining F-15 top view
        # Points ordered clockwise starting from nose, tracing right side,
        # then left side back to nose
        self.silhouette_points = [
            # Nose section (pointed forward)
            (1.0, 0.0),  # Nose tip
            (0.9, 0.03),
            (0.8, 0.05),
            # Cockpit and forward fuselage
            (0.6, 0.08),
            (0.4, 0.10),
            (0.2, 0.12),
            # Wing leading edge (swept back)
            (0.1, 0.35),  # Wing root
            (-0.1, 0.45),  # Wing max width
            (-0.3, 0.42),
            # Wing trailing edge
            (-0.4, 0.25),
            (-0.5, 0.15),
            # Twin tail (right stabilizer)
            (-0.7, 0.12),
            (-0.8, 0.25),  # Tail height
            (-0.85, 0.22),
            (-0.9, 0.08),
            # Rear fuselage center
            (-0.95, 0.0),
            # Twin tail (left stabilizer) - mirror image
            (-0.9, -0.08),
            (-0.85, -0.22),
            (-0.8, -0.25),  # Tail height
            (-0.7, -0.12),
            # Wing trailing edge (left)
            (-0.5, -0.15),
            (-0.4, -0.25),
            # Wing leading edge (left, swept back)
            (-0.3, -0.42),
            (-0.1, -0.45),  # Wing max width
            (0.1, -0.35),  # Wing root
            # Forward fuselage (left)
            (0.2, -0.12),
            (0.4, -0.10),
            (0.6, -0.08),
            # Nose (left side)
            (0.8, -0.05),
            (0.9, -0.03),
        ]


class B52Bomber(BaseAircraft):
    """B-52 Stratofortress strategic bomber aircraft.

    The B-52 is a long-range strategic bomber with a distinctive swept-wing
    design and eight engines. The silhouette emphasizes its wide wingspan
    and elongated fuselage characteristic of heavy bombers.

    Characteristics:
        - Extreme wingspan with swept wings
        - Long fuselage for bomb payload
        - Eight engines (not individually modeled in silhouette)
        - Vertical stabilizer (single tall tail)
        - Low-slung design

    Silhouette Points: 30 points defining the aircraft outline in top-down view.
    """

    def __init__(self):
        """Initialize B-52 Stratofortress bomber model."""
        super().__init__("B-52 Stratofortress", "bomber")
        # Silhouette emphasizes the B-52's characteristic wide, swept wings
        # Points ordered clockwise starting from nose
        self.silhouette_points = [
            # Nose section
            (1.0, 0.0),  # Nose tip
            (0.9, 0.02),
            (0.8, 0.04),
            # Long fuselage (bomber payload area)
            (0.6, 0.06),
            (0.4, 0.07),
            (0.2, 0.08),
            (0.0, 0.08),
            # Swept wing (right side) - very wide wingspan
            (-0.1, 0.25),  # Wing root
            (-0.3, 0.55),  # Mid-wing
            (-0.5, 0.65),  # Wing tip (maximum span)
            (-0.7, 0.60),
            # Wing trailing edge
            (-0.6, 0.35),
            (-0.5, 0.12),
            # Rear fuselage and tail section
            (-0.7, 0.08),
            (-0.8, 0.06),
            (-0.85, 0.15),  # Vertical stabilizer
            (-0.9, 0.12),
            (-0.95, 0.0),  # Tail end
            # Left side (mirror image)
            (-0.9, -0.12),
            (-0.85, -0.15),  # Vertical stabilizer
            (-0.8, -0.06),
            (-0.7, -0.08),
            # Wing trailing edge (left)
            (-0.5, -0.12),
            (-0.6, -0.35),
            # Swept wing (left side)
            (-0.7, -0.60),
            (-0.5, -0.65),  # Wing tip (maximum span)
            (-0.3, -0.55),  # Mid-wing
            (-0.1, -0.25),  # Wing root
            # Fuselage (left)
            (0.0, -0.08),
            (0.2, -0.08),
            (0.4, -0.07),
            (0.6, -0.06),
            # Nose (left side)
            (0.8, -0.04),
            (0.9, -0.02),
        ]


class C130Transport(BaseAircraft):
    """C-130 Hercules military transport aircraft.

    The C-130 is a tactical transport aircraft with a distinctive high-wing
    design and T-tail configuration. The silhouette captures its characteristic
    boxy fuselage and straight wings optimized for cargo capacity.

    Characteristics:
        - High-mounted straight wings (for ground clearance)
        - T-tail configuration (horizontal tail on vertical stabilizer)
        - Four turboprop engines (not individually modeled)
        - Wide fuselage for cargo
        - Rear loading ramp capability

    Silhouette Points: 30 points defining the aircraft outline in top-down view.
    """

    def __init__(self):
        """Initialize C-130 Hercules transport model."""
        super().__init__("C-130 Hercules", "transport")
        # Silhouette emphasizes the C-130's high-wing and T-tail design
        # Points ordered clockwise starting from nose
        self.silhouette_points = [
            # Nose section
            (0.9, 0.0),  # Nose tip
            (0.8, 0.02),
            (0.7, 0.04),
            # Forward fuselage
            (0.5, 0.06),
            (0.3, 0.08),
            (0.1, 0.09),
            # High wing (right side) - relatively straight for cargo aircraft
            (0.2, 0.25),  # Wing root (high position)
            (0.0, 0.40),  # Wing mid-section
            (-0.2, 0.45),  # Wing maximum span
            (-0.4, 0.42),
            (-0.5, 0.35),
            # Wing trailing edge
            (-0.3, 0.15),
            (-0.2, 0.10),
            # Rear fuselage
            (-0.4, 0.09),
            (-0.6, 0.08),
            # T-tail vertical stabilizer
            (-0.7, 0.06),
            (-0.75, 0.25),  # Top of vertical tail
            (-0.8, 0.22),  # Horizontal tail on T-tail
            (-0.85, 0.0),  # Tail end
            # T-tail horizontal stabilizer
            (-0.82, 0.0),
            (-0.8, -0.22),  # Horizontal tail on T-tail
            (-0.75, -0.25),  # Top of vertical tail
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
            (-0.2, -0.45),  # Wing maximum span
            (0.0, -0.40),  # Wing mid-section
            (0.2, -0.25),  # Wing root (high position)
            # Forward fuselage (left)
            (0.1, -0.09),
            (0.3, -0.08),
            (0.5, -0.06),
            # Nose (left side)
            (0.7, -0.04),
            (0.8, -0.02),
        ]
