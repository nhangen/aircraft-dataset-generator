#!/usr/bin/env python3
"""
Sample code for integrating realistic 3D aircraft model libraries
with the Aircraft Dataset Generator.

This file demonstrates how to use different libraries to create
better quality aircraft models than the current hand-coded vertices.
"""

import numpy as np
from typing import Tuple, List


# ==============================================================================
# Option 1: CadQuery Integration (Recommended for ease of use)
# ==============================================================================
def cadquery_f15_example():
    """
    Create a realistic F-15 fighter using CadQuery's parametric modeling.
    Install: conda install -c cadquery cadquery=2
    """
    try:
        import cadquery as cq

        # Create parametric F-15 fighter
        # Main fuselage with tapered nose
        fuselage = (
            cq.Workplane("XY")
            .moveTo(0, 0)
            .box(8, 1.2, 1, centered=(True, True, True))
            .faces(">X")
            .edges("|Z")
            .fillet(0.3)  # Round the nose
        )

        # Swept wings
        wing = (
            cq.Workplane("XY")
            .center(0, 0)
            .transformed(rotate=(0, 0, -15))  # Sweep angle
            .box(3, 7, 0.2, centered=(True, True, True))
            .edges("|Z")
            .fillet(0.1)
        )

        # Twin vertical stabilizers
        tail_left = (
            cq.Workplane("YZ")
            .center(-0.4, 0)
            .transformed(offset=(-3, 0, 0))
            .transformed(rotate=(15, 0, 0))  # Angle outward
            .rect(1.5, 2)
            .extrude(0.15)
        )

        tail_right = (
            cq.Workplane("YZ")
            .center(0.4, 0)
            .transformed(offset=(-3, 0, 0))
            .transformed(rotate=(-15, 0, 0))  # Angle outward
            .rect(1.5, 2)
            .extrude(0.15)
        )

        # Combine all parts
        f15_model = fuselage.union(wing).union(tail_left).union(tail_right)

        # Export to STL for mesh extraction
        f15_model.exportStl("f15_model.stl")

        # Convert to vertices and faces for your renderer
        # You can use trimesh to load the STL and get vertices/faces
        import trimesh
        mesh = trimesh.load("f15_model.stl")
        vertices = mesh.vertices
        faces = mesh.faces

        return vertices, faces

    except ImportError:
        print("CadQuery not installed. Install with: conda install -c cadquery cadquery=2")
        return None, None


# ==============================================================================
# Option 2: Open3D Integration (Good for mesh processing)
# ==============================================================================
def open3d_aircraft_example():
    """
    Create aircraft using Open3D's mesh operations.
    Install: pip install open3d
    """
    try:
        import open3d as o3d

        # Create fuselage from cylinder
        fuselage = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.5,
            height=8.0,
            resolution=20,
            split=4
        )
        fuselage.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi/2, 0)),
            center=(0, 0, 0)
        )

        # Create wings from boxes
        wing = o3d.geometry.TriangleMesh.create_box(
            width=7.0,
            height=0.2,
            depth=2.0
        )
        wing.translate((-3.5, -0.1, -1.0))

        # Create tail from box
        tail = o3d.geometry.TriangleMesh.create_box(
            width=0.2,
            height=2.0,
            depth=1.5
        )
        tail.translate((-4.0, -1.0, -0.75))

        # Combine meshes
        aircraft = fuselage + wing + tail

        # Smooth the mesh
        aircraft = aircraft.filter_smooth_simple(number_of_iterations=2)
        aircraft.compute_vertex_normals()

        # Convert to numpy arrays
        vertices = np.asarray(aircraft.vertices)
        faces = np.asarray(aircraft.triangles)

        return vertices, faces

    except ImportError:
        print("Open3D not installed. Install with: pip install open3d")
        return None, None


# ==============================================================================
# Option 3: PyMesh Integration (Good for procedural generation)
# ==============================================================================
def pymesh_bomber_example():
    """
    Create a B-52 style bomber using PyMesh procedural generation.
    Install: pip install pymesh-fix
    """
    try:
        import pymesh

        # Create main fuselage (elongated cylinder)
        fuselage = pymesh.generate_cylinder(
            p0=np.array([-5, 0, 0]),
            p1=np.array([5, 0, 0]),
            r0=0.6,
            r1=0.4,
            num_segments=24
        )

        # Create swept wings (using box and transform)
        wing_box = pymesh.generate_box_mesh(
            box_min=np.array([-1, -5, -0.1]),
            box_max=np.array([2, 5, 0.1]),
            num_samples=10
        )

        # Perform CSG operations to combine
        aircraft = pymesh.boolean(fuselage, wing_box, operation="union")

        vertices = aircraft.vertices
        faces = aircraft.faces

        return vertices, faces

    except ImportError:
        print("PyMesh not installed. Install with: pip install pymesh-fix")
        return None, None


# ==============================================================================
# Option 4: TiGL Integration (Best for realistic aircraft)
# ==============================================================================
def tigl_aircraft_example():
    """
    Create aircraft using TiGL's parametric CPACS format.
    This is the most aircraft-specific library.
    Install: conda install dlr-sc::tigl3
    """
    example_code = """
    from tigl3 import tigl3wrapper
    from tigl3.geometry import CTiglPoint

    # TiGL requires CPACS XML files for aircraft definition
    # You would typically load a CPACS file like this:
    tigl_handle = tigl3wrapper.Tigl3()
    tigl_handle.open("aircraft.cpacs.xml")

    # Get wing geometry
    wing = tigl_handle.get_wing(1)
    wing_shape = wing.get_loft()

    # Export to STEP/STL
    tigl_handle.export_step("aircraft.step")

    # Then convert STEP/STL to vertices/faces using trimesh
    """
    print("TiGL example (requires CPACS XML file):")
    print(example_code)
    return None, None


# ==============================================================================
# Integration Helper: Convert any mesh to your format
# ==============================================================================
def integrate_with_dataset_3d(vertices: np.ndarray, faces: np.ndarray):
    """
    Helper to integrate external mesh data with your Dataset3D class.

    This would replace the hand-coded vertices in your aircraft models.
    """
    class ImprovedAircraft3D:
        def __init__(self, name: str, vertices: np.ndarray, faces: np.ndarray):
            self.name = name
            self.vertices = vertices
            self.faces = faces

        def get_mesh(self):
            return self.vertices, self.faces

    # Scale and center the model as needed
    vertices_centered = vertices - np.mean(vertices, axis=0)
    scale_factor = 10.0 / np.max(np.abs(vertices_centered))
    vertices_scaled = vertices_centered * scale_factor

    return ImprovedAircraft3D("F-15", vertices_scaled, faces)


# ==============================================================================
# Comparison with your current implementation
# ==============================================================================
def compare_mesh_quality():
    """
    Compare vertex counts between current and improved methods.
    """
    print("\n=== Mesh Quality Comparison ===\n")

    # Your current implementation
    print("Current Implementation:")
    print("  F-15: ~14 vertices, 19 faces (hand-coded)")
    print("  Quality: Blocky, unrealistic\n")

    # CadQuery approach
    vertices_cq, faces_cq = cadquery_f15_example()
    if vertices_cq is not None:
        print(f"CadQuery F-15:")
        print(f"  Vertices: {len(vertices_cq)}")
        print(f"  Faces: {len(faces_cq)}")
        print(f"  Quality: Smooth NURBS surfaces, realistic aerodynamics\n")

    # Open3D approach
    vertices_o3d, faces_o3d = open3d_aircraft_example()
    if vertices_o3d is not None:
        print(f"Open3D Aircraft:")
        print(f"  Vertices: {len(vertices_o3d)}")
        print(f"  Faces: {len(faces_o3d)}")
        print(f"  Quality: Smooth primitives, good for simple models\n")

    # PyMesh approach
    vertices_pm, faces_pm = pymesh_bomber_example()
    if vertices_pm is not None:
        print(f"PyMesh Bomber:")
        print(f"  Vertices: {len(vertices_pm)}")
        print(f"  Faces: {len(faces_pm)}")
        print(f"  Quality: CSG operations, procedural generation\n")


# ==============================================================================
# Quick start guide
# ==============================================================================
def main():
    print("=" * 60)
    print("IMPROVED 3D AIRCRAFT MODEL GENERATION EXAMPLES")
    print("=" * 60)

    print("\nThese examples show how to replace your current hand-coded")
    print("vertex arrays with realistic 3D models from various libraries.\n")

    print("INSTALLATION COMMANDS:")
    print("-" * 40)
    print("1. CadQuery (RECOMMENDED):")
    print("   conda install -c cadquery cadquery=2")
    print("   pip install trimesh  # for STL loading\n")

    print("2. Open3D:")
    print("   pip install open3d\n")

    print("3. PyMesh:")
    print("   pip install pymesh-fix\n")

    print("4. TiGL (Aircraft-specific):")
    print("   conda install dlr-sc::tigl3\n")

    print("=" * 60)

    # Try to run examples
    compare_mesh_quality()

    print("\nTo integrate with your Dataset3D class:")
    print("-" * 40)
    print("1. Choose a library above and install it")
    print("2. Replace the _create_mesh() methods in dataset_3d.py")
    print("3. Use the integrate_with_dataset_3d() helper function")
    print("4. Your renders will now use high-quality meshes!\n")


if __name__ == "__main__":
    main()