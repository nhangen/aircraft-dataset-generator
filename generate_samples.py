#!/usr/bin/env python3
"""
Generate sample images for documentation and testing
Organized by dimension then aircraft type
"""

import os
import shutil

from aircraft_toolkit.core.dataset_2d import Dataset2D
from aircraft_toolkit.core.dataset_3d import Dataset3D


def generate_2d_samples_by_type():
    """Generate 2D sample images - 10 per aircraft type"""
    print("🛩️  Generating 2D samples by aircraft type...")

    aircraft_types = ['F15', 'B52', 'C130']

    for aircraft_type in aircraft_types:
        print(f"   Generating 2D {aircraft_type} samples...")

        generator = Dataset2D(
            aircraft_types=[aircraft_type],
            num_samples=10,
            image_size=(512, 512),
            pose_range={
                'pitch': (-30, 30),
                'yaw': (-180, 180),
                'roll': (-15, 15)
            }
        )

        # Generate to temp directory
        temp_dir = f'temp_2d_{aircraft_type}'
        generator.generate(
            output_dir=temp_dir,
            split_ratios=(1.0, 0.0, 0.0),
            annotation_format='custom',
            num_workers=1
        )

        # Move images to organized structure
        src_dir = f'{temp_dir}/train/images'
        dst_dir = f'sample_images/2d/{aircraft_type}'
        if os.path.exists(src_dir):
            for img in os.listdir(src_dir):
                shutil.copy2(f'{src_dir}/{img}', f'{dst_dir}/{img}')

        # Clean up temp directory
        shutil.rmtree(temp_dir)

    print("✅ Generated 2D samples organized by aircraft type")


def generate_3d_samples_by_type():
    """Generate 3D sample images - 10 per aircraft type"""
    print("🛩️  Generating 3D samples by aircraft type...")

    aircraft_types = ['F15', 'B52', 'C130']

    for aircraft_type in aircraft_types:
        print(f"   Generating 3D {aircraft_type} samples...")

        generator = Dataset3D(
            aircraft_types=[aircraft_type],
            num_scenes=4,  # 4 scenes × 3 views = 12 images per type
            views_per_scene=3,
            image_size=(512, 512),
            include_depth_maps=False  # Skip depth for samples
        )

        # Generate to temp directory
        temp_dir = f'temp_3d_{aircraft_type}'
        generator.generate(
            output_dir=temp_dir,
            split_ratios=(1.0, 0.0, 0.0)
        )

        # Move images to organized structure
        src_dir = f'{temp_dir}/train/images'
        dst_dir = f'sample_images/3d/{aircraft_type}'
        if os.path.exists(src_dir):
            for img in os.listdir(src_dir):
                shutil.copy2(f'{src_dir}/{img}', f'{dst_dir}/{img}')

        # Clean up temp directory
        shutil.rmtree(temp_dir)

    print("✅ Generated 3D samples organized by aircraft type")


if __name__ == "__main__":
    print("🛩️  Aircraft Dataset Sample Generator")
    print("=" * 50)

    # Ensure output directories exist
    aircraft_types = ['F15', 'B52', 'C130']
    for aircraft_type in aircraft_types:
        os.makedirs(f'sample_images/2d/{aircraft_type}', exist_ok=True)
        os.makedirs(f'sample_images/3d/{aircraft_type}', exist_ok=True)

    # Generate samples organized by type
    generate_2d_samples_by_type()
    generate_3d_samples_by_type()

    print("\n🎯 Sample images saved to:")
    print("   • 2D samples: sample_images/2d/{F15,B52,C130}/")
    print("   • 3D samples: sample_images/3d/{F15,B52,C130}/")
    print("\n📝 These samples show various aircraft types and orientations")
    print("   organized by dimension then aircraft type.")