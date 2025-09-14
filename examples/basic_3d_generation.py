#!/usr/bin/env python3
"""
Basic 3D Aircraft Dataset Generation Example

This example shows how to generate a 3D multi-view aircraft dataset
with camera poses and depth maps using the Aircraft Dataset Generator toolkit.
"""

from aircraft_toolkit import Dataset3D


def main():
    """Generate basic 3D aircraft dataset"""

    print("🛩️  Aircraft Dataset Generator - Basic 3D Example")
    print("=" * 50)

    # Create 3D dataset generator
    dataset = Dataset3D(
        aircraft_types=['F15', 'B52', 'C130'],  # Military aircraft
        num_scenes=10,                           # Generate 10 scenes
        views_per_scene=8,                      # 8 camera views per scene
        camera_distance=(8, 12),                # Distance from aircraft
        camera_height_range=(-5, 10),           # Camera height variation
        include_depth_maps=True,                # Generate depth maps
        image_size=(512, 512)                   # Higher res for 3D
    )

    # Generate the dataset
    print("\n🚀 Starting 3D dataset generation...")

    results = dataset.generate(
        output_dir='aircraft_3d_basic',         # Output directory
        split_ratios=(0.7, 0.2, 0.1),          # 70% train, 20% val, 10% test
        annotation_format='custom_3d',          # 3D annotation format
        num_workers=1                           # Single-threaded for now
    )

    # Print results
    print("\n✅ 3D Dataset generation complete!")
    print(f"📊 Generated {results['total_images']} total images:")
    print(f"   • Train scenes: {results['train_scenes']} ({results['train_scenes'] * results['views_per_scene']} images)")
    print(f"   • Val scenes: {results['val_scenes']} ({results['val_scenes'] * results['views_per_scene']} images)")
    print(f"   • Test scenes: {results['test_scenes']} ({results['test_scenes'] * results['views_per_scene']} images)")
    print(f"   • Views per scene: {results['views_per_scene']}")
    print(f"   • Aircraft types: {', '.join(results['aircraft_types'])}")
    print(f"📁 Dataset saved to: {results['output_dir']}")

    print("\n🎯 Next steps:")
    print("   1. Inspect generated images in aircraft_3d_basic/train/images/")
    print("   2. Check depth maps in aircraft_3d_basic/train/depth/")
    print("   3. Review 3D annotations in aircraft_3d_basic/train_3d_annotations.json")
    print("   4. Use dataset for training 3D pose estimation or multi-view models")


if __name__ == "__main__":
    main()