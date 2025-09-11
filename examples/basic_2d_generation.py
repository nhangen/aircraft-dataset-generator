#!/usr/bin/env python3
"""
Basic 2D Aircraft Dataset Generation Example

This example shows how to generate a simple 2D aircraft dataset
with pose annotations using the Aircraft Dataset Generator toolkit.
"""

from aircraft_toolkit import Dataset2D


def main():
    """Generate basic 2D aircraft dataset"""
    
    print("🛩️  Aircraft Dataset Generator - Basic 2D Example")
    print("=" * 50)
    
    # Create dataset generator
    dataset = Dataset2D(
        aircraft_types=['F15', 'B52', 'C130'],  # Military aircraft
        num_samples=1000,                        # Generate 1K images
        image_size=(224, 224),                  # Standard ViT input size
        pose_range={                            # Pose variation ranges
            'pitch': (-30, 30),                 # Pitch angle range
            'yaw': (-180, 180),                 # Full yaw rotation
            'roll': (-15, 15),                  # Limited roll
        }
    )
    
    # Generate the dataset
    print("\n🚀 Starting dataset generation...")
    
    results = dataset.generate(
        output_dir='aircraft_2d_basic',          # Output directory
        split_ratios=(0.7, 0.2, 0.1),          # 70% train, 20% val, 10% test
        annotation_format='custom',              # Custom JSON format
        num_workers=1                           # Single-threaded for now
    )
    
    # Print results
    print("\n✅ Dataset generation complete!")
    print(f"📊 Generated {results['total_samples']} total images:")
    print(f"   • Train: {results['train_samples']} images")
    print(f"   • Validation: {results['val_samples']} images") 
    print(f"   • Test: {results['test_samples']} images")
    print(f"   • Aircraft types: {', '.join(results['aircraft_types'])}")
    print(f"📁 Dataset saved to: {results['output_dir']}")
    
    print("\n🎯 Next steps:")
    print("   1. Inspect generated images in aircraft_2d_basic/train/images/")
    print("   2. Review annotations in aircraft_2d_basic/train_annotations.json")
    print("   3. Use dataset for training pose estimation models")


if __name__ == "__main__":
    main()