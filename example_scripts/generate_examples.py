#!/usr/bin/env python3
"""
Aircraft Dataset Generator - Example Scripts

This script demonstrates how to generate different types of aircraft datasets
using the Aircraft Dataset Generator toolkit.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from aircraft_toolkit import Dataset2D, Dataset3D


class DatasetGenerator:
    # Main class for generating aircraft datasets.

    def __init__(self):
        self.aircraft_types = ["F15", "B52", "C130"]

    def test_dataset(self, output_dir="output/test", include_obb=False):
        # Generate a small test dataset for validation.
        print("\n🧪 Generating Test Dataset")
        print("=" * 50)

        dataset = Dataset3D(
            aircraft_types=["F15"],
            num_scenes=2,
            views_per_scene=3,
            include_oriented_bboxes=include_obb,
            image_size=(256, 256),
        )

        results = dataset.generate(output_dir)
        print(f"✓ Generated {results['total_images']} test images in {output_dir}")
        return results

    def production_3d_dataset(
        self,
        output_dir="output/production_3d",
        num_scenes=50,
        views_per_scene=8,
        include_obb=False,
    ):
        # Generate production-quality 3D multi-view dataset.
        print("\n🚀 Generating Production 3D Dataset")
        print("=" * 50)

        dataset = Dataset3D(
            aircraft_types=self.aircraft_types,
            num_scenes=num_scenes,
            views_per_scene=views_per_scene,
            image_size=(512, 512),
            include_depth_maps=True,
            include_oriented_bboxes=include_obb,
        )

        results = dataset.generate(output_dir)

        print(f"✓ Generated {results['total_images']} images")
        print(f"  📊 {results['train_scenes']} train scenes")
        print(f"  📊 {results['val_scenes']} validation scenes")
        print(f"  📊 {results['test_scenes']} test scenes")
        print(f"  📁 Output: {output_dir}")
        return results

    def silhouette_2d_dataset(
        self, output_dir="output/silhouettes_2d", num_samples=300, format="coco"
    ):
        # Generate 2D silhouette dataset.
        print("\n✈️ Generating 2D Silhouette Dataset")
        print("=" * 50)

        dataset = Dataset2D(
            aircraft_types=self.aircraft_types,
            num_samples=num_samples,
            image_size=(224, 224),
        )

        results = dataset.generate(output_dir, annotation_format=format)

        print(f"✓ Generated {results['total_samples']} silhouettes")
        print(f"  📊 Format: {format.upper()} annotations")
        print(f"  📁 Output: {output_dir}")
        return results

    def comparison_dataset(self, output_dir="output/comparison"):
        # Generate datasets for quality comparison.
        print("\n📊 Generating Comparison Dataset")
        print("=" * 50)

        # Generate both 2D and 3D for comparison
        results_2d = self.silhouette_2d_dataset(f"{output_dir}/2d", num_samples=100)
        results_3d = self.production_3d_dataset(
            f"{output_dir}/3d", num_scenes=10, views_per_scene=4
        )

        print(f"\n📈 Comparison Summary:")
        print(f"  2D Silhouettes: {results_2d['total_samples']} images")
        print(f"  3D Multi-view: {results_3d['total_images']} images")
        return {"2d": results_2d, "3d": results_3d}


def main():
    # Main entry point.
    parser = argparse.ArgumentParser(
        description="Generate aircraft datasets with real 3D models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_examples.py --mode test         # Quick test dataset
  python generate_examples.py --mode 3d           # Production 3D dataset
  python generate_examples.py --mode 3d --obb     # 3D dataset with oriented bounding boxes
  python generate_examples.py --mode 2d           # 2D silhouette dataset
  python generate_examples.py --mode comparison   # Both 2D and 3D
  python generate_examples.py --mode all --obb    # All dataset types with OBBs
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["test", "3d", "2d", "comparison", "all"],
        default="test",
        help="Dataset generation mode (default: test)",
    )

    parser.add_argument(
        "--output", default="output", help="Output directory (default: output)"
    )

    parser.add_argument(
        "--obb",
        action="store_true",
        help="Include oriented bounding box computation (useful for pose estimation)",
    )

    parser.add_argument(
        "--scenes",
        type=int,
        default=50,
        help="Number of scenes for 3D datasets (default: 50)",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Number of samples for 2D datasets (default: 300)",
    )

    args = parser.parse_args()

    print("🛩️ Aircraft Dataset Generator")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")

    generator = DatasetGenerator()

    try:
        if args.mode == "test":
            generator.test_dataset(f"{args.output}/test", include_obb=args.obb)

        elif args.mode == "3d":
            generator.production_3d_dataset(
                f"{args.output}/3d", num_scenes=args.scenes, include_obb=args.obb
            )

        elif args.mode == "2d":
            generator.silhouette_2d_dataset(
                f"{args.output}/2d", num_samples=args.samples
            )

        elif args.mode == "comparison":
            generator.comparison_dataset(f"{args.output}/comparison")

        elif args.mode == "all":
            generator.test_dataset(f"{args.output}/test", include_obb=args.obb)
            generator.silhouette_2d_dataset(
                f"{args.output}/2d", num_samples=args.samples
            )
            generator.production_3d_dataset(
                f"{args.output}/3d", num_scenes=args.scenes, include_obb=args.obb
            )

        print("\n🎯 Generation Complete!")
        print(f"Check the output directory: {args.output}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
