#!/usr/bin/env python3
"""
Generate aircraft dataset with oriented bounding boxes for pose estimation training.

This script generates a comprehensive dataset with 3D aircraft models and their
oriented bounding boxes to help ViT models learn geometric relationships before
abstract pose estimation.
"""

import logging
import sys
from pathlib import Path
from pprint import pformat

# Add the toolkit to Python path
sys.path.insert(0, str(Path(__file__).parent))

from aircraft_toolkit import Dataset3D  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Generate comprehensive OBB dataset for pose estimation training.

    # Output directory in the pose estimation project
    output_dir = Path("../pose-estimation-vit/pe-vit-data/data/aircraft_3d_obb_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting OBB dataset generation...")
    logger.info(f"Output directory: {output_dir.absolute()}")

    # Create dataset with oriented bounding boxes
    dataset = Dataset3D(
        aircraft_types=["F15", "B52", "C130"],  # All three real aircraft models
        num_scenes=500,  # Sufficient data for training
        views_per_scene=4,  # Multiple views per scene
        include_oriented_bboxes=True,  # Enable 3D bounding boxes
        image_size=(224, 224),  # Standard ViT input size
        camera_distance=(8, 12),  # Varied viewing distances
        camera_height_range=(-5, 10),  # Varied camera heights
        include_depth_maps=False,  # Focus on RGB + OBB annotations
        include_surface_normals=False,  # Not needed for pose estimation
    )

    logger.info("Dataset configuration:")
    logger.info(f"  Aircraft types: {dataset.aircraft_types}")
    logger.info(f"  Total scenes: {dataset.num_scenes}")
    logger.info(f"  Views per scene: {dataset.views_per_scene}")
    logger.info(f"  Total images: {dataset.num_scenes * dataset.views_per_scene}")
    logger.info(f"  Image size: {dataset.image_size}")
    logger.info(f"  Oriented bounding boxes: {dataset.include_oriented_bboxes}")

    # Generate the dataset
    try:
        _results = dataset.generate(
            output_dir=str(output_dir),
            split_ratios=(0.7, 0.2, 0.1),  # 70% train, 20% val, 10% test
            annotation_format="custom_3d",  # Custom format with OBB data
            num_workers=1,  # Single worker to avoid rendering issues
        )

        # Log a concise summary and sample values from the returned object.

        logger.info(f"Generation returned type: {type(_results).__name__}")

        if isinstance(_results, dict):
            logger.info(f"Result keys: {list(_results.keys())}")
            for k, v in _results.items():
                if isinstance(v, (list, tuple, set)):
                    logger.info(f"  {k}: {len(v)} items")
                    for i, item in enumerate(v[:5]):  # show up to first 5 entries
                        logger.debug(f"    {k}[{i}]: {pformat(item)}")
                else:
                    logger.info(f"  {k}: {pformat(v)}")
        elif isinstance(_results, (list, tuple, set)):
            logger.info(f"Result is a sequence of length {len(_results)}")
            for i, item in enumerate(_results[:10]):  # show up to first 10 entries
                logger.debug(f"  [{i}]: {pformat(item)}")
        else:
            logger.info(f"Result value: {pformat(_results)}")

        logger.info("Dataset generation completed successfully!")
        logger.info("Dataset generation finished")

        # Print dataset statistics
        print("\n" + "=" * 60)
        print("AIRCRAFT 3D OBB DATASET GENERATION COMPLETE")
        print("=" * 60)
        print(f"📁 Output directory: {output_dir}")
        print(f"🛩️  Aircraft types: {len(dataset.aircraft_types)} (F15, B52, C130)")
        print(f"🎯 Total scenes: {dataset.num_scenes}")
        print(f"📸 Views per scene: {dataset.views_per_scene}")
        print(f"🖼️  Total images: {dataset.num_scenes * dataset.views_per_scene}")
        print("📦 Oriented bounding boxes: ✅ Enabled")
        print("📊 Data splits: 70% train / 20% val / 10% test")
        print("\n🎯 Purpose: Solve ViT pose estimation training failures")
        print("   Previous experiments showed ~120° rotation errors (random guessing)")
        print("   OBB annotations provide geometric anchors for learning")
        print("\n✅ Ready for multi-task ViT training (pose + OBB prediction)")

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
