#!/usr/bin/env python3
"""
Generate complete 40k aircraft dataset with PyVista rendering and OBB annotations.
Includes proper COCO-style JSON annotations saved to the dataset folder.
"""

import json
import os
import subprocess
import time
from pathlib import Path

OUTPUT_DIR = "aircraft_3d_pyvista_obb_40k"
TOTAL_SCENES = 5000  # 5000 scenes × 8 views = 40,000 images
BATCH_SIZE = 50  # scenes per batch
VIEWS_PER_SCENE = 8
STATE_FILE = f"{OUTPUT_DIR}_generation_state.json"


def load_state():
    # Load generation state from file.
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "batch_num": 0,
        "total_generated": 0,
        "train_scenes": 0,
        "val_scenes": 0,
        "test_scenes": 0,
        "target_train": 3500,  # 70%
        "target_val": 1000,  # 20%
        "target_test": 500,  # 10%
    }


def save_state(state):
    # Save generation state to file.
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def create_single_batch_script():
    # Create the single batch generation script.
    script_content = f'''#!/usr/bin/env python3
"""
Single batch generation for aircraft_3d_pyvista_obb_40k dataset.
Generates one batch then exits to prevent memory leaks.
"""

import os
import json
import shutil
from pathlib import Path
from aircraft_toolkit.core.dataset_3d import Dataset3D
from aircraft_toolkit.config import get_config

BATCH_SIZE = {BATCH_SIZE}
VIEWS_PER_SCENE = {VIEWS_PER_SCENE}
OUTPUT_DIR = "{OUTPUT_DIR}"
STATE_FILE = "{STATE_FILE}"

def load_state():
    # Load generation state from file.
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {{
        'batch_num': 0,
        'total_generated': 0,
        'train_scenes': 0,
        'val_scenes': 0,
        'test_scenes': 0,
        'target_train': 3500,
        'target_val': 1000,
        'target_test': 500
    }}

def save_state(state):
    # Save generation state to file.
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def main():
    # Load current state
    state = load_state()

    # Check if complete
    total_target = state['target_train'] + state['target_val'] + state['target_test']
    total_current = state['train_scenes'] + state['val_scenes'] + state['test_scenes']

    if total_current >= total_target:
        print("✅ Dataset generation already complete!")
        print(f"Total scenes: {{total_current}}/{{total_target}}")
        return

    # Determine which split this batch belongs to
    if state['train_scenes'] < state['target_train']:
        split = 'train'
        scenes_to_generate = min(BATCH_SIZE, state['target_train'] - state['train_scenes'])
        scene_offset = state['train_scenes']
    elif state['val_scenes'] < state['target_val']:
        split = 'val'
        scenes_to_generate = min(BATCH_SIZE, state['target_val'] - state['val_scenes'])
        scene_offset = state['val_scenes']
    else:
        split = 'test'
        scenes_to_generate = min(BATCH_SIZE, state['target_test'] - state['test_scenes'])
        scene_offset = state['test_scenes']

    print(f"""
╔══════════════════════════════════════════════╗
║       AIRCRAFT 3D PYVISTA OBB 40K           ║
╚══════════════════════════════════════════════╝
Batch #: {{state['batch_num'] + 1}}
Split: {{split}}
Scenes: {{scenes_to_generate}}
Images: {{scenes_to_generate * VIEWS_PER_SCENE}}
Output: {{OUTPUT_DIR}}/{{split}}/

Current Progress:
- Train: {{state['train_scenes']}}/{{state['target_train']}} scenes
- Val: {{state['val_scenes']}}/{{state['target_val']}} scenes
- Test: {{state['test_scenes']}}/{{state['target_test']}} scenes
- Total: {{state['total_generated']}} images generated
══════════════════════════════════════════════
""")

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{{OUTPUT_DIR}}/{{split}}/images", exist_ok=True)
    os.makedirs(f"{{OUTPUT_DIR}}/{{split}}/depth", exist_ok=True)
    os.makedirs(f"{{OUTPUT_DIR}}/{{split}}/annotations", exist_ok=True)

    # Configure PyVista provider
    config = get_config()
    config.aircraft.model_provider = 'pyvista'

    # Generate batch
    print("🚀 Starting batch generation...")
    temp_dir = f"temp_batch_{{state['batch_num']}}"

    try:
        # Create dataset with annotations
        dataset = Dataset3D(
            aircraft_types=['F15', 'B52', 'C130'],
            num_scenes=scenes_to_generate,
            views_per_scene=VIEWS_PER_SCENE,
            include_oriented_bboxes=True,
            include_depth_maps=True,
            image_size=(512, 512),
            camera_distance=(8, 15),
            camera_height_range=(-5, 10)
        )

        # Generate to temp directory with proper split ratios
        result = dataset.generate(temp_dir, split_ratios=(1.0, 0.0, 0.0))

        # Move files to final location with correct numbering
        temp_split_dir = f"{{temp_dir}}/train"
        target_split_dir = f"{{OUTPUT_DIR}}/{{split}}"

        # Move images
        images_moved = 0
        for img_file in sorted(Path(f"{{temp_split_dir}}/images").glob("*.png")):
            old_num = int(img_file.stem.split('_')[1])
            view_num = img_file.stem.split('_')[2]
            new_scene_num = old_num + scene_offset
            new_name = f"{{split}}_{{new_scene_num:06d}}_{{view_num}}.png"
            shutil.move(str(img_file), f"{{target_split_dir}}/images/{{new_name}}")
            images_moved += 1

        # Move depth maps
        if os.path.exists(f"{{temp_split_dir}}/depth"):
            for depth_file in sorted(Path(f"{{temp_split_dir}}/depth").glob("*.png")):
                old_num = int(depth_file.stem.split('_')[1])
                view_num = depth_file.stem.split('_')[2]
                new_scene_num = old_num + scene_offset
                new_name = f"{{split}}_{{new_scene_num:06d}}_{{view_num}}_depth.png"
                shutil.move(str(depth_file), f"{{target_split_dir}}/depth/{{new_name}}")

        # Move annotations
        if os.path.exists(f"{{temp_split_dir}}/annotations"):
            for ann_file in sorted(Path(f"{{temp_split_dir}}/annotations").glob("*.json")):
                if ann_file.name == "annotations.json":
                    # This is the main annotations file - we'll handle it separately
                    continue
                old_num = int(ann_file.stem.split('_')[1])
                view_num = ann_file.stem.split('_')[2]
                new_scene_num = old_num + scene_offset
                new_name = f"{{split}}_{{new_scene_num:06d}}_{{view_num}}.json"
                shutil.move(str(ann_file), f"{{target_split_dir}}/annotations/{{new_name}}")

        # Handle main annotations file
        main_ann_file = f"{{temp_split_dir}}/annotations/annotations.json"
        if os.path.exists(main_ann_file):
            # Load annotations and update paths/IDs
            with open(main_ann_file, 'r') as f:
                annotations = json.load(f)

            # Update annotations with correct paths and scene IDs
            for ann in annotations:
                old_scene_id = ann['scene_id']
                new_scene_id = old_scene_id + scene_offset
                ann['scene_id'] = new_scene_id

                # Update image path
                old_image_path = ann['image_path']
                view_id = ann['view_id']
                new_image_name = f"{{split}}_{{new_scene_id:06d}}_{{view_id:02d}}.png"
                ann['image_path'] = f"{{OUTPUT_DIR}}/{{split}}/images/{{new_image_name}}"

                # Update depth path if it exists
                if ann.get('depth_path'):
                    new_depth_name = f"{{split}}_{{new_scene_id:06d}}_{{view_id:02d}}_depth.png"
                    ann['depth_path'] = f"{{OUTPUT_DIR}}/{{split}}/depth/{{new_depth_name}}"

            # Save updated annotations
            target_ann_file = f"{{target_split_dir}}/annotations/batch_{{state['batch_num']}}_annotations.json"
            with open(target_ann_file, 'w') as f:
                json.dump(annotations, f, indent=2)

        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Update state
        state['batch_num'] += 1
        state['total_generated'] += images_moved
        state[f'{{split}}_scenes'] += scenes_to_generate

        # Save state
        save_state(state)

        print(f"""
✅ Batch Complete!
══════════════════════════════════════════════
Images generated: {{images_moved}}
Total progress: {{state['total_generated']}}/{{total_target * VIEWS_PER_SCENE}} images
Batch saved to: {{OUTPUT_DIR}}/{{split}}/
Annotations saved to: {{OUTPUT_DIR}}/{{split}}/annotations/

Next: Run this script again for the next batch
══════════════════════════════════════════════
""")

    except Exception as e:
        print(f"❌ Error: {{e}}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise

if __name__ == "__main__":
    main()
'''

    with open("generate_single_batch_obb.py", "w") as f:
        f.write(script_content)


def run_batch():
    # Run a single batch generation.
    try:
        result = subprocess.run(
            ["python", "generate_single_batch_obb.py"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print("⚠️  Batch timed out after 10 minutes")
        return False, "", "Timeout"
    except Exception as e:
        print(f"❌ Error running batch: {e}")
        return False, "", str(e)


def merge_annotations():
    # Merge all batch annotation files into final COCO-style files.
    print("\n🔗 Merging annotations...")

    for split in ["train", "val", "test"]:
        split_dir = f"{OUTPUT_DIR}/{split}"
        if not os.path.exists(split_dir):
            continue

        annotations_dir = f"{split_dir}/annotations"
        if not os.path.exists(annotations_dir):
            continue

        # Collect all batch annotation files
        all_annotations = []
        for batch_file in sorted(Path(annotations_dir).glob("batch_*_annotations.json")):
            try:
                with open(batch_file, "r") as f:
                    batch_data = json.load(f)
                    all_annotations.extend(batch_data)
            except Exception as e:
                print(f"⚠️  Error reading {batch_file}: {e}")

        # Save merged annotations
        if all_annotations:
            merged_file = f"{split_dir}/annotations.json"
            with open(merged_file, "w") as f:
                json.dump(all_annotations, f, indent=2)
            print(f"✅ Merged {len(all_annotations)} annotations for {split}")


def main():
    print(
        f"""
╔══════════════════════════════════════════════════════════════╗
║                AIRCRAFT 3D PYVISTA OBB 40K                  ║
╚══════════════════════════════════════════════════════════════╝
Target: 40,000 images with oriented bounding boxes
Dataset: {OUTPUT_DIR}
Rendering: PyVista high-quality 3D with proper annotations
Splits: train/val/test with COCO-style JSON annotations
══════════════════════════════════════════════════════════════
"""
    )

    # Create the single batch script
    create_single_batch_script()

    batch_count = 0
    consecutive_failures = 0
    start_time = time.time()

    while True:
        # Check current state
        state = load_state()
        total_target_scenes = state["target_train"] + state["target_val"] + state["target_test"]
        current_scenes = state["train_scenes"] + state["val_scenes"] + state["test_scenes"]

        # Check if complete
        if current_scenes >= total_target_scenes:
            print("🎉 GENERATION COMPLETE!")
            break

        # Determine current split
        if state["train_scenes"] < state["target_train"]:
            current_split = "train"
            split_progress = f"{state['train_scenes']}/{state['target_train']}"
        elif state["val_scenes"] < state["target_val"]:
            current_split = "val"
            split_progress = f"{state['val_scenes']}/{state['target_val']}"
        else:
            current_split = "test"
            split_progress = f"{state['test_scenes']}/{state['target_test']}"

        # Progress display
        batch_count += 1
        elapsed = time.time() - start_time
        progress_pct = (current_scenes / total_target_scenes) * 100

        print(
            f"""
┌─────────────────────────────────────────┐
│ Batch #{batch_count} - {current_split.upper()} split        │
├─────────────────────────────────────────┤
│ Progress: {current_scenes:,}/{total_target_scenes:,} scenes ({progress_pct:.1f}%)   │
│ Images: {state['total_generated']:,} created          │
│ Current: {split_progress} {current_split} scenes       │
│ Time: {elapsed/60:.1f} minutes elapsed          │
└─────────────────────────────────────────┘"""
        )

        # Run batch
        print("🚀 Running batch generation...", end=" ")
        success, stdout, stderr = run_batch()

        if success:
            print("✅ SUCCESS")
            consecutive_failures = 0

            # Show completion line from batch script
            for line in stdout.split("\\n"):
                if "Batch Complete!" in line or "Total progress:" in line:
                    print(f"   {line}")

            # Small delay between batches
            time.sleep(1)

        else:
            consecutive_failures += 1
            print(f"❌ FAILED (attempt {consecutive_failures})")

            if consecutive_failures >= 3:
                print("💥 Too many consecutive failures. Stopping.")
                print("Last error:", stderr[-200:] if stderr else "No error details")
                break

            print("⏱️  Waiting 10 seconds before retry...")
            time.sleep(10)

    # Merge annotations
    merge_annotations()

    # Final summary
    final_state = load_state()
    total_final_images = final_state["total_generated"]
    final_time = (time.time() - start_time) / 60

    print(
        f"""
╔══════════════════════════════════════════════════════════════╗
║                     GENERATION SUMMARY                      ║
╚══════════════════════════════════════════════════════════════╝
✅ Total images generated: {total_final_images:,}
📊 Distribution:
   • Train: {final_state['train_scenes']:,} scenes ({final_state['train_scenes'] * VIEWS_PER_SCENE:,} images)
   • Val: {final_state['val_scenes']:,} scenes ({final_state['val_scenes'] * VIEWS_PER_SCENE:,} images)
   • Test: {final_state['test_scenes']:,} scenes ({final_state['test_scenes'] * VIEWS_PER_SCENE:,} images)
⏱️  Total time: {final_time:.1f} minutes
📁 Output directory: {OUTPUT_DIR}/
📋 Annotations: COCO-style JSON with oriented bounding boxes
══════════════════════════════════════════════════════════════
"""
    )

    # Clean up
    if os.path.exists("generate_single_batch_obb.py"):
        os.remove("generate_single_batch_obb.py")


if __name__ == "__main__":
    main()
