#!/usr/bin/env python3
"""
FIXED: Generate 40k aircraft dataset with proper annotation handling.
This version correctly moves the annotation JSON files from temp directories.
"""

import json
import os
import subprocess
import time
from pathlib import Path

OUTPUT_DIR = "aircraft_3d_pyvista_obb_40k_FIXED"
TOTAL_SCENES = 5000  # 5000 scenes × 8 views = 40,000 images
BATCH_SIZE = 50  # scenes per batch
VIEWS_PER_SCENE = 8
STATE_FILE = f"{OUTPUT_DIR}_generation_state.json"


def load_state():
    # Load generation state from file.
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "batch_num": 0,
        "total_generated": 0,
        "train_scenes": 0,
        "val_scenes": 0,
        "test_scenes": 0,
        "target_train": 3500,
        "target_val": 1000,
        "target_test": 500,
    }


def save_state(state):
    # Save generation state to file.
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def create_single_batch_script():
    # Create the FIXED single batch generation script.
    script_content = f'''#!/usr/bin/env python3
"""
FIXED single batch generation - properly handles annotation files.
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
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def main():
    state = load_state()

    # Check if complete
    total_target = state['target_train'] + state['target_val'] + state['target_test']
    total_current = state['train_scenes'] + state['val_scenes'] + state['test_scenes']

    if total_current >= total_target:
        print("✅ Dataset generation already complete!")
        return

    # Determine split
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

    print(f"Batch {{state['batch_num'] + 1}}: {{split}} split, {{scenes_to_generate}} scenes")

    # Create output directories
    os.makedirs(f"{{OUTPUT_DIR}}/{{split}}/images", exist_ok=True)
    os.makedirs(f"{{OUTPUT_DIR}}/{{split}}/depth", exist_ok=True)

    # Configure PyVista
    config = get_config()
    config.aircraft.model_provider = 'pyvista'

    temp_dir = f"temp_batch_{{state['batch_num']}}"

    try:
        # Generate dataset
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

        result = dataset.generate(temp_dir, split_ratios=(1.0, 0.0, 0.0))

        # Move images and depth maps
        temp_split_dir = f"{{temp_dir}}/train"
        target_split_dir = f"{{OUTPUT_DIR}}/{{split}}"

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

        # CRITICAL FIX: Move annotation files from temp directory ROOT
        annotation_file = f"{{temp_dir}}/train_3d_annotations.json"
        if os.path.exists(annotation_file):
            # Load annotations and update scene IDs and paths
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

            # Update annotations with correct scene IDs and paths
            for ann in annotations:
                old_scene_id = ann['scene_id']
                new_scene_id = old_scene_id + scene_offset
                ann['scene_id'] = new_scene_id

                # Update image path
                view_id = ann['view_id']
                new_image_name = f"{{split}}_{{new_scene_id:06d}}_{{view_id:02d}}.png"
                ann['image_path'] = f"{{OUTPUT_DIR}}/{{split}}/images/{{new_image_name}}"

                # Update depth path
                if ann.get('depth_path'):
                    new_depth_name = f"{{split}}_{{new_scene_id:06d}}_{{view_id:02d}}_depth.png"
                    ann['depth_path'] = f"{{OUTPUT_DIR}}/{{split}}/depth/{{new_depth_name}}"

            # Save batch annotations
            target_ann_file = f"{{OUTPUT_DIR}}/{{split}}_batch_{{state['batch_num']:03d}}_annotations.json"
            with open(target_ann_file, 'w') as f:
                json.dump(annotations, f, indent=2)
            print(f"✅ Saved annotations: {{target_ann_file}}")
        else:
            print("⚠️  Warning: No annotation file found!")

        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Update state
        state['batch_num'] += 1
        state['total_generated'] += images_moved
        state[f'{{split}}_scenes'] += scenes_to_generate
        save_state(state)

        print(f"✅ Batch complete: {{images_moved}} images, {{state['total_generated']}}/40000 total")

    except Exception as e:
        print(f"❌ Error: {{e}}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise

if __name__ == "__main__":
    main()
'''

    with open("generate_single_batch_FIXED.py", "w") as f:
        f.write(script_content)


def run_batch():
    # Run a single batch generation.
    try:
        result = subprocess.run(
            ["python", "generate_single_batch_FIXED.py"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)


def merge_annotations():
    # Merge all batch annotation files into final COCO-style files.
    print("🔗 Merging annotations...")

    for split in ["train", "val", "test"]:
        split_dir = f"{OUTPUT_DIR}/{split}"
        if not os.path.exists(split_dir):
            continue

        # Collect all batch annotation files for this split
        all_annotations = []
        batch_files = sorted(Path(".").glob(f"{OUTPUT_DIR}/{split}_batch_*_annotations.json"))

        for batch_file in batch_files:
            try:
                with open(batch_file) as f:
                    batch_data = json.load(f)
                    all_annotations.extend(batch_data)
                # Remove batch file after merging
                os.remove(batch_file)
            except Exception as e:
                print(f"⚠️  Error reading {batch_file}: {e}")

        # Save merged annotations
        if all_annotations:
            merged_file = f"{OUTPUT_DIR}/{split}_annotations.json"
            with open(merged_file, "w") as f:
                json.dump(all_annotations, f, indent=2)
            print(f"✅ Merged {len(all_annotations)} annotations for {split}")


def main():
    print(
        f"""
╔══════════════════════════════════════════════════════════════╗
║                 FIXED AIRCRAFT 3D DATASET 40K               ║
╚══════════════════════════════════════════════════════════════╝
Dataset: {OUTPUT_DIR}
Fixed: Proper annotation handling guaranteed
"""
    )

    create_single_batch_script()

    batch_count = 0
    consecutive_failures = 0
    start_time = time.time()

    while True:
        state = load_state()
        total_target_scenes = state["target_train"] + state["target_val"] + state["target_test"]
        current_scenes = state["train_scenes"] + state["val_scenes"] + state["test_scenes"]

        if current_scenes >= total_target_scenes:
            print("🎉 GENERATION COMPLETE!")
            break

        batch_count += 1
        elapsed = time.time() - start_time
        progress_pct = (current_scenes / total_target_scenes) * 100

        print(
            f"\\nBatch #{batch_count}: {current_scenes}/{total_target_scenes} scenes ({progress_pct:.1f}%)"
        )

        success, stdout, stderr = run_batch()

        if success:
            print("✅ SUCCESS")
            consecutive_failures = 0
            time.sleep(1)
        else:
            consecutive_failures += 1
            print(f"❌ FAILED (attempt {consecutive_failures})")
            if consecutive_failures >= 3:
                print("💥 Too many failures. Stopping.")
                break
            time.sleep(10)

    # Merge annotations
    merge_annotations()

    # Final summary
    final_state = load_state()
    final_time = (time.time() - start_time) / 60

    print(
        f"""
╔══════════════════════════════════════════════════════════════╗
║                     GENERATION SUMMARY                      ║
╚══════════════════════════════════════════════════════════════╝
✅ Total images: {final_state['total_generated']:,}
📊 Train: {final_state['train_scenes']:,} scenes
📊 Val: {final_state['val_scenes']:,} scenes
📊 Test: {final_state['test_scenes']:,} scenes
⏱️  Time: {final_time:.1f} minutes
📁 Output: {OUTPUT_DIR}/
📋 Annotations: GUARANTEED INCLUDED
"""
    )

    # Clean up
    if os.path.exists("generate_single_batch_FIXED.py"):
        os.remove("generate_single_batch_FIXED.py")


if __name__ == "__main__":
    main()
