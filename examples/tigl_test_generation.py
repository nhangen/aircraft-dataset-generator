#!/usr/bin/env python3
"""
TiGL Test Dataset Generation

Generate sample datasets demonstrating the TiGL integration
with comparison to the basic provider.
"""

import os
import time
from aircraft_toolkit import Dataset3D
from aircraft_toolkit.providers import get_provider, list_providers
from aircraft_toolkit.config import get_config_manager


def generate_basic_sample():
    """Generate sample with basic provider for comparison."""
    print("🔧 Generating Basic Provider Sample...")

    # Force basic provider
    config_mgr = get_config_manager()
    config = config_mgr.get_config()
    config.aircraft.model_provider = 'basic'
    config_mgr.save_config(config)

    dataset = Dataset3D(
        aircraft_types=['F15', 'B52', 'C130'],
        num_scenes=3,
        views_per_scene=4,
        image_size=(256, 256)
    )

    start_time = time.time()
    results = dataset.generate('output/basic_provider_sample')
    generation_time = time.time() - start_time

    print(f"✓ Basic provider sample generated in {generation_time:.2f}s")
    print(f"  Total images: {results['total_images']}")
    print(f"  Output dir: {results['output_dir']}")

    return results


def generate_tigl_sample():
    """Generate sample with TiGL provider if available."""
    print("\n🚀 Generating TiGL Provider Sample...")

    try:
        # Try to use TiGL provider
        config_mgr = get_config_manager()
        config = config_mgr.get_config()
        config.aircraft.model_provider = 'tigl'
        config.aircraft.detail_level = 'medium'
        config_mgr.save_config(config)

        dataset = Dataset3D(
            aircraft_types=['F15', 'B52', 'C130'],
            num_scenes=3,
            views_per_scene=4,
            image_size=(256, 256)
        )

        start_time = time.time()
        results = dataset.generate('output/tigl_provider_sample')
        generation_time = time.time() - start_time

        print(f"✓ TiGL provider sample generated in {generation_time:.2f}s")
        print(f"  Total images: {results['total_images']}")
        print(f"  Output dir: {results['output_dir']}")

        return results

    except Exception as e:
        print(f"✗ TiGL provider not available: {e}")
        print("  Install with: conda install -c dlr-sc tigl3")
        return None


def generate_comparison_sample():
    """Generate samples with different detail levels."""
    print("\n🔍 Generating Detail Level Comparison...")

    try:
        provider = get_provider('tigl')

        for detail_level in ['low', 'medium', 'high']:
            print(f"  Generating {detail_level} detail sample...")

            config_mgr = get_config_manager()
            config = config_mgr.get_config()
            config.aircraft.model_provider = 'tigl'
            config.aircraft.detail_level = detail_level
            config_mgr.save_config(config)

            dataset = Dataset3D(
                aircraft_types=['F15'],
                num_scenes=1,
                views_per_scene=2,
                image_size=(256, 256)
            )

            start_time = time.time()
            results = dataset.generate(f'output/tigl_{detail_level}_detail')
            generation_time = time.time() - start_time

            print(f"    ✓ {detail_level} detail: {generation_time:.2f}s")

    except Exception as e:
        print(f"  ✗ TiGL not available for comparison: {e}")


def analyze_mesh_quality():
    """Analyze and compare mesh quality between providers."""
    print("\n📊 Mesh Quality Analysis")
    print("=" * 50)

    providers = list_providers()
    print(f"Available providers: {list(providers.keys())}")

    results = {}

    # Test each provider
    for provider_name in providers:
        try:
            provider = get_provider(provider_name)

            print(f"\n{provider_name.upper()} PROVIDER:")
            print("-" * 25)

            for aircraft_type in ['F15', 'B52', 'C130']:
                try:
                    if provider_name == 'tigl':
                        mesh = provider.create_aircraft(aircraft_type, detail_level='medium')
                    else:
                        mesh = provider.create_aircraft(aircraft_type)

                    results[f"{provider_name}_{aircraft_type}"] = {
                        'vertices': mesh.num_vertices,
                        'faces': mesh.num_faces,
                        'provider': provider_name
                    }

                    print(f"  {aircraft_type}: {mesh.num_vertices:,} vertices, {mesh.num_faces:,} faces")

                except Exception as e:
                    print(f"  {aircraft_type}: Error - {e}")

        except Exception as e:
            print(f"{provider_name} provider error: {e}")

    # Generate comparison table
    print(f"\n{'Aircraft':<15} {'Provider':<10} {'Vertices':<10} {'Faces':<10} {'Quality Improvement'}")
    print("-" * 70)

    for aircraft in ['F15', 'B52', 'C130']:
        basic_key = f"basic_{aircraft}"
        tigl_key = f"tigl_{aircraft}"

        if basic_key in results and tigl_key in results:
            basic_verts = results[basic_key]['vertices']
            tigl_verts = results[tigl_key]['vertices']
            improvement = tigl_verts / basic_verts if basic_verts > 0 else 0

            print(f"{aircraft:<15} {'Basic':<10} {basic_verts:<10,} {results[basic_key]['faces']:<10,}")
            print(f"{aircraft:<15} {'TiGL':<10} {tigl_verts:<10,} {results[tigl_key]['faces']:<10,} {improvement:.0f}x improvement")
            print()


def create_github_samples():
    """Create samples specifically for GitHub documentation."""
    print("\n📁 Creating GitHub Documentation Samples...")

    # Create samples directory
    os.makedirs('samples', exist_ok=True)

    try:
        # Quick basic sample
        print("  Creating basic provider showcase...")
        config_mgr = get_config_manager()
        config = config_mgr.get_config()
        config.aircraft.model_provider = 'basic'
        config_mgr.save_config(config)

        basic_dataset = Dataset3D(
            aircraft_types=['F15'],
            num_scenes=1,
            views_per_scene=4,
            image_size=(512, 512)
        )
        basic_dataset.generate('samples/basic_showcase')

        # TiGL sample if available
        try:
            print("  Creating TiGL provider showcase...")
            config.aircraft.model_provider = 'tigl'
            config.aircraft.detail_level = 'medium'
            config_mgr.save_config(config)

            tigl_dataset = Dataset3D(
                aircraft_types=['F15'],
                num_scenes=1,
                views_per_scene=4,
                image_size=(512, 512)
            )
            tigl_dataset.generate('samples/tigl_showcase')
            print("  ✓ GitHub samples created in samples/ directory")

        except Exception as e:
            print(f"  ✗ TiGL showcase skipped: {e}")

    except Exception as e:
        print(f"  ✗ Error creating GitHub samples: {e}")


def main():
    """Main test generation workflow."""
    print("🛩️  TiGL INTEGRATION TEST DATASET GENERATION")
    print("=" * 60)

    # Show system status
    print("System Status:")
    providers = list_providers()
    print(f"  Available providers: {list(providers.keys())}")

    config = get_config_manager().get_config()
    print(f"  Preferred provider: {config.get_preferred_provider()}")
    print(f"  Detail level: {config.aircraft.detail_level}")

    # Generate samples
    print("\n" + "=" * 60)
    print("GENERATING TEST DATASETS")
    print("=" * 60)

    # 1. Basic provider sample
    basic_results = generate_basic_sample()

    # 2. TiGL provider sample
    tigl_results = generate_tigl_sample()

    # 3. Detail level comparison
    generate_comparison_sample()

    # 4. Mesh quality analysis
    analyze_mesh_quality()

    # 5. GitHub samples
    create_github_samples()

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)

    print("Generated Datasets:")
    if basic_results:
        print(f"  ✓ Basic Provider: output/basic_provider_sample/")
    if tigl_results:
        print(f"  ✓ TiGL Provider: output/tigl_provider_sample/")

    print(f"\nGitHub Samples: samples/")
    print(f"Documentation: TIGL_INTEGRATION.md")
    print(f"Integration Plan: INTEGRATION_PLAN.md")

    print("\nNext Steps:")
    print("  1. Review generated samples in output/ directories")
    print("  2. Compare image quality between providers")
    print("  3. Check GitHub samples for documentation")
    print("  4. Run: git add . && git commit -m 'Add TiGL integration samples'")


if __name__ == "__main__":
    main()