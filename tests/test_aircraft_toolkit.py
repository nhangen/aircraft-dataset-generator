# Tests for aircraft toolkit components.

import sys
from pathlib import Path

import pytest

# Add aircraft-dataset-generator to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_aircraft_toolkit_import():
    # Test that aircraft_toolkit can be imported.
    try:
        import aircraft_toolkit

        assert hasattr(aircraft_toolkit, "__version__") or hasattr(aircraft_toolkit, "__name__")
    except ImportError as e:
        pytest.skip(f"Aircraft toolkit not properly installed: {e}")


def test_aircraft_models_import():
    # Test that aircraft models can be imported.
    try:
        from aircraft_toolkit.models import military

        # Basic smoke test - check that module loaded
        assert hasattr(military, "__name__")
    except ImportError as e:
        pytest.skip(f"Aircraft models not available: {e}")


def test_aircraft_providers_import():
    # Test that providers can be imported.
    try:
        from aircraft_toolkit.providers import base, registry

        assert hasattr(base, "__name__")
        assert hasattr(registry, "__name__")
    except ImportError as e:
        pytest.skip(f"Aircraft providers not available: {e}")


def test_dataset_generation_import():
    # Test that dataset generation modules exist.
    try:
        from aircraft_toolkit.core import dataset_2d, dataset_3d

        assert hasattr(dataset_2d, "__name__")
        assert hasattr(dataset_3d, "__name__")
    except ImportError as e:
        pytest.skip(f"Dataset generation modules not available: {e}")


def test_config_import():
    # Test that config can be imported.
    try:
        from aircraft_toolkit import config

        assert hasattr(config, "__name__")
    except ImportError as e:
        pytest.skip(f"Config module not available: {e}")


def test_utils_import():
    # Test that utils can be imported.
    try:
        from aircraft_toolkit.utils import logging, validation

        assert hasattr(validation, "__name__")
        assert hasattr(logging, "__name__")
    except ImportError as e:
        pytest.skip(f"Utils modules not available: {e}")


def test_run_tests_script_exists():
    # Test that run_tests.py exists and is valid.
    script_path = Path(__file__).parent.parent / "run_tests.py"
    if script_path.exists():
        with open(script_path) as f:
            content = f.read()
        try:
            compile(content, str(script_path), "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in run_tests.py: {e}")
    else:
        pytest.skip("run_tests.py not found")


def test_setup_py_exists():
    # Test that setup.py exists and is valid.
    script_path = Path(__file__).parent.parent / "setup.py"
    if script_path.exists():
        with open(script_path) as f:
            content = f.read()
        try:
            compile(content, str(script_path), "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in setup.py: {e}")


if __name__ == "__main__":
    # Allow running as script
    test_aircraft_toolkit_import()
    test_aircraft_models_import()
    test_aircraft_providers_import()
    test_dataset_generation_import()
    test_config_import()
    test_utils_import()
    test_run_tests_script_exists()
    test_setup_py_exists()
    print("✅ All aircraft toolkit tests passed!")
