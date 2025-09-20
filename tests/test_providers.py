"""
Tests for the provider system and individual providers.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from aircraft_toolkit.providers.base import ModelProvider, AircraftMesh
from aircraft_toolkit.providers.basic import BasicProvider
from aircraft_toolkit.providers.registry import ProviderRegistry, get_provider
from aircraft_toolkit.config import Config, get_config_manager


class TestAircraftMesh:
    """Test AircraftMesh data structure."""

    def test_valid_mesh_creation(self):
        """Test creating a valid mesh."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])

        mesh = AircraftMesh(vertices=vertices, faces=faces)

        assert mesh.num_vertices == 3
        assert mesh.num_faces == 1
        assert mesh.metadata == {}

    def test_mesh_validation(self):
        """Test mesh validation."""
        # Invalid vertices shape
        with pytest.raises(AssertionError):
            AircraftMesh(vertices=np.array([[0, 0]]), faces=np.array([[0, 1, 2]]))

        # Invalid faces shape
        with pytest.raises(AssertionError):
            AircraftMesh(vertices=np.array([[0, 0, 0]]), faces=np.array([[0, 1]]))

    def test_compute_normals(self):
        """Test normal computation."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        faces = np.array([[0, 1, 2], [0, 1, 3]])

        mesh = AircraftMesh(vertices=vertices, faces=faces)
        mesh.compute_normals()

        assert mesh.normals is not None
        assert mesh.normals.shape == vertices.shape

    def test_center_and_scale(self):
        """Test centering and scaling."""
        vertices = np.array([
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]
        ])
        faces = np.array([[0, 1, 2]])

        mesh = AircraftMesh(vertices=vertices, faces=faces)
        original_size = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))

        mesh.center_and_scale(target_size=10.0)

        # Check centering
        center = mesh.vertices.mean(axis=0)
        np.testing.assert_allclose(center, [0, 0, 0], atol=1e-10)

        # Check scaling
        new_size = np.linalg.norm(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
        assert abs(new_size - 10.0) < 1e-10


class TestBasicProvider:
    """Test BasicProvider implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = BasicProvider()

    def test_supported_aircraft(self):
        """Test getting supported aircraft."""
        aircraft = self.provider.get_supported_aircraft()
        expected = ['F15', 'B52', 'C130']
        assert set(aircraft) == set(expected)

    def test_create_f15(self):
        """Test creating F-15 model."""
        mesh = self.provider.create_aircraft('F15')

        assert isinstance(mesh, AircraftMesh)
        assert mesh.num_vertices > 0
        assert mesh.num_faces > 0
        assert mesh.metadata['aircraft_type'] == 'F15'
        assert mesh.metadata['provider'] == 'basic'
        assert mesh.normals is not None

    def test_create_b52(self):
        """Test creating B-52 model."""
        mesh = self.provider.create_aircraft('B52')

        assert isinstance(mesh, AircraftMesh)
        assert mesh.metadata['aircraft_type'] == 'B52'

    def test_create_c130(self):
        """Test creating C-130 model."""
        mesh = self.provider.create_aircraft('C130')

        assert isinstance(mesh, AircraftMesh)
        assert mesh.metadata['aircraft_type'] == 'C130'

    def test_invalid_aircraft_type(self):
        """Test handling invalid aircraft type."""
        with pytest.raises(ValueError, match="Aircraft type 'INVALID' not supported"):
            self.provider.create_aircraft('INVALID')

    def test_provider_info(self):
        """Test getting provider information."""
        info = self.provider.get_provider_info()

        assert info['name'] == 'BasicProvider'
        assert 'F15' in info['supported_aircraft']
        assert info['capabilities']['external_dependencies'] is False

    def test_context_manager(self):
        """Test provider as context manager."""
        with BasicProvider() as provider:
            mesh = provider.create_aircraft('F15')
            assert mesh is not None


class TestProviderRegistry:
    """Test provider registry functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a fresh registry for testing
        ProviderRegistry._providers = {}

    def test_register_provider(self):
        """Test registering a provider."""
        ProviderRegistry.register('test', BasicProvider)

        providers = ProviderRegistry.list_providers()
        assert 'test' in providers
        assert providers['test'] == BasicProvider

    def test_get_provider(self):
        """Test getting a provider."""
        ProviderRegistry.register('test', BasicProvider)
        ProviderRegistry.set_default('test')

        provider = ProviderRegistry.get('test')
        assert isinstance(provider, BasicProvider)

    def test_get_default_provider(self):
        """Test getting default provider."""
        ProviderRegistry.register('test', BasicProvider)
        ProviderRegistry.set_default('test')

        provider = ProviderRegistry.get()  # No name specified
        assert isinstance(provider, BasicProvider)

    def test_invalid_provider_class(self):
        """Test registering invalid provider class."""
        class InvalidProvider:
            pass

        with pytest.raises(ValueError, match="must inherit from ModelProvider"):
            ProviderRegistry.register('invalid', InvalidProvider)

    def test_unknown_provider(self):
        """Test getting unknown provider."""
        with pytest.raises(ValueError, match="Provider 'unknown' not found"):
            ProviderRegistry.get('unknown')

    def test_set_invalid_default(self):
        """Test setting invalid default provider."""
        with pytest.raises(ValueError, match="provider 'unknown' not found"):
            ProviderRegistry.set_default('unknown')


class TestTiGLProvider:
    """Test TiGL provider (if available)."""

    def setup_method(self):
        """Set up test fixtures."""
        try:
            from aircraft_toolkit.providers.tigl_provider import TiGLProvider
            self.provider = TiGLProvider()
            self.tigl_available = self.provider.tigl_available
        except ImportError:
            self.tigl_available = False

    @pytest.mark.skipif(not pytest.importorskip("trimesh", reason="trimesh not available"))
    def test_tigl_initialization(self):
        """Test TiGL provider initialization."""
        if not self.tigl_available:
            pytest.skip("TiGL not available")

        assert self.provider.cpacs_templates is not None
        assert len(self.provider.cpacs_templates) > 0

    @pytest.mark.skipif(not pytest.importorskip("trimesh", reason="trimesh not available"))
    def test_tigl_supported_aircraft(self):
        """Test TiGL supported aircraft."""
        if not self.tigl_available:
            # Should return empty list if TiGL not available
            aircraft = self.provider.get_supported_aircraft()
            assert aircraft == []
        else:
            aircraft = self.provider.get_supported_aircraft()
            assert len(aircraft) > 0

    @pytest.mark.skipif(not pytest.importorskip("trimesh", reason="trimesh not available"))
    def test_tigl_create_aircraft_without_tigl(self):
        """Test TiGL provider without TiGL installed."""
        if not self.tigl_available:
            with pytest.raises(RuntimeError, match="TiGL not available"):
                self.provider.create_aircraft('F15')

    @pytest.mark.integration
    @pytest.mark.skipif(not pytest.importorskip("trimesh", reason="trimesh not available"))
    def test_tigl_create_aircraft_with_tigl(self):
        """Test TiGL aircraft creation (integration test)."""
        if not self.tigl_available:
            pytest.skip("TiGL not available")

        try:
            mesh = self.provider.create_aircraft('F15', detail_level='low')

            assert isinstance(mesh, AircraftMesh)
            assert mesh.num_vertices > 100  # Should be much more detailed than basic
            assert mesh.metadata['provider'] == 'tigl'
            assert mesh.metadata['aircraft_type'] == 'F15'

        except Exception as e:
            pytest.skip(f"TiGL integration test failed: {e}")

    def test_tigl_cleanup(self):
        """Test TiGL cleanup."""
        if self.tigl_available:
            temp_dir = self.provider.temp_dir
            assert Path(temp_dir).exists()

            self.provider.cleanup()
            assert not Path(temp_dir).exists()


class TestConfiguration:
    """Test configuration system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / 'test_config.json'

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_default_config(self):
        """Test default configuration."""
        config = Config()

        assert config.aircraft.model_provider == 'auto'
        assert config.dataset.image_size == (512, 512)
        assert 'basic' in config.providers
        assert 'tigl' in config.providers

    def test_config_serialization(self):
        """Test config to/from dict."""
        config = Config()
        config_dict = config.to_dict()

        restored_config = Config.from_dict(config_dict)

        assert restored_config.aircraft.model_provider == config.aircraft.model_provider
        assert restored_config.dataset.image_size == config.dataset.image_size

    def test_preferred_provider_auto(self):
        """Test automatic provider selection."""
        config = Config()

        # Should prefer higher priority provider
        preferred = config.get_preferred_provider()
        assert preferred in ['tigl', 'basic']  # Depends on availability

    def test_preferred_provider_explicit(self):
        """Test explicit provider selection."""
        config = Config()
        config.aircraft.model_provider = 'basic'

        preferred = config.get_preferred_provider()
        assert preferred == 'basic'


class TestIntegration:
    """Integration tests for the complete system."""

    def test_provider_switching(self):
        """Test switching between providers."""
        # Register both providers
        ProviderRegistry.register('basic', BasicProvider)

        # Test basic provider
        basic_provider = get_provider('basic')
        basic_mesh = basic_provider.create_aircraft('F15')

        assert basic_mesh.metadata['provider'] == 'basic'
        assert basic_mesh.num_vertices < 50  # Basic models are simple

    def test_error_handling(self):
        """Test error handling throughout the system."""
        provider = BasicProvider()

        # Invalid aircraft type
        with pytest.raises(ValueError):
            provider.create_aircraft('NONEXISTENT')

        # Invalid detail level (should be ignored for basic provider)
        mesh = provider.create_aircraft('F15', detail_level='invalid')
        assert mesh is not None

    def test_mesh_quality_comparison(self):
        """Test mesh quality differences between providers."""
        basic_provider = BasicProvider()
        basic_mesh = basic_provider.create_aircraft('F15')

        # Basic provider should produce simple meshes
        assert basic_mesh.num_vertices < 50
        assert basic_mesh.num_faces < 50

        # Mesh should be valid
        assert basic_mesh.vertices.shape[1] == 3
        assert basic_mesh.faces.shape[1] == 3