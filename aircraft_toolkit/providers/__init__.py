"""
Aircraft Model Providers Module

This module provides a modular architecture for integrating different
3D aircraft model generation libraries. The design allows for easy
switching between providers and addition of new providers without
affecting the core dataset generation logic.

Available Providers:
    - BasicProvider: Hand-coded simple meshes (original implementation)
    - TiGLProvider: High-quality CPACS-based parametric models
    - Future: CadQueryProvider, Open3DProvider, etc.

Usage:
    from aircraft_toolkit.providers import get_provider

    # Get default provider
    provider = get_provider()

    # Get specific provider
    provider = get_provider('tigl')

    # Generate aircraft model
    model = provider.create_aircraft('F15')
"""

from .base import ModelProvider, AircraftMesh
from .basic import BasicProvider
from .registry import ProviderRegistry, get_provider, register_provider, list_providers

# Try to register PyVista provider
try:
    from .pyvista_models_provider import PyVistaModelsProvider
    register_provider('pyvista', PyVistaModelsProvider)
except ImportError:
    pass

# Register default providers
register_provider('basic', BasicProvider)

__all__ = [
    'ModelProvider',
    'AircraftMesh',
    'BasicProvider',
    'get_provider',
    'register_provider',
    'list_providers',
]