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
from .tigl_provider import TiGLProvider
from .registry import ProviderRegistry, get_provider, register_provider, list_providers

# Register default providers
register_provider('basic', BasicProvider)
register_provider('tigl', TiGLProvider)

__all__ = [
    'ModelProvider',
    'AircraftMesh',
    'BasicProvider',
    'TiGLProvider',
    'get_provider',
    'register_provider',
    'list_providers',
]