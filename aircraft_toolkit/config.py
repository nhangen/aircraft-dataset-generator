"""
Configuration management for Aircraft Dataset Generator.

This module handles configuration loading, validation, and provider
selection for the aircraft dataset generation toolkit.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    name: str
    enabled: bool = True
    config: Optional[Dict[str, Any]] = None
    detail_level: str = 'medium'
    priority: int = 1  # Higher priority = preferred provider

    def __post_init__(self):
        if self.config is None:
            self.config = {}


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    output_format: str = 'custom_3d'
    image_size: tuple = (512, 512)
    views_per_scene: int = 8
    camera_distance: tuple = (8, 12)
    camera_height_range: tuple = (-5, 10)
    include_depth_maps: bool = True
    include_surface_normals: bool = False
    background_color: tuple = (135, 206, 235)  # Sky blue


@dataclass
class AircraftConfig:
    """Configuration for aircraft generation."""
    model_provider: str = 'auto'  # 'auto', 'basic', 'tigl', etc.
    detail_level: str = 'medium'
    scaling_factor: float = 10.0
    center_models: bool = True
    compute_normals: bool = True


@dataclass
class Config:
    """Main configuration class."""
    providers: Dict[str, ProviderConfig] = None
    dataset: DatasetConfig = None
    aircraft: AircraftConfig = None

    def __post_init__(self):
        if self.providers is None:
            self.providers = self._get_default_providers()
        if self.dataset is None:
            self.dataset = DatasetConfig()
        if self.aircraft is None:
            self.aircraft = AircraftConfig()

    def _get_default_providers(self) -> Dict[str, ProviderConfig]:
        """Get default provider configurations."""
        providers = {
            'basic': ProviderConfig(
                name='basic',
                enabled=True,
                priority=1,
                detail_level='low'
            )
        }

        # Add PyVista provider if available
        try:
            import pyvista
            providers['pyvista'] = ProviderConfig(
                name='pyvista',
                enabled=True,
                priority=100,  # Highest priority - uses real models
                detail_level='high'
            )
        except ImportError:
            pass

        return providers

    def get_preferred_provider(self) -> str:
        """
        Get the preferred provider based on availability and priority.

        Returns:
            Name of preferred provider
        """
        if self.aircraft.model_provider != 'auto':
            return self.aircraft.model_provider

        # Import here to avoid circular imports
        from .providers import list_providers

        available_providers = list_providers()
        enabled_providers = [
            (name, config) for name, config in self.providers.items()
            if config.enabled and name in available_providers
        ]

        if not enabled_providers:
            raise RuntimeError("No enabled providers available")

        # Sort by priority (highest first)
        enabled_providers.sort(key=lambda x: x[1].priority, reverse=True)

        return enabled_providers[0][0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        providers = {}
        if 'providers' in data:
            for name, provider_data in data['providers'].items():
                providers[name] = ProviderConfig(**provider_data)

        dataset_config = DatasetConfig()
        if 'dataset' in data:
            dataset_config = DatasetConfig(**data['dataset'])

        aircraft_config = AircraftConfig()
        if 'aircraft' in data:
            aircraft_config = AircraftConfig(**data['aircraft'])

        return cls(
            providers=providers,
            dataset=dataset_config,
            aircraft=aircraft_config
        )


class ConfigManager:
    """Manages configuration loading and saving."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize config manager.

        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or self._get_default_config_file()
        self._config = None

    def _get_default_config_file(self) -> str:
        """Get default configuration file path."""
        # Try user config directory first
        config_dir = Path.home() / '.aircraft_toolkit'
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / 'config.json')

    def load_config(self) -> Config:
        """
        Load configuration from file.

        Returns:
            Configuration object
        """
        if self._config is not None:
            return self._config

        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                self._config = Config.from_dict(data)
                logger.info(f"Loaded configuration from {self.config_file}")
            else:
                self._config = Config()
                logger.info("Using default configuration")

        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            self._config = Config()

        return self._config

    def save_config(self, config: Config):
        """
        Save configuration to file.

        Args:
            config: Configuration to save
        """
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            self._config = config
            logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def get_config(self) -> Config:
        """Get current configuration."""
        return self.load_config()

    def update_provider_config(self, provider_name: str, config: Dict[str, Any]):
        """
        Update configuration for a specific provider.

        Args:
            provider_name: Name of provider to update
            config: New configuration values
        """
        current_config = self.get_config()
        if provider_name not in current_config.providers:
            current_config.providers[provider_name] = ProviderConfig(name=provider_name)

        # Update configuration
        provider_config = current_config.providers[provider_name]
        for key, value in config.items():
            if hasattr(provider_config, key):
                setattr(provider_config, key, value)
            else:
                provider_config.config[key] = value

        self.save_config(current_config)


# Global config manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> Config:
    """Get current global configuration."""
    return get_config_manager().get_config()