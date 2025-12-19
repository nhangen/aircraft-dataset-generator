"""
Provider registry for managing and accessing model providers.

This module implements a registry pattern for model providers,
allowing dynamic registration and retrieval of providers.
"""

import logging
from typing import Dict, Optional, Type

from .base import ModelProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Singleton registry for model providers.

    This class manages the registration and retrieval of model providers,
    ensuring a single source of truth for available providers.
    """

    _instance = None
    _providers: Dict[str, Type[ModelProvider]] = {}
    _default_provider: str = 'basic'

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str, provider_class: Type[ModelProvider]):
        """
        Register a new provider.

        Args:
            name: Unique name for the provider
            provider_class: Provider class (must inherit from ModelProvider)

        Raises:
            ValueError: If name is already registered or provider is invalid
        """
        if not issubclass(provider_class, ModelProvider):
            raise ValueError(
                f"Provider class {provider_class} must inherit from ModelProvider"
            )

        if name in cls._providers:
            logger.warning(f"Overwriting existing provider: {name}")

        cls._providers[name] = provider_class
        logger.info(f"Registered provider: {name} ({provider_class.__name__})")

    @classmethod
    def get(cls, name: Optional[str] = None, config: Optional[Dict] = None) -> ModelProvider:
        """
        Get a provider instance.

        Args:
            name: Provider name (uses default if None)
            config: Configuration dictionary for the provider

        Returns:
            Instantiated provider

        Raises:
            ValueError: If provider name is not found
        """
        if name is None:
            name = cls._default_provider

        if name not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(
                f"Provider '{name}' not found. Available providers: {available}"
            )

        provider_class = cls._providers[name]
        return provider_class(config=config)

    @classmethod
    def list_providers(cls) -> Dict[str, Type[ModelProvider]]:
        """
        Get all registered providers.

        Returns:
            Dictionary mapping provider names to classes
        """
        return cls._providers.copy()

    @classmethod
    def set_default(cls, name: str):
        """
        Set the default provider.

        Args:
            name: Name of provider to use as default

        Raises:
            ValueError: If provider name is not found
        """
        if name not in cls._providers:
            raise ValueError(f"Cannot set default: provider '{name}' not found")
        cls._default_provider = name
        logger.info(f"Default provider set to: {name}")


# Convenience functions
def register_provider(name: str, provider_class: Type[ModelProvider]):
    """
    Register a provider with the global registry.

    Args:
        name: Unique name for the provider
        provider_class: Provider class
    """
    ProviderRegistry.register(name, provider_class)


def get_provider(name: Optional[str] = None, config: Optional[Dict] = None) -> ModelProvider:
    """
    Get a provider from the global registry.

    Args:
        name: Provider name (uses default if None)
        config: Configuration dictionary

    Returns:
        Instantiated provider
    """
    return ProviderRegistry.get(name, config)


def list_providers() -> Dict[str, Type[ModelProvider]]:
    """
    List all registered providers.

    Returns:
        Dictionary mapping provider names to classes
    """
    return ProviderRegistry.list_providers()


def set_default_provider(name: str):
    """
    Set the default provider.

    Args:
        name: Name of provider to use as default
    """
    ProviderRegistry.set_default(name)