"""
Validation utilities for Aircraft Dataset Generator.
"""

from typing import List, Union

# Supported aircraft types
SUPPORTED_AIRCRAFT_TYPES = ['F15', 'B52', 'C130']

# Supported output formats
SUPPORTED_OUTPUT_FORMATS = ['coco', 'yolo', 'pascal_voc', 'custom_3d']

# Supported annotation formats
SUPPORTED_ANNOTATION_FORMATS = ['coco', 'yolo', 'pascal_voc']


def validate_aircraft_types(aircraft_types: List[str]) -> List[str]:
    """
    Validate and normalize aircraft type names.

    Args:
        aircraft_types: List of aircraft type strings

    Returns:
        List of validated and normalized aircraft types

    Raises:
        ValueError: If any aircraft type is not supported
    """
    if not aircraft_types:
        raise ValueError("At least one aircraft type must be specified")

    normalized = []
    for aircraft_type in aircraft_types:
        aircraft_upper = aircraft_type.upper()
        if aircraft_upper not in SUPPORTED_AIRCRAFT_TYPES:
            raise ValueError(
                f"Unsupported aircraft type: {aircraft_type}. "
                f"Supported types: {SUPPORTED_AIRCRAFT_TYPES}"
            )
        normalized.append(aircraft_upper)

    return normalized


def validate_output_format(output_format: str) -> str:
    """
    Validate output format.

    Args:
        output_format: Output format string

    Returns:
        Validated output format

    Raises:
        ValueError: If output format is not supported
    """
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output format: {output_format}. "
            f"Supported formats: {SUPPORTED_OUTPUT_FORMATS}"
        )
    return output_format


def validate_annotation_format(annotation_format: str) -> str:
    """
    Validate annotation format.

    Args:
        annotation_format: Annotation format string

    Returns:
        Validated annotation format

    Raises:
        ValueError: If annotation format is not supported
    """
    if annotation_format not in SUPPORTED_ANNOTATION_FORMATS:
        raise ValueError(
            f"Unsupported annotation format: {annotation_format}. "
            f"Supported formats: {SUPPORTED_ANNOTATION_FORMATS}"
        )
    return annotation_format


def validate_image_size(image_size: Union[tuple, int]) -> tuple:
    """
    Validate and normalize image size.

    Args:
        image_size: Image size as (width, height) tuple or single integer

    Returns:
        Validated image size as (width, height) tuple

    Raises:
        ValueError: If image size is invalid
    """
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("Image size must be positive")
        return (image_size, image_size)

    elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        return (width, height)

    else:
        raise ValueError(
            "Image size must be a positive integer or (width, height) tuple"
        )