"""Utility functions for canopy architecture analysis."""

import numpy as np
from typing import Union, Dict, Optional


def bin_leaf_angles(
    angles: Union[np.ndarray, list],
    angle_type: str = 'horizontal',
    n_bins: int = 18,
    bin_range: tuple = (0, 90),
    remove_empty: bool = True
) -> Dict[str, np.ndarray]:
    """Bin leaf angle measurements into frequency distribution.

    This preprocessing function converts raw leaf angle measurements into
    binned frequency data suitable for fitting with LeafAngleDistribution.

    Args:
        angles: Array or list of leaf angle measurements (degrees)
        angle_type: Type of angle measurement:
            - 'horizontal': Angle from horizontal (0° = horizontal, 90° = vertical)
            - 'zenith': Angle from zenith/vertical (0° = vertical, 90° = horizontal)
        n_bins: Number of bins for histogram (default: 18, giving 5° bins)
        bin_range: Tuple of (min, max) angles in degrees (default: (0, 90))
        remove_empty: Whether to remove bins with zero frequency (default: True)

    Returns:
        Dictionary with:
            'x': Bin center angles (degrees from horizontal)
            'y': Frequency density (probability per degree)

    Examples:
        >>> from phytorch import fit
        >>> from phytorch.models.canopy import LeafAngleDistribution, bin_leaf_angles
        >>>
        >>> # Raw zenith angle measurements
        >>> zenith_angles = [65, 71, 68, 60, 71, 84, 68, 80, 76, 87]
        >>>
        >>> # Preprocess into binned data
        >>> data = bin_leaf_angles(zenith_angles, angle_type='zenith')
        >>>
        >>> # Fit with standard API
        >>> result = fit(LeafAngleDistribution(), data)
        >>> print(result.canopy_type)
    """
    angles = np.asarray(angles, dtype=float)

    # Convert angle convention if needed
    if angle_type == 'zenith':
        angles = 90.0 - angles
    elif angle_type == 'horizontal':
        pass  # Already in correct convention
    else:
        raise ValueError(
            f"angle_type must be 'horizontal' or 'zenith', got '{angle_type}'"
        )

    # Create histogram bins
    bin_edges = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1] - bin_edges[0]

    # Compute histogram
    counts, _ = np.histogram(angles, bins=bin_edges)

    # Convert to frequency density (probability per degree)
    # Normalize so integral over range equals 1
    total_count = np.sum(counts)
    if total_count == 0:
        raise ValueError("No valid angle measurements in specified range")

    frequency = counts / (total_count * bin_width)

    # Remove empty bins if requested
    if remove_empty:
        non_zero = frequency > 0
        bin_centers = bin_centers[non_zero]
        frequency = frequency[non_zero]

    return {
        'x': bin_centers,
        'y': frequency
    }


def angle_statistics(
    angles: Union[np.ndarray, list],
    angle_type: str = 'horizontal'
) -> Dict[str, float]:
    """Compute descriptive statistics for leaf angle measurements.

    Args:
        angles: Array or list of leaf angle measurements (degrees)
        angle_type: Type of angle measurement ('horizontal' or 'zenith')

    Returns:
        Dictionary with statistical measures:
            - mean: Mean angle (degrees)
            - median: Median angle (degrees)
            - std: Standard deviation (degrees)
            - min: Minimum angle (degrees)
            - max: Maximum angle (degrees)
            - n: Number of measurements

    Examples:
        >>> from phytorch.models.canopy import angle_statistics
        >>> stats = angle_statistics([65, 71, 68, 60], angle_type='zenith')
        >>> print(f"Mean: {stats['mean']:.1f}°")
    """
    angles = np.asarray(angles, dtype=float)

    # Convert to horizontal convention for consistency
    if angle_type == 'zenith':
        angles = 90.0 - angles

    return {
        'mean': float(np.mean(angles)),
        'median': float(np.median(angles)),
        'std': float(np.std(angles)),
        'min': float(np.min(angles)),
        'max': float(np.max(angles)),
        'n': len(angles)
    }
