from typing import Self

import torch as th
from opensimplex import OpenSimplex


def hyperspheric_noise(angles: th.Tensor, seed: int = 0) -> float:
    """
    Defines a noise pattern as a continuous mapping from the unit hypersphere to R.
    Returns the value of the noise at the given angles.

    Args:
        angles (th.Tensor): The angles defining the point on the hypersphere.
        seed (int): The seed for the noise.

    Returns:
        float: The value of the noise at the given angles, from -1 to 1.
    """

    # Convert hyperspherical angles to Cartesian coordinates on the unit hypersphere
    dims = angles.shape[0] + 1  # The hypersphere is one dimension higher than the angles provided
    coords = th.zeros(dims, dtype=angles.dtype)

    coords[0] = th.cos(angles[0])
    prod = th.sin(angles[0])

    for i in range(1, dims - 1):
        coords[i] = prod * th.cos(angles[i])
        prod *= th.sin(angles[i])

    coords[-1] = prod  # Last coordinate

    # Convert to numpy for OpenSimplex noise
    coords_np = coords.numpy()

    # Generate noise using OpenSimplex
    noise_gen = OpenSimplex(seed)
    noise_value = noise_gen.noiseN(coords_np)

    return noise_value
