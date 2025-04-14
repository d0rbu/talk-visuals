from typing import Self

import torch as th
import opensimplex


def hyperspheric_noise(
    angles: th.Tensor, seed: int = 0
) -> th.Tensor:
    assert angles.ndim == 2 or angles.ndim == 1, "angles should be 1D or 2D tensor"

    if angles.ndim == 1:
        # If angles is 1D, we need to add a batch dimension
        angles = angles.unsqueeze(0)

    return hyperspheric_noise_array(angles, seed=seed).squeeze(0) if angles.shape[0] == 1 else hyperspheric_noise_array(angles, seed=seed)


def hyperspheric_noise_array(angles: th.Tensor, seed: int = 0) -> th.Tensor:
    """
    Defines a noise pattern as a continuous mapping from the unit hypersphere to R.
    Returns the value of the noise at the given angles.

    Args:
        angles (th.Tensor): The angles defining the point on the hypersphere. This should be [theta, phi, ...] in radians.
        seed (int): The seed for the noise.

    Returns:
        th.Tensor: The value of the noise at the given angles, from -1 to 1.
    """

    # Convert hyperspherical angles to Cartesian coordinates on the unit hypersphere
    dims = angles.shape[1] + 1  # The hypersphere is one dimension higher than the angles provided
    coords = th.zeros(angles.shape[0], dims, dtype=angles.dtype)

    coords[:, 0] = th.cos(angles[:, 0])
    prod = th.sin(angles[:, 0])

    for i in range(1, dims - 1):
        coords[:, i] = prod * th.cos(angles[:, i])
        prod *= th.sin(angles[:, i])

    coords[:, -1] = prod  # Last coordinate
    # Convert to numpy for OpenSimplex noise
    coords_np = coords.numpy()

    # Generate noise using OpenSimplex
    opensimplex.seed(seed)
    match dims:
        case 2:
            return th.asarray(opensimplex.noise2array(
                x=coords_np[:, 0], y=coords_np[:, 1]
            ))
        case 3:
            return th.asarray(opensimplex.noise3array(
                x=coords_np[:, 1], y=coords_np[:, 2], z=coords_np[:, 0]
            ))
        case 4:
            return th.asarray(opensimplex.noise4array(
                x=coords_np[:, 0],
                y=coords_np[:, 1],
                z=coords_np[:, 2],
                w=coords_np[:, 3],
            ))
        case _:
            raise ValueError("Unsupported number of dimensions for hyperspheric noise.")
