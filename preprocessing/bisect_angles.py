import torch as th


def bisect_angles(points: th.Tensor, angles: th.Tensor) -> th.Tensor:
    """
    Given a set of points and a set of angles, return the bias for each hyperplane such that it bisects the points.
    This is possible because of the ham sandwich theorem!

    Args:
        points (th.Tensor): A set of points in 2D.
        angles (th.Tensor): A set of angles in radians.

    Returns:
        biases (th.Tensor): A set of biases for each hyperplane.

    Example:
        >>> points = th.tensor([[0, 0], [1, 1], [2, 0]])
        >>> angles = th.tensor([0, th.pi / 2, th.pi])
        >>> bisect_angles(points, angles)
        tensor([0.0000, 0.5000, 1.000])
    """
    assert points.ndim == 2 and points.shape[1] == 2, (
        "Points must be a 2D tensor with shape (N, 2)."
    )
    assert angles.ndim == 1, "Angles must be a 1D tensor."

    normals = th.stack([th.cos(angles), th.sin(angles)], dim=1)  # shape: (A, 2)

    # the signed distance of a point to a hyperplane is the dot product of the point and the unit normal + the bias
    # but for simplicity we take hyperplanes crossing the origin so the bias is 0
    signed_distances = points @ normals.T  # shape: (N, A)

    # the median of the signed distances is the bias of the hyperplane that bisects the points
    biases = th.median(signed_distances, dim=0).values  # shape: (A,)

    return biases
