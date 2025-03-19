import torch as th


def filter_for_n_neighbors(
    mask: th.BoolTensor,
    n_neighbors: int = 1,
    kernel_size: int = 3,
) -> th.BoolTensor:
    """
    Filter binary masks to keep only pixels with at least specific number of neighbors, including themselves.

    Args:
        mask (th.BoolTensor): Input binary masks of shape (B, H, W) or (H, W).
        n_neighbors (int): Number of neighbors for a pixel to be kept.
        kernel_size (int): Size of the square structuring element.
        iterations (int): Number of times to apply the erosion.

    Returns:
        th.BoolTensor: Filtered binary mask.
    """
    match mask.ndim:
        case 2:
            batched_mask = mask.unsqueeze(0)
        case 3:
            batched_mask = mask
        case _:
            raise ValueError(
                "Mask must be a 2D or 3D tensor of shape (H, W) or (B, H, W)."
            )

    assert n_neighbors > 0 and n_neighbors <= kernel_size**2, (
        "Invalid number of neighbors, must be in [1, kernel_size ** 2]."
    )
    assert kernel_size % 2 == 1, "Kernel size must be odd."

    kernel = th.ones(kernel_size, kernel_size, device=batched_mask.device)
    neighbors = th.nn.functional.conv2d(
        batched_mask.int(),
        kernel.view(1, 1, kernel_size, kernel_size).int(),
        padding=kernel_size // 2,
    )
    modified_mask = neighbors >= n_neighbors

    match mask.ndim:
        case 2:
            return modified_mask.squeeze(0)
        case 3:
            return modified_mask


def dilate(
    mask: th.BoolTensor, kernel_size: int = 3, iterations: int = 1
) -> th.BoolTensor:
    """
    Dilate a binary mask using a square structuring element.
    It filters to pixels that have at least one neighbor, meaning they are adjacent to foreground.
    """

    for _ in range(iterations):
        mask = filter_for_n_neighbors(
            mask, n_neighbors=1, kernel_size=kernel_size
        )

    return mask


def erode(
    mask: th.BoolTensor, kernel_size: int = 3, iterations: int = 1
) -> th.BoolTensor:
    """
    Erode a binary mask using a square structuring element.
    It filters to pixels that have an equal number of neighbors to the kernel size squared, meaning they are surrounded by foreground.
    """

    for _ in range(iterations):
        mask = filter_for_n_neighbors(mask, n_neighbors=kernel_size**2, kernel_size=kernel_size)

    return mask
