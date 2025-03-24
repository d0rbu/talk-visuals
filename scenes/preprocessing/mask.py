import torch as th
from PIL import Image


def color_mask(
    image: Image.Image,
    color: tuple[int, int, int],
    alpha_threshold: int = 16,
    delta: float = 10.0,
) -> th.BoolTensor:
    """
    Create a mask for a specific color in an image, excluding pixels with alpha below a threshold.

    Args:
        image (PIL.Image.Image): Input image.
        color (tuple[int, int, int]): RGB color to mask.
        alpha_threshold (int): Minimum alpha value for a pixel to be included in the mask (0-255).
        delta (float): Maximum distance between a pixel and the target color to be included in the mask.

    Returns:
        th.BoolTensor: A boolean tensor of shape (H, W) where True indicates pixels matching the color and meeting alpha criteria.
    """
    # rgba format is rgb + alpha, which encodes transparency
    image = image.convert("RGBA")
    w, h = image.size
    pixels = th.tensor(image.getdata(), dtype=th.uint8).reshape(
        h, w, 4
    )  # shape: (H, W, 4)

    rgb_tensor = pixels[:, :, :3]  # shape: (H, W, 3)
    alpha_tensor = pixels[:, :, 3]  # shape: (H, W)

    color_tensor = th.tensor(color, dtype=th.uint8)  # shape: (3,)
    # distance of each pixel to the target color
    color_diff = th.pairwise_distance(
        rgb_tensor, color_tensor, p=2, keepdim=False
    )  # shape: (H, W)

    # keep pixels that are close to the target color
    color_mask = color_diff < delta

    # keep pixels that have enough opacity
    alpha_mask = alpha_tensor > alpha_threshold

    return color_mask & alpha_mask


def alpha_channel(image: Image.Image) -> th.ByteTensor:
    """
    Extract the alpha channel from an RGBA image.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        th.ByteTensor: A byte tensor of shape (H, W) representing the alpha channel.
    """
    image = image.convert("RGBA")
    w, h = image.size
    pixels = th.tensor(image.getdata(), dtype=th.uint8).reshape(
        h, w, 4
    )  # shape: (H, W, 4)

    return pixels[:, :, 3]  # shape: (H, W)
