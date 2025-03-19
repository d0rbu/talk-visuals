import os

import torch as th
from PIL import Image

from functools import reduce

from preprocessing.mask import color_mask, alpha_channel
from preprocessing.erode_dilate import erode, dilate

COLORS = (
    (28, 151, 179),
    (157, 157, 156),
    (219, 8, 18),
)

UK_AND_IRELAND_PATH = "media/uk_and_ireland.png"


def isolate_ireland() -> None:
    assert os.path.exists(UK_AND_IRELAND_PATH), f"File not found: {UK_AND_IRELAND_PATH}"

    uk_and_ireland_image = Image.open(UK_AND_IRELAND_PATH)
    uk_masks = [color_mask(uk_and_ireland_image, color) for color in COLORS]
    uk_mask = reduce(lambda mask_0, mask_1: mask_0 | mask_1, uk_masks)
    uk_mask_clean = erode(dilate(erode(uk_mask), kernel_size=7), kernel_size=5)


    original_alpha = alpha_channel(uk_and_ireland_image)
    combined_alpha = original_alpha * uk_mask_clean
    uk_mask_image = Image.fromarray(combined_alpha.to(th.uint8).numpy())
    uk_mask_image.save("media/uk_mask.png")

    uk_image = uk_and_ireland_image.copy()
    uk_image.putalpha(uk_mask_image)

    # zoom into the opaque region
    uk_bbox = uk_image.getbbox()
    uk_image = uk_image.crop(uk_bbox)

    uk_image.convert("RGBA").save("media/uk.png")


if __name__ == "__main__":
    isolate_ireland()
