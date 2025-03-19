import os

import torch as th
from PIL import Image

from preprocessing.mask import color_mask, alpha_channel
from preprocessing.erode_dilate import erode, dilate

COLOR = (183, 209, 108)
UK_AND_IRELAND_PATH = "media/uk_and_ireland.png"


def isolate_ireland() -> None:
    assert os.path.exists(UK_AND_IRELAND_PATH), f"File not found: {UK_AND_IRELAND_PATH}"

    uk_and_ireland_image = Image.open(UK_AND_IRELAND_PATH)
    ireland_mask = color_mask(uk_and_ireland_image, COLOR)
    ireland_mask_clean = dilate(erode(ireland_mask))

    original_alpha = alpha_channel(uk_and_ireland_image)
    combined_alpha = original_alpha * ireland_mask_clean
    ireland_mask_image = Image.fromarray(combined_alpha.to(th.uint8).numpy())
    ireland_mask_image.save("media/ireland_mask.png")

    ireland_image = uk_and_ireland_image.copy()
    ireland_image.putalpha(ireland_mask_image)

    # zoom into the opaque region
    ireland_bbox = ireland_image.getbbox()
    ireland_image = ireland_image.crop(ireland_bbox)

    ireland_image.convert("RGBA").save("media/ireland.png")


if __name__ == "__main__":
    isolate_ireland()
