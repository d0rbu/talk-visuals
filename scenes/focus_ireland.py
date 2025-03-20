from typing import Self

from manim import Scene, ImageMobject


class FocusIreland(Scene):
    UK_AND_IRELAND_PATH = "media/uk_and_ireland.png"
    UK_MASK_PATH = "media/uk_mask.png"
    IRELAND_MASK_PATH = "media/ireland_mask.png"

    def construct(self: Self) -> None:
        uk_and_ireland_image = ImageMobject(self.UK_AND_IRELAND_PATH)
        uk_mask_image = ImageMobject(self.UK_MASK_PATH)
        ireland_mask_image = ImageMobject(self.IRELAND_MASK_PATH)

        uk_image = uk_and_ireland_image.copy()
        uk_image.pixel_array = uk_image.pixel_array * (uk_mask_image.get_pixel_array() / 255)

        ireland_image = uk_and_ireland_image.copy()
        ireland_image.pixel_array = ireland_image.pixel_array * (ireland_mask_image.get_pixel_array() / 255)

        self.add(uk_image)
        self.wait()
        self.add(ireland_image)
        self.wait()
