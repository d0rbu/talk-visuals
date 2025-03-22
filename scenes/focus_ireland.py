from typing import Self

import numpy as np
from manim import BLUE, RED, ImageMobject, MovingCameraScene, color_to_int_rgba
from PIL import Image


class FocusIreland(MovingCameraScene):
    def construct(self: Self) -> None:
        uk_image, ireland_image = self.get_uk_and_ireland_images()

        start_height = 6

        uk_scene_image = ImageMobject(uk_image)
        ireland_scene_image = ImageMobject(ireland_image)
        uk_scene_image.set_height(start_height)
        ireland_scene_image.set_height(start_height)

        self.add(uk_scene_image)
        self.add(ireland_scene_image)
        self.wait(6)

        uk_fade = uk_scene_image.animate.set_opacity(0)

        # zoom in on ireland
        ireland_bbox = self.get_world_space_bbox(ireland_image, ireland_scene_image)
        ireland_center = np.array(
            [
                (ireland_bbox[0] + ireland_bbox[1]) / 2,
                (ireland_bbox[2] + ireland_bbox[3]) / 2,
                0,
            ]
        )
        camera_transform = self.camera.frame.animate.scale(0.5).move_to(ireland_center)

        self.play(camera_transform, uk_fade)
        self.wait(2)

        self.play(ireland_scene_image.animate.set_opacity(0), run_time=0.5)

    @staticmethod
    def get_world_space_bbox(
        image: Image.Image, scene_image: ImageMobject
    ) -> tuple[float, float, float, float]:
        bbox_image_space = np.array(image.getbbox())

        # convert image-space bbox to world-space
        x_min, y_min, x_max, y_max = bbox_image_space

        # flip y-coordinates since manim has y+ going up and most image libraries have y+ going down
        y_min, y_max = image.height - y_max, image.height - y_min

        # normalize relative to the image size
        x_min /= image.width
        x_max /= image.width
        y_min /= image.height
        y_max /= image.height

        # make the coordinates relative to center rather than bottom left
        x_min -= 0.5
        x_max -= 0.5
        y_min -= 0.5
        y_max -= 0.5

        world_space_center_x, world_space_center_y, _ = scene_image.get_center()

        # convert to world-space coordinates
        x_min = scene_image.width * x_min + world_space_center_x
        x_max = scene_image.width * x_max + world_space_center_x
        y_min = scene_image.height * y_min + world_space_center_y
        y_max = scene_image.height * y_max + world_space_center_y

        return x_min, x_max, y_min, y_max

    @staticmethod
    def get_uk_and_ireland_images() -> tuple[Image.Image, Image.Image]:
        uk_and_ireland_image = Image.open("media/uk_and_ireland.png")
        uk_mask_image = Image.open("media/uk_mask.png")
        ireland_mask_image = Image.open("media/ireland_mask.png")

        uk_image = uk_and_ireland_image.copy().convert("RGBA")
        uk_image.paste(
            tuple(color_to_int_rgba(BLUE)), (0, 0, uk_image.width, uk_image.height)
        )
        uk_image.putalpha(uk_mask_image)

        ireland_image = uk_and_ireland_image.copy().convert("RGBA")
        ireland_image.paste(
            tuple(color_to_int_rgba(RED)),
            (0, 0, ireland_image.width, ireland_image.height),
        )
        ireland_image.putalpha(ireland_mask_image)

        return uk_image, ireland_image
