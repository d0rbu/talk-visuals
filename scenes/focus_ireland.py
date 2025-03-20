from typing import Self

import numpy as np
from manim import ImageMobject, MovingCameraScene


class FocusIreland(MovingCameraScene):
    UK_PATH = "media/uk.png"
    IRELAND_PATH = "media/ireland.png"

    START_HEIGHT = 6
    IRELAND_CENTER = np.array([-1.39530062, -0.72845888, 0.])
    UK_CENTER = np.array([0.607, -0.0005, 0.])
    IRELAND_SCALE_FACTOR = 0.3975
    ZOOM_FACTOR = 0.4

    def construct(self: Self) -> None:
        uk_image = ImageMobject(self.UK_PATH).set_height(self.START_HEIGHT).move_to(self.UK_CENTER)
        ireland_image = ImageMobject(self.IRELAND_PATH).set_height(self.START_HEIGHT * self.IRELAND_SCALE_FACTOR).move_to(self.IRELAND_CENTER)

        self.add(uk_image, ireland_image)
        self.wait()

        # center the camera on ireland
        camera_transform = self.camera.frame.animate.scale(self.ZOOM_FACTOR).move_to(ireland_image)

        # fade out uk
        uk_fade = uk_image.animate.set_opacity(0)

        self.play(camera_transform, uk_fade)
        self.wait()
