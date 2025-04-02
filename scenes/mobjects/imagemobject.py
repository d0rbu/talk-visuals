from typing import Self, Any

from manim import ImageMobject as ManimImageMobject


class ImageMobject(ManimImageMobject):
    def set_opacity(
        self: Self,
        alpha: float,
    ) -> Self:
        """Sets the image's opacity.

        Parameters
        ----------
        alpha
            The alpha value of the object, 1 being opaque and 0 being
            transparent.
        """
        self.pixel_array[:, :, 3] = (self.pixel_array[:, :, 3] * alpha).astype(int)
        self.fill_opacity = alpha
        self.stroke_opacity = alpha
        return self
