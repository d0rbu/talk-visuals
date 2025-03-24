from typing import Self

import numpy as np
from manim import BLUE, RED, ImageMobject, MovingCameraScene, Circle, ValueTracker, always_redraw, FadeIn, VGroup, WHITE, DR, UL, Line, PI, Arrow
from focus_ireland import FocusIreland


class IVTProof(MovingCameraScene):
    def construct(self: Self) -> None:
        uk_image, ireland_image = FocusIreland.get_uk_and_ireland_images()

        start_height = 6

        uk_scene_image = ImageMobject(uk_image)
        ireland_scene_image = ImageMobject(ireland_image)
        uk_scene_image.set_height(start_height)
        ireland_scene_image.set_height(start_height)

        # zoom in on ireland
        ireland_bbox = FocusIreland.get_world_space_bbox(ireland_image, ireland_scene_image)
        ireland_center = np.array(
            [
                (ireland_bbox[0] + ireland_bbox[1]) / 2,
                (ireland_bbox[2] + ireland_bbox[3]) / 2,
                0,
            ]
        )
        self.camera.frame.scale(0.5).move_to(ireland_center)

        self.play(FadeIn(ireland_scene_image))

        theta = ValueTracker(0)
        # draw a circle thats always on the bottom right of the screen
        angle_indicator = always_redraw(lambda: self.draw_angle_circle(theta))
        self.play(FadeIn(angle_indicator))
        self.play(theta.animate.set_value(PI / 8), run_time=0.2)
        self.play(theta.animate.set_value(-PI / 8), run_time=0.4)
        self.play(theta.animate.set_value(0), run_time=0.2)

        self.wait(2)

        origin = self.camera.frame.get_center()
        bias = ValueTracker(0)
        hyperplane = always_redraw(lambda: self.draw_hyperplane(origin, theta, bias, self.camera.frame.get_width()))

        self.play(FadeIn(hyperplane))
        self.play(theta.animate.set_value(PI / 16), run_time=0.2)
        self.play(theta.animate.set_value(-PI / 16), run_time=0.4)
        self.play(theta.animate.set_value(0), run_time=0.2)

        self.wait(5)

    def draw_angle_circle(self: Self, theta: ValueTracker) -> VGroup:
        relative_radius = 0.05
        radius = relative_radius * self.camera.frame.width
        bottom_right_offset = 0.5  # relative to the radius
        offset_vector = UL * (1 + bottom_right_offset) * radius
        relative_center = self.camera.frame.get_corner(DR) + offset_vector

        circle = Circle(radius=radius, color=WHITE)
        circle.move_to(relative_center)

        angle_vector = radius * np.array([np.cos(theta.get_value()), np.sin(theta.get_value()), 0])
        line = Line(start=circle.get_center(), end=circle.get_center() + angle_vector, color=WHITE)

        return VGroup(circle, line)

    def draw_hyperplane(self: Self, origin: np.ndarray, theta: ValueTracker, bias: ValueTracker, camera_width: float) -> VGroup:
        angle_vector = np.array([np.cos(theta.get_value()), np.sin(theta.get_value()), 0])
        angle_normal_vector = np.array([-np.sin(theta.get_value()), np.cos(theta.get_value()), 0])
        bias_vector = angle_vector * bias.get_value()

        center = origin + bias_vector

        LINE_RADIUS = 10
        ARROW_LENGTH = 0.07
        line = Line(start=center - LINE_RADIUS * angle_normal_vector, end=center + LINE_RADIUS * angle_normal_vector, color=WHITE)
        arrow = Arrow(start=center, end=center + ARROW_LENGTH * camera_width * angle_vector, buff=0, color=WHITE)

        return VGroup(line, arrow)
