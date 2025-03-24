from typing import Self

import numpy as np
import torch as th
from manim import GREEN, ImageMobject, MovingCameraScene, Circle, ValueTracker, always_redraw, FadeIn, VGroup, WHITE, DR, UL, Line, PI, Arrow, Polygon
from focus_ireland import FocusIreland
from preprocessing.cutting import bisect_angles, count_positive


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
        hyperplane = always_redraw(lambda: self.draw_hyperplane(origin, theta, bias))

        self.play(FadeIn(hyperplane))
        self.play(theta.animate.set_value(PI / 24), run_time=0.2)
        self.play(theta.animate.set_value(-PI / 24), run_time=0.4)
        self.play(theta.animate.set_value(PI / 24), run_time=0.4)
        self.play(theta.animate.set_value(-PI / 24), run_time=0.4)
        self.play(theta.animate.set_value(PI / 24), run_time=0.4)
        self.play(theta.animate.set_value(0), run_time=0.2)

        self.wait(1)
        
        positive_side = always_redraw(lambda: self.draw_positive_side(origin, theta, bias))

        self.play(FadeIn(positive_side), run_time=0.5)

        ireland_pixels = th.tensor(np.asarray(ireland_image)).permute(2, 0, 1)
        solid_pixels = th.nonzero(ireland_pixels[3] > 200)
        solid_pixels[0] = ireland_pixels.shape[1] - solid_pixels[0]

        solid_pixels_normalized = solid_pixels / th.tensor(ireland_pixels.shape[1:]).float()
        solid_pixels_normalized -= 0.5

        world_space_shape = th.tensor([ireland_scene_image.get_height(), ireland_scene_image.get_width()])
        solid_pixels_world_space = solid_pixels_normalized * world_space_shape + th.tensor(ireland_scene_image.get_center()[-2:-4:-1].copy())

        bias_of_ireland_center = ireland_center[:2] @ np.array([np.cos(theta.get_value()), np.sin(theta.get_value())])
        num_solid_pixels = solid_pixels_world_space.shape[0]

        # yx -> xy
        solid_pixels_world_space_xy = th.roll(solid_pixels_world_space, 1, 1)
        num_positive_pixels = count_positive(solid_pixels_world_space_xy, theta.get_value(), bias.get_value() + bias_of_ireland_center)

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

    LINE_RADIUS = 10
    ARROW_LENGTH = 0.5

    def draw_hyperplane(self: Self, origin: np.ndarray, theta: ValueTracker, bias: ValueTracker) -> VGroup:
        angle_vector = np.array([np.cos(theta.get_value()), np.sin(theta.get_value()), 0])
        angle_normal_vector = np.array([-np.sin(theta.get_value()), np.cos(theta.get_value()), 0])
        bias_vector = angle_vector * bias.get_value()

        center = origin + bias_vector
        line = Line(start=center - self.LINE_RADIUS * angle_normal_vector, end=center + self.LINE_RADIUS * angle_normal_vector, color=WHITE)
        arrow = Arrow(start=center, end=center + self.ARROW_LENGTH * angle_vector, buff=0, color=WHITE)

        hyperplane = VGroup(line, arrow)
        hyperplane.set_z_index(2)

        return VGroup(line, arrow)

    def draw_positive_side(self: Self, origin: np.ndarray, theta: ValueTracker, bias: ValueTracker) -> Polygon:
        angle_vector = np.array([np.cos(theta.get_value()), np.sin(theta.get_value()), 0])
        angle_normal_vector = np.array([-np.sin(theta.get_value()), np.cos(theta.get_value()), 0])
        bias_vector = angle_vector * bias.get_value()

        center = origin + bias_vector
        right = center + self.LINE_RADIUS * angle_normal_vector
        left = center - self.LINE_RADIUS * angle_normal_vector
        top = right + self.LINE_RADIUS * angle_vector
        bottom = left + self.LINE_RADIUS * angle_vector

        positive_side = Polygon(right, left, bottom, top, fill_opacity=0.3, color=GREEN)
        positive_side.set_z_index(1)

        return positive_side
