from typing import Self

import numpy as np
import torch as th
from focus_ireland import FocusIreland
from manim import (
    BLUE,
    DL,
    DR,
    GREEN,
    RED,
    PI,
    UL,
    UR,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    WHITE,
    Text,
    Arrow,
    Circle,
    FadeIn,
    FadeOut,
    ImageMobject,
    Line,
    MovingCameraScene,
    Polygon,
    Rectangle,
    ValueTracker,
    VGroup,
    always_redraw,
    rate_functions,
    ManimColor,
)
from preprocessing.cutting import count_positive, bisect_angles


class HamSandwichProof(MovingCameraScene):
    def construct(self: Self) -> None:
        uk_image, ireland_image = FocusIreland.get_uk_and_ireland_images()

        start_height = 6

        uk_scene_image = ImageMobject(uk_image)
        ireland_scene_image = ImageMobject(ireland_image)
        uk_scene_image.set_height(start_height)
        ireland_scene_image.set_height(start_height)

        # zoom in on ireland
        ireland_bbox = FocusIreland.get_world_space_bbox(
            ireland_image, ireland_scene_image
        )
        ireland_center = np.array(
            [
                (ireland_bbox[0] + ireland_bbox[1]) / 2,
                (ireland_bbox[2] + ireland_bbox[3]) / 2,
                0,
            ]
        )
        self.camera.frame.scale(0.5).move_to(ireland_center)

        theta = ValueTracker(0)
        bias = ValueTracker(0)
        origin = self.camera.frame.get_center()
        hyperplane = always_redraw(lambda: self.draw_hyperplane(origin, theta, bias))
        angle_indicator = always_redraw(lambda: self.draw_angle_circle(theta))

        ireland_pixels = th.tensor(np.asarray(ireland_image)).permute(2, 0, 1)  # C, H, W
        solid_pixels = th.nonzero(ireland_pixels[3] > 200)  # N, 2
        solid_pixels[:, 0] = ireland_pixels.shape[1] - solid_pixels[:, 0]

        solid_pixels_normalized = (
            solid_pixels / th.tensor(ireland_pixels.shape[1:]).float()
        )
        solid_pixels_normalized -= 0.5

        world_space_shape = th.tensor(
            [ireland_scene_image.get_height(), ireland_scene_image.get_width()]
        )
        solid_pixels_world_space = (
            solid_pixels_normalized * world_space_shape
            + th.tensor(ireland_scene_image.get_center()[-2:-4:-1].copy())
        )

        bias_of_ireland_center = ireland_center[:2] @ np.array(
            [np.cos(theta.get_value()), np.sin(theta.get_value())]
        )

        # yx -> xy
        solid_pixels_world_space_xy = th.roll(solid_pixels_world_space, 1, 1)

        covered_ratio = always_redraw(
            lambda: self.draw_covered_ratio(
                solid_pixels_world_space_xy, theta, bias, bias_of_ireland_center
            )
        )

        self.play(FadeIn(angle_indicator), FadeIn(ireland_scene_image))

        self.wait(2)

        self.play(self.camera.frame.animate.move_to(ORIGIN).scale(2), FadeIn(uk_scene_image))

        self.wait(2)

    def draw_angle_circle(self: Self, theta: ValueTracker) -> VGroup:
        relative_radius = 0.05
        radius = relative_radius * self.camera.frame.width
        bottom_right_offset = 0.5  # relative to the radius
        offset_vector = UL * (1 + bottom_right_offset) * radius
        relative_center = self.camera.frame.get_corner(DR) + offset_vector

        relative_stroke_width = 0.5
        stroke_width = self.camera.frame.width * relative_stroke_width

        circle = Circle(radius=radius, color=WHITE, stroke_width=stroke_width)
        circle.move_to(relative_center)

        angle_vector = radius * np.array(
            [np.cos(theta.get_value()), np.sin(theta.get_value()), 0]
        )
        line = Line(
            start=circle.get_center(),
            end=circle.get_center() + angle_vector,
            color=WHITE,
            stroke_width=stroke_width
        )

        angle_circle = VGroup(circle, line)
        angle_circle.set_z_index(2)

        return angle_circle

    def draw_covered_ratio(
        self: Self,
        points: th.Tensor,
        theta: ValueTracker,
        bias: ValueTracker,
        bias_of_ireland_center: float,
    ) -> VGroup:
        num_positive_pixels = count_positive(
            points, theta.get_value(), bias.get_value() + bias_of_ireland_center
        )
        num_solid_pixels = points.shape[0]

        ratio = num_positive_pixels / num_solid_pixels

        relative_width = 0.05
        relative_height = 0.2
        width = self.camera.frame.width * relative_width
        height = self.camera.frame.height * relative_height

        outline_box = Rectangle(width=width, height=height, color=WHITE)
        ratio_box = Rectangle(
            width=width, height=height * ratio, color=GREEN, fill_opacity=0.3
        )

        graphic = VGroup(outline_box, ratio_box)
        graphic.arrange(ORIGIN, aligned_edge=DOWN)

        # move to bottom left of camera
        bottom_left_offset = 0.5  # relative to the width
        bottom_left_offset_vector = UR * bottom_left_offset * width
        offset_vector = np.array([width, height, 0]) / 2 + bottom_left_offset_vector
        relative_center = self.camera.frame.get_corner(DL) + offset_vector

        graphic.move_to(relative_center)

        return VGroup(outline_box, ratio_box)

    LINE_RADIUS = 10
    ARROW_LENGTH = 0.5

    def draw_hyperplane(
        self: Self, origin: np.ndarray, theta: ValueTracker, bias: ValueTracker
    ) -> VGroup:
        angle_vector = np.array(
            [np.cos(theta.get_value()), np.sin(theta.get_value()), 0]
        )
        angle_normal_vector = np.array(
            [-np.sin(theta.get_value()), np.cos(theta.get_value()), 0]
        )
        bias_vector = angle_vector * bias.get_value()

        center = origin + bias_vector
        line = Line(
            start=center - self.LINE_RADIUS * angle_normal_vector,
            end=center + self.LINE_RADIUS * angle_normal_vector,
            color=WHITE,
        )
        arrow = Arrow(
            start=center,
            end=center + self.ARROW_LENGTH * angle_vector,
            buff=0.02,
            color=WHITE,
        )

        hyperplane = VGroup(line, arrow)
        hyperplane.set_z_index(2)
        hyperplane.set_opacity(0.5)

        return VGroup(line, arrow)

    def draw_positive_side(
        self: Self, origin: np.ndarray, theta: ValueTracker, bias: ValueTracker
    ) -> Polygon:
        angle_vector = np.array(
            [np.cos(theta.get_value()), np.sin(theta.get_value()), 0]
        )
        angle_normal_vector = np.array(
            [-np.sin(theta.get_value()), np.cos(theta.get_value()), 0]
        )
        bias_vector = angle_vector * bias.get_value()

        center = origin + bias_vector
        right = center + self.LINE_RADIUS * angle_normal_vector
        left = center - self.LINE_RADIUS * angle_normal_vector
        top = right + self.LINE_RADIUS * angle_vector
        bottom = left + self.LINE_RADIUS * angle_vector

        positive_side = Polygon(right, left, bottom, top, fill_opacity=0.3, color=GREEN)
        positive_side.set_z_index(1)

        return positive_side
