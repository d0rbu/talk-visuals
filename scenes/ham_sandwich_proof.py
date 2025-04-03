from typing import Self

import numpy as np
import torch as th
from focus_ireland import FocusIreland
from mobjects.imagemobject import ImageMobject
from manim import (
    BLUE,
    DL,
    DR,
    GREEN,
    YELLOW,
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
        ireland_center_tensor = th.tensor(ireland_center[:2])

        # yx -> xy
        solid_pixels_world_space_xy = th.roll(solid_pixels_world_space, 1, 1)

        theta = ValueTracker(0)
        bias = ValueTracker(bisect_angles(solid_pixels_world_space_xy - ireland_center_tensor, th.tensor([theta.get_value()])))

        origin = self.camera.frame.get_center()
        hyperplane = always_redraw(lambda: self.draw_hyperplane(origin, theta, bias))
        angle_indicator = always_redraw(lambda: self.draw_angle_circle(theta))

        self.play(bias.animate.set_value(0), run_time=0.01)

        bias.add_updater(lambda m: self.update_bias_to_bisect(m, solid_pixels_world_space_xy, theta, ireland_center_tensor))

        self.play(FadeIn(angle_indicator), FadeIn(ireland_scene_image))

        self.wait(0.5)

        self.play(self.camera.frame.animate.move_to(ORIGIN).scale(2), FadeIn(uk_scene_image), ireland_scene_image.animate.set_opacity(0.4))

        self.wait(2)

        uk_pixels = th.tensor(np.asarray(uk_image)).permute(2, 0, 1)  # C, H, W
        solid_pixels = th.nonzero(uk_pixels[3] > 200)  # N, 2
        solid_pixels[:, 0] = uk_pixels.shape[1] - solid_pixels[:, 0]
        solid_pixels_normalized = (
            solid_pixels / th.tensor(uk_pixels.shape[1:]).float()
        )
        solid_pixels_normalized -= 0.5
        world_space_shape = th.tensor(
            [uk_scene_image.get_height(), uk_scene_image.get_width()]
        )
        solid_pixels_world_space = (
            solid_pixels_normalized * world_space_shape
            + th.tensor(uk_scene_image.get_center()[-2:-4:-1].copy())
        )
        uk_solid_pixels_world_space_xy = th.roll(solid_pixels_world_space, 1, 1)

        covered_ratio = always_redraw(
            lambda: self.draw_covered_ratio(
                uk_solid_pixels_world_space_xy, theta, bias, ireland_center_tensor
            )
        )
        positive_side = always_redraw(
            lambda: self.draw_positive_side(
                origin, theta, bias
            )
        )

        self.play(FadeIn(hyperplane), FadeIn(covered_ratio), FadeIn(positive_side))

        self.wait(4)

        self.play(theta.animate.set_value(2 * PI), run_time=8)

        self.wait(2)

        self.circular_graph_radius = 3
        big_circle = Circle(radius=self.circular_graph_radius, color=WHITE)

        self.play(FadeOut(angle_indicator), FadeOut(positive_side), FadeOut(hyperplane), FadeOut(ireland_scene_image), FadeOut(uk_scene_image), FadeIn(big_circle))

        generate_graph = ValueTracker(True)
        self.graph_lines = []
        self.angle_radii = {}
        circular_graph = self.draw_circular_graph(bias, uk_solid_pixels_world_space_xy, theta, ireland_center_tensor, generate_graph)
        line_indicator = always_redraw(lambda: self.draw_line_indicator(theta))
        self.play(FadeIn(circular_graph), FadeIn(line_indicator))
        self.play(theta.animate.set_value(4 * PI), run_time=4)

        generate_graph.set_value(False)

        # get values of theta where the radius is closest to self.circular_graph_radius
        angles_sorted = sorted(self.angle_radii.keys(), key=lambda x: abs(self.angle_radii[x] - self.circular_graph_radius))
        closest_angle = angles_sorted[0]
        antipode = closest_angle - PI

        closest_angle_dot = Circle(radius=0.06, color=YELLOW, fill_opacity=1)
        closest_angle_dot.move_to(
            self.circular_graph_radius * np.array(
                [np.cos(closest_angle), np.sin(closest_angle), 0]
            )
        )
        antipode_dot = Circle(radius=0.06, color=YELLOW, fill_opacity=1)
        antipode_dot.move_to(
            self.circular_graph_radius * np.array(
                [np.cos(antipode), np.sin(antipode), 0]
            )
        )

        self.play(FadeIn(closest_angle_dot), FadeIn(antipode_dot))

        self.wait(7)

        self.play(theta.animate.set_value(closest_angle), run_time=1)

        self.wait(2)

        hyperplane = always_redraw(lambda: self.draw_hyperplane(origin, theta, bias))
        angle_indicator = always_redraw(lambda: self.draw_angle_circle(theta))
        positive_side = always_redraw(
            lambda: self.draw_positive_side(
                origin, theta, bias
            )
        )

        self.play(FadeIn(angle_indicator), FadeIn(positive_side), FadeIn(hyperplane), FadeIn(ireland_scene_image), FadeIn(uk_scene_image), FadeOut(big_circle), FadeOut(circular_graph), FadeOut(line_indicator), FadeOut(closest_angle_dot), FadeOut(antipode_dot))

        self.play(ireland_scene_image.animate.set_opacity(2.0))

        self.wait(3)

        self.play(theta.animate.set_value(antipode), run_time=0.4)

        self.wait(15)
    
    def draw_line_indicator(self: Self, theta: ValueTracker) -> Line:
        return Line(
            start=ORIGIN,
            end=self.circular_graph_radius * np.array(
                [np.cos(theta.get_value()), np.sin(theta.get_value()), 0]
            ),
            color=WHITE
        )

    def draw_circular_graph(self: Self, bias: ValueTracker, points: th.Tensor, theta: ValueTracker, origin: th.Tensor, generate_graph: ValueTracker) -> VGroup:
        last_point = None

        def update_graph():
            nonlocal last_point
            if not generate_graph.get_value():
                return VGroup(self.graph_lines)

            angle = theta.get_value()
            num_positive_pixels = count_positive(
                points - origin, theta.get_value(), bias.get_value()
            )
            num_solid_pixels = points.shape[0]

            ratio = num_positive_pixels / num_solid_pixels
            radius = ratio - 0.5 + self.circular_graph_radius
            new_point = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0
            ])

            self.angle_radii[angle] = radius

            if last_point is not None:
                if radius > self.circular_graph_radius:
                    color = GREEN
                else:
                    color = RED

                self.graph_lines.append(Line(last_point, new_point, color=color))

            last_point = new_point

            return VGroup(self.graph_lines)

        return always_redraw(update_graph)

    def update_bias_to_bisect(self: Self, bias: ValueTracker, points: th.Tensor, theta: ValueTracker, origin: th.Tensor) -> None:
        # Update bias using bisect_angles based on the current angle (theta) and solid pixels
        transformed_points = points - origin
        new_bias = bisect_angles(transformed_points, th.tensor([theta.get_value()]))
        bias.set_value(new_bias)

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
        origin: th.Tensor,
    ) -> VGroup:
        num_positive_pixels = count_positive(
            points - origin, theta.get_value(), bias.get_value()
        )
        num_solid_pixels = points.shape[0]

        ratio = num_positive_pixels / num_solid_pixels

        relative_width = 0.05
        relative_height = 0.2
        width = self.camera.frame.width * relative_width
        height = self.camera.frame.height * relative_height
        
        relative_stroke_width = 0.5
        stroke_width = self.camera.frame.width * relative_stroke_width

        outline_box = Rectangle(width=width, height=height, color=WHITE, stroke_width=stroke_width)
        ratio_box = Rectangle(
            width=width, height=height * ratio, color=GREEN, fill_opacity=0.3, stroke_width=stroke_width
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
