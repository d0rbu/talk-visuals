from typing import Self

import numpy as np
import torch as th
from focus_ireland import FocusIreland
from manim import (
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
)
from preprocessing.cutting import count_positive, bisect_angles


class IVTProof(MovingCameraScene):
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

        self.play(FadeIn(ireland_scene_image))

        theta = ValueTracker(0)
        # draw a circle thats always on the bottom right of the screen
        angle_indicator = always_redraw(lambda: self.draw_angle_circle(theta))
        self.play(FadeIn(angle_indicator))
        self.play(theta.animate.set_value(PI / 8), run_time=0.2)
        self.play(theta.animate.set_value(-PI / 8), run_time=0.4)
        self.play(theta.animate.set_value(0), run_time=0.2)

        self.wait(3)

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

        self.wait(4)

        positive_side = always_redraw(
            lambda: self.draw_positive_side(origin, theta, bias)
        )

        self.play(FadeIn(positive_side), run_time=0.5)

        ireland_pixels = th.tensor(np.asarray(ireland_image)).permute(2, 0, 1)
        solid_pixels = th.nonzero(ireland_pixels[3] > 200)
        solid_pixels[0] = ireland_pixels.shape[1] - solid_pixels[0]

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

        self.wait(1)

        covered_ratio = always_redraw(
            lambda: self.draw_covered_ratio(
                solid_pixels_world_space_xy, theta, bias, bias_of_ireland_center
            )
        )

        self.play(FadeIn(covered_ratio), run_time=0.5)

        self.wait(1.5)

        self.play(bias.animate.set_value(1), run_time=0.75)

        self.wait(5)

        self.play(bias.animate.set_value(-1), run_time=1.5)

        self.wait(2)

        self.play(FadeOut(covered_ratio))

        self.wait(2)

        generate_graph = ValueTracker(True)
        self.graph_points = []
        graph = self.draw_covered_graph(bias, solid_pixels_world_space_xy, theta, bias_of_ireland_center, generate_graph)
        self.play(FadeIn(graph), run_time=0.5)
        self.play(bias.animate.set_value(1), rate_func=rate_functions.linear, run_time=4)

        generate_graph.set_value(False)

        self.wait(2)

        ireland_bisection_bias = bisect_angles(th.tensor(solid_pixels_world_space_xy), th.tensor([theta.get_value()]))

        self.play(bias.animate.set_value(ireland_bisection_bias - bias_of_ireland_center), run_time=1.5)

        start_x = self.graph_points[0][0]
        end_x = self.graph_points[-1][0]
        midpoint_y = (self.graph_points[0][1] + self.graph_points[-1][1]) / 2
        halfway_line = Line(start=np.array([start_x, midpoint_y, 0]), end=np.array([end_x, midpoint_y, 0]), color=WHITE)
        halfway_line.set_z_index(2)
        halfway_line_label = Text("0.5", font_size=12, color=WHITE)

        halfway_line_label.next_to(halfway_line, RIGHT, buff=0.1)

        self.play(FadeIn(halfway_line), FadeIn(halfway_line_label), run_time=0.5)

        self.wait(6)

        self.play(FadeOut(halfway_line), FadeOut(halfway_line_label), FadeOut(graph))

        # TODO: rotate hyperplane around, bisecting the points at each angle

    def draw_covered_graph(self: Self, bias: ValueTracker, points: th.Tensor, theta: ValueTracker, bias_of_ireland_center: float, generate_graph: ValueTracker) -> VGroup:
        last_point = None

        graph_size = np.array([1.0, 1.0, 0])
        left_offset = 0.7  # relative to the width
        graph_center = self.camera.frame.get_corner(LEFT) + RIGHT * graph_size[0] / 2 + RIGHT * left_offset * graph_size[0] + DOWN * graph_size[1] / 2  # center at 0.5

        def update_graph():
            nonlocal last_point
            x = bias.get_value()
            y = count_positive(points, theta.get_value(), x + bias_of_ireland_center) / points.shape[0]
            current_point_world_space = np.array([x, y, 0])

            current_point = current_point_world_space * graph_size + graph_center

            if generate_graph.get_value() and (last_point is None or not np.isclose(last_point[0], current_point[0])):
                self.graph_points.append(current_point)

            last_point = current_point

            if len(self.graph_points) < 2:
                point_indicator = Circle(radius=0.05, color=RED, fill_opacity=1)
                point_indicator.move_to(current_point)
                point_indicator.set_z_index(3)

                return VGroup(point_indicator)

            first_x = self.graph_points[0][0]
            last_x = self.graph_points[-1][0]
            polygon_points = self.graph_points + [np.array([last_x, graph_center[1], 0]), np.array([first_x, graph_center[1], 0])]
            
            polygon = Polygon(*polygon_points, color=GREEN, fill_opacity=0.5)
            polygon.set_z_index(1)

            point_indicator = Circle(radius=0.05, color=RED, fill_opacity=1)
            point_indicator.move_to(current_point)
            point_indicator.set_z_index(3)

            return VGroup(polygon, point_indicator)

        return always_redraw(update_graph)


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

    def draw_angle_circle(self: Self, theta: ValueTracker) -> VGroup:
        relative_radius = 0.05
        radius = relative_radius * self.camera.frame.width
        bottom_right_offset = 0.5  # relative to the radius
        offset_vector = UL * (1 + bottom_right_offset) * radius
        relative_center = self.camera.frame.get_corner(DR) + offset_vector

        circle = Circle(radius=radius, color=WHITE)
        circle.move_to(relative_center)

        angle_vector = radius * np.array(
            [np.cos(theta.get_value()), np.sin(theta.get_value()), 0]
        )
        line = Line(
            start=circle.get_center(),
            end=circle.get_center() + angle_vector,
            color=WHITE,
        )

        angle_circle = VGroup(circle, line)
        angle_circle.set_z_index(2)

        return angle_circle

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
            buff=0,
            color=WHITE,
        )

        hyperplane = VGroup(line, arrow)
        hyperplane.set_z_index(2)

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
