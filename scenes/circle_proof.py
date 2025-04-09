from typing import Self

from tqdm import tqdm
import numpy as np
import torch as th
from manim import (
    BLUE,
    BLUE_A,
    BLUE_B,
    DL,
    DR,
    GREEN,
    GREEN_A,
    GREEN_E,
    YELLOW,
    RED,
    RED_A,
    RED_E,
    PI,
    UL,
    UR,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    WHITE,
    IN,
    OUT,
    UP,
    DEGREES,
    Scene,
    FadeIn,
    FadeOut,
    Sphere,
    Line,
    Arc,
    ValueTracker,
    Rectangle,
    VGroup,
    always_redraw,
    ManimColor,
    Text,
    Circle,
    Transform,
)
from noise.hypersphere import hyperspheric_noise

class CircleProof(Scene):
    CIRCLE_RADIUS = 2

    def construct(self: Self) -> None:
        temperature_seed = 0

        # calculate the noise values on a grid
        grid_theta_size = 1000

        grid = th.linspace(0, 2 * np.pi, grid_theta_size + 1)[:-1]
        noise_grid = th.empty_like(grid)

        for i, theta in tqdm(enumerate(grid), desc="Calculating noise values", total=grid_theta_size, leave=False):
            coordinates = th.tensor([theta])

            temperature_noise = hyperspheric_noise(
                coordinates, seed=temperature_seed
            ) * 0.5 + 0.5

            noise_grid[i] = temperature_noise

        # find the most similar noise values on antipodal points
        # first we pair up opposite thetas
        antipodal_grid = noise_grid.reshape(2, -1)

        # now we take the difference
        difference_grid = th.abs(antipodal_grid[0] - antipodal_grid[1])
        minimum_difference_index = th.argmin(difference_grid)
        optimal_theta = grid[minimum_difference_index]

        point_theta = ValueTracker(optimal_theta)
        antipodal_points_opacity = ValueTracker(0)
        antipodal_point_labels_opacity = ValueTracker(0)

        circle = always_redraw(
            lambda: self.draw_antipodal_points_on_circle(point_theta.get_value(), antipodal_points_opacity.get_value(), antipodal_point_labels_opacity.get_value())
        )
        self.play(FadeIn(circle))

        self.wait(4)

        def get_noise(seed: int, antipode: bool = False) -> float:
            if antipode:
                coordinates = th.tensor([point_theta.get_value() + PI])
            else:
                coordinates = th.tensor([point_theta.get_value()])

            return hyperspheric_noise(
                coordinates,
                seed=seed,
            ) * 0.5 + 0.5

        # draw the temperature bars
        temperature_bar = always_redraw(
            lambda: self.draw_bar(
                get_noise(temperature_seed),
                GREEN,
                position=np.array([-4, 2, 0]),
            )
        )
        antipodal_temperature_bar = always_redraw(
            lambda: self.draw_bar(
                get_noise(temperature_seed, antipode=True),
                RED,
                position=np.array([4, 2, 0]),
            )
        )

        temperature_label = always_redraw(
            lambda: Text(
                f"{get_noise(temperature_seed):.2f}",
                color=GREEN,
                font_size=24,
            ).next_to(temperature_bar, DOWN)
        )
        antipodal_temperature_label = always_redraw(
            lambda: Text(
                f"{get_noise(temperature_seed, antipode=True):.2f}",
                color=RED,
                font_size=24,
            ).next_to(antipodal_temperature_bar, DOWN)
        )

        self.play(
            antipodal_points_opacity.animate.set_value(1),
            antipodal_point_labels_opacity.animate.set_value(1),
        )

        self.wait(2)

        self.play(
            FadeIn(temperature_bar),
            FadeIn(antipodal_temperature_bar),
            FadeIn(temperature_label),
            FadeIn(antipodal_temperature_label),
            antipodal_point_labels_opacity.animate.set_value(0),
        )

        self.wait(4)

        self.play(point_theta.animate.set_value(0), run_time=1)

        alpha_label = Text(
            "α",
            color=GREEN,
            font_size=24,
        ).next_to(temperature_bar, DOWN)
        beta_label = Text(
            "β",
            color=RED,
            font_size=24,
        ).next_to(antipodal_temperature_bar, DOWN)

        self.wait(2)

        temperature_label.clear_updaters()
        antipodal_temperature_label.clear_updaters()

        self.play(
            Transform(temperature_label, alpha_label),
            Transform(antipodal_temperature_label, beta_label),
        )

        self.wait(1)

        self.temperatures = []
        graph_start = np.array([-6.5, -2, 0])
        graph_scale = np.array([2, 4, 1])
        graph = always_redraw(
            lambda: self.draw_graph(
                get_noise(temperature_seed),
                get_noise(temperature_seed, antipode=True),
                point_theta.get_value(),
                start=graph_start,
                scale=graph_scale,
            )
        )

        alpha = get_noise(temperature_seed)
        beta = get_noise(temperature_seed, antipode=True)
        start_alpha_location = graph_start + graph_scale * np.array([0, alpha, 0])
        start_beta_location = graph_start + graph_scale * np.array([0, beta, 0])

        self.play(
            alpha_label.animate.next_to(start_alpha_location, LEFT),
            beta_label.animate.next_to(start_beta_location, LEFT),
            FadeIn(graph),
            run_time=1.5,
        )

        self.wait(1)

        theta_arc = always_redraw(
            lambda: Arc(
                radius=self.CIRCLE_RADIUS,
                start_angle=0,
                angle=point_theta.get_value(),
                color=GREEN,
                stroke_width=2,
                stroke_opacity=0.5,
            )
        )
        theta_antipodal_arc = always_redraw(
            lambda: Arc(
                radius=self.CIRCLE_RADIUS,
                start_angle=PI,
                angle=point_theta.get_value() + PI,
                color=RED,
                stroke_width=2,
                stroke_opacity=0.5,
            )
        )

        self.add(theta_arc, theta_antipodal_arc)
        self.play(
            point_theta.animate.set_value(PI),
            run_time=8,
        )

    def draw_graph(
        self: Self,
        alpha: float,
        beta: float,
        theta: float,
        start: np.ndarray = ORIGIN,
        scale: np.ndarray = np.array([1, 1, 1]),
    ) -> VGroup:
        graph = VGroup()

        # draw the x-axis
        x_axis = Line(
            start,
            start + scale * RIGHT,
            color=WHITE,
            stroke_width=2,
            stroke_opacity=0.5,
        )
        graph.add(x_axis)
        # draw the y-axis
        y_axis = Line(
            start,
            start + scale * UP,
            color=WHITE,
            stroke_width=2,
            stroke_opacity=0.5,
        )
        graph.add(y_axis)
        
        # add to the lines, if needed
        last_point = self.temperatures[-1] if len(self.temperatures) > 0 else None
        if last_point is None:
            self.temperatures.append((theta, alpha, beta))
        else:
            last_theta, last_a_temperature, last_b_temperature = last_point
            # only append if the point is different
            if last_theta != theta:
                self.temperatures.append((theta, alpha, beta))
            else:
                self.temperatures[-1] = (theta, alpha, beta)

        a_lines = VGroup()
        b_lines = VGroup()
        # draw the lines
        for (old_theta, old_a_temperature, old_b_temperature), (theta, a_temperature, b_temperature) in zip(self.temperatures[:-1], self.temperatures[1:]):
            # calculate the position of the point
            a_point = np.array([theta, a_temperature, 0])
            b_point = np.array([theta, b_temperature, 0])
            a_point_location = start + scale * a_point
            b_point_location = start + scale * b_point

            old_a_point = np.array([old_theta, old_a_temperature, 0])
            old_b_point = np.array([old_theta, old_b_temperature, 0])
            old_a_point_location = start + scale * old_a_point
            old_b_point_location = start + scale * old_b_point

            # draw the line
            a_line = Line(
                old_a_point_location,
                a_point_location,
                color=GREEN,
                stroke_width=2,
                stroke_opacity=0.5,
            )
            b_line = Line(
                old_b_point_location,
                b_point_location,
                color=RED,
                stroke_width=2,
                stroke_opacity=0.5,
            )
            a_lines.add(a_line)
            b_lines.add(b_line)
        
        graph.add(a_lines)
        graph.add(b_lines)

        return graph

    def draw_antipodal_points_on_circle(
        self: Self,
        theta: float,
        antipodal_points_opacity: float,
        antipodal_point_labels_opacity: float,
    ) -> VGroup:
        circle = Circle(radius=self.CIRCLE_RADIUS, color=BLUE)

        point = Circle(radius=0.1, color=GREEN, fill_color=GREEN)
        point.move_to(
            np.array([self.CIRCLE_RADIUS * np.cos(theta), self.CIRCLE_RADIUS * np.sin(theta), 0])
        )
        point.set_opacity(antipodal_points_opacity)
        point.set_z_index(1)

        antipodal_point = Circle(radius=0.1, color=RED, fill_color=RED)
        antipodal_point.move_to(
            np.array([-self.CIRCLE_RADIUS * np.cos(theta), -self.CIRCLE_RADIUS * np.sin(theta), 0])
        )
        antipodal_point.set_opacity(antipodal_points_opacity)
        antipodal_point.set_z_index(1)

        point_label = Text(
            "A",
            color=GREEN,
            font_size=24,
        ).next_to(point, UP, buff=0.1)
        point_label.set_opacity(antipodal_point_labels_opacity)
        point_label.set_z_index(2)

        antipodal_point_label = Text(
            "B",
            color=RED,
            font_size=24,
        ).next_to(antipodal_point, DOWN, buff=0.1)
        antipodal_point_label.set_opacity(antipodal_point_labels_opacity)
        antipodal_point_label.set_z_index(2)

        return VGroup(
            circle,
            point,
            antipodal_point,
            point_label,
            antipodal_point_label,
        )

    def draw_bar(
        self: Self,
        value: float,
        color: ManimColor,
        position: np.ndarray = ORIGIN,
    ) -> VGroup:
        width = 0.3
        height = 1

        bar = Rectangle(
            width=width,
            height=height * value,
            color=color,
            fill_opacity=0.8,
        )
        bar.move_to(position + UP * height * value / 2)
        bar.set_color(color)
        bar.set_fill(color)

        return bar
