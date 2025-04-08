from typing import Self
from itertools import product

from tqdm import tqdm
import numpy as np
import torch as th
from mobjects.imagemobject import ImageMobject
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

        temperature_label = Text(
            f"{get_noise(temperature_seed):.2f}",
            color=GREEN,
            font_size=24,
        ).next_to(temperature_bar, DOWN)
        antipodal_temperature_label = Text(
            f"{get_noise(temperature_seed, antipode=True):.2f}",
            color=RED,
            font_size=24,
        ).next_to(antipodal_temperature_bar, DOWN)

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

        self.wait(3)

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

        self.play(
            Transform(temperature_label, alpha_label),
            Transform(antipodal_temperature_label, beta_label),
        )

        self.wait(2)

        self.play(point_theta.animate.set_value(0), run_time=1)

        self.wait(4)

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
