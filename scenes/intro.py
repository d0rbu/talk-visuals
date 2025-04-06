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
    ThreeDScene,
    FadeIn,
    FadeOut,
    Sphere,
    ValueTracker,
    Rectangle,
    VGroup,
    always_redraw,
    ManimColor,
    Text,
)
from noise.hypersphere import hyperspheric_noise

class Intro(ThreeDScene):
    EARTH_RADIUS = 2

    def construct(self: Self) -> None:
        self.set_camera_orientation(phi=90 * DEGREES, theta=0 * DEGREES)

        point_theta = ValueTracker(0)
        point_phi = ValueTracker(0)

        antipodal_points = always_redraw(
            lambda: self.draw_points_on_surface(point_theta, point_phi)
        )
        self.play(FadeIn(antipodal_points))

        self.wait(4)

        # constantly updated computed value based on the noise
        temperature_seed = 0
        air_pressure_seed = 4

        def get_noise(seed: int, antipode: bool = False) -> float:
            if antipode:
                coordinates = th.tensor([point_theta.get_value() + PI, PI - point_phi.get_value()])
            else:
                coordinates = th.tensor([point_theta.get_value(), point_phi.get_value()])

            return hyperspheric_noise(
                coordinates,
                seed=seed,
            ) * 0.5 + 0.5

        # draw the temperature/air pressure bars
        temperature_bar = always_redraw(
            lambda: self.draw_bar(
                get_noise(temperature_seed),
                GREEN_A,
                position=np.array([-4, 2, 0]),
            )
        )
        air_pressure_bar = always_redraw(
            lambda: self.draw_bar(
                get_noise(air_pressure_seed),
                GREEN_E,
                position=np.array([3, 2, 0]),
            )
        )
        antipodal_temperature_bar = always_redraw(
            lambda: self.draw_bar(
                get_noise(temperature_seed, antipode=True),
                RED_A,
                position=np.array([-3, 2, 0]),
            )
        )
        antipodal_air_pressure_bar = always_redraw(
            lambda: self.draw_bar(
                get_noise(air_pressure_seed, antipode=True),
                RED_E,
                position=np.array([4, 2, 0]),
            )
        )

        temperature_bars = VGroup(temperature_bar, antipodal_temperature_bar)
        air_pressure_bars = VGroup(air_pressure_bar, antipodal_air_pressure_bar)

        temperature_label = Text(
            "temperatures",
            color=BLUE_A,
            font_size=24,
        ).next_to(temperature_bars, DOWN, buff=0.5)
        air_pressure_label = Text(
            "air pressures",
            color=BLUE_B,
            font_size=24,
        ).next_to(air_pressure_bars, DOWN, buff=0.5)

        self.add_fixed_in_frame_mobjects(temperature_bars, air_pressure_bars, temperature_label, air_pressure_label)

        self.play(
            FadeIn(temperature_bars),
            FadeIn(air_pressure_bars),
            FadeIn(temperature_label),
            FadeIn(air_pressure_label),
        )

        # calculate the noise values on a grid
        grid_theta_size = 200
        grid_phi_size = 100

        grid_theta = th.linspace(0, 2 * np.pi, grid_theta_size + 1)[:-1]
        grid_phi = th.linspace(0, np.pi, grid_phi_size + 1)[:-1]
        grid_theta, grid_phi = th.meshgrid(grid_theta, grid_phi)
        grid = th.stack((grid_theta, grid_phi), dim=-1)  # (grid_theta_size, grid_phi_size, 2)
        noise_grid = th.empty_like(grid)

        for i, j in tqdm(product(range(grid_theta_size), range(grid_phi_size)), desc="Calculating noise values", total=grid_theta_size * grid_phi_size, leave=False):
            theta, phi = grid[i, j]
            coordinates = th.tensor([theta, phi])

            temperature_noise = hyperspheric_noise(
                coordinates, seed=temperature_seed
            ) * 0.5 + 0.5
            air_pressure_noise = hyperspheric_noise(
                coordinates, seed=air_pressure_seed
            ) * 0.5 + 0.5

            noise_grid[i, j] = th.tensor([temperature_noise, air_pressure_noise])

        # find the most similar noise values on antipodal points
        # first we pair up opposite thetas
        antipodal_grid = noise_grid.reshape(
            2, -1, grid_phi_size, 2
        )
        # now we invert the phis
        antipodal_grid[1] = th.flip(antipodal_grid[1], [1])

        # now we take the difference
        difference_grid = th.abs(antipodal_grid[0] - antipodal_grid[1]).sum(dim=-1)
        minimum_difference_indices_flattened = th.argmin(difference_grid)
        minimum_difference_indices = th.unravel_index(
            minimum_difference_indices_flattened, difference_grid.shape
        )
        minimum_difference_location = grid[minimum_difference_indices]

        optimal_theta, optimal_phi = minimum_difference_location

        self.play(point_theta.animate.set_value(optimal_theta), point_phi.animate.set_value(optimal_phi), run_time=5)

        self.wait(5)

    def draw_points_on_surface(self: Self, point_theta: ValueTracker, point_phi: ValueTracker) -> VGroup:
        """
        Draw a point on the surface of the sphere using spherical coordinates as well as its antipodal point.
        """
        theta = th.tensor(point_theta.get_value())
        phi = th.tensor(point_phi.get_value())

        x = self.EARTH_RADIUS * th.sin(phi) * th.cos(theta)
        y = self.EARTH_RADIUS * th.sin(phi) * th.sin(theta)
        z = self.EARTH_RADIUS * th.cos(phi)

        point = Sphere(radius=0.1, fill_opacity=1, resolution=(5, 5)).move_to([x, y, z])
        antipodal_point = Sphere(radius=0.1, fill_opacity=1, resolution=(5, 5)).move_to([-x, -y, -z])

        point.set_color(GREEN)
        antipodal_point.set_color(RED)

        earth = Sphere(radius=self.EARTH_RADIUS)

        return VGroup(earth, point, antipodal_point)

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
