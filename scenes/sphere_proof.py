from os import close
from typing import Self
from itertools import product

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
    ThreeDScene,
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

class SphereProof(ThreeDScene):
    SPHERE_RADIUS = 2
    TEMPERATURE_SEED = 0
    AIR_PRESSURE_SEED = 1

    def construct(self: Self) -> None:
        self.set_camera_orientation(phi=90 * DEGREES, theta=10 * DEGREES)

        point_theta = ValueTracker(0)
        point_phi = ValueTracker(0)

        # calculate the noise values on a grid
        grid_theta_size = 80
        grid_phi_size = 40

        grid_theta = th.linspace(0, 2 * PI, grid_theta_size + 1)[:-1]
        grid_phi = th.linspace(0, PI, grid_phi_size + 1)[:-1]
        mesh_grid_theta, mesh_grid_phi = th.meshgrid(grid_theta, grid_phi)
        grid = th.stack((mesh_grid_phi, mesh_grid_theta), dim=-1)  # (grid_theta_size, grid_phi_size, 2)
        noise_grid = th.empty_like(grid)

        for i, j in tqdm(product(range(grid_theta_size), range(grid_phi_size)), desc="Calculating noise values", total=grid_theta_size * grid_phi_size, leave=False):
            phi, theta = grid[i, j]
            coordinates = th.tensor([phi, theta])

            temperature_noise = hyperspheric_noise(
                coordinates, seed=self.TEMPERATURE_SEED
            ) * 0.5 + 0.5
            air_pressure_noise = hyperspheric_noise(
                coordinates, seed=self.AIR_PRESSURE_SEED
            ) * 0.5 + 0.5

            noise_grid[i, j] = th.tensor([temperature_noise, air_pressure_noise])

        # find the most similar noise values on antipodal points
        # first we pair up opposite thetas
        antipodal_grid = noise_grid.reshape(
            2, -1, grid_phi_size, 2
        )  # 2, grid_theta_size // 2, grid_phi_size, 2
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

        # now we want to find the temperature-optimal phi for each value of theta
        temperature_difference_grid = th.abs(antipodal_grid[0, :, :, 0] - antipodal_grid[1, :, :, 0])  # grid_theta_size // 2, grid_phi_size
        minimum_difference_phi_indices = th.argmin(temperature_difference_grid, dim=1)  # grid_theta_size // 2
        minimum_difference_phis = grid_phi[minimum_difference_phi_indices]  # grid_theta_size // 2
        minimum_difference_phis = th.cat([
            minimum_difference_phis,
            PI - minimum_difference_phis,
        ], dim=0)  # grid_theta_size

        antipodal_points_opacity = ValueTracker(1)
        update_points_with_equivalent_temperatures = ValueTracker(True)
        self.points_with_equivalent_temperatures = {}

        antipodal_points = always_redraw(
            lambda: self.draw_points_on_surface(
                point_theta.get_value(),
                point_phi.get_value(),
                antipodal_points_opacity.get_value(),
                bool(update_points_with_equivalent_temperatures.get_value()),
                minimum_difference_phis=minimum_difference_phis,
                grid_theta=grid_theta,
            )
        )
        self.play(FadeIn(antipodal_points))
        
        self.wait(2)

        self.play(point_phi.animate.set_value(PI), run_time=4)

        self.wait(5)


    def draw_points_on_surface(
        self: Self,
        theta: float | th.Tensor,
        phi: float | th.Tensor,
        antipodal_points_opacity: float,
        update_points_with_equivalent_temperatures: bool,
        minimum_difference_phis: th.Tensor,
        grid_theta: th.Tensor,
    ) -> VGroup:
        """
        Draw a point on the surface of the sphere using spherical coordinates as well as its antipodal point.
        """

        phi, theta = th.tensor(phi), th.tensor(theta)

        x = self.SPHERE_RADIUS * th.sin(phi) * th.cos(theta)
        y = self.SPHERE_RADIUS * th.sin(phi) * th.sin(theta)
        z = self.SPHERE_RADIUS * th.cos(phi)

        earth = Sphere(radius=self.SPHERE_RADIUS)

        point = Sphere(radius=0.1, fill_opacity=1, resolution=(5, 5)).move_to(np.array([x, y, z]))
        antipodal_point = Sphere(radius=0.1, fill_opacity=1, resolution=(5, 5)).move_to(np.array([-x, -y, -z]))

        point.set_color(GREEN)
        antipodal_point.set_color(RED)
        point.set_opacity(antipodal_points_opacity)
        antipodal_point.set_opacity(antipodal_points_opacity)

        # find the closest theta in grid_theta
        closest_theta_index = int(th.argmin(th.abs(grid_theta - theta)).item())
        import pdb; pdb.set_trace()
        optimal_phi_for_this_theta = minimum_difference_phis[closest_theta_index]
        past_optimal_phi = (phi > optimal_phi_for_this_theta).item()

        if update_points_with_equivalent_temperatures:
            if past_optimal_phi and closest_theta_index not in self.points_with_equivalent_temperatures:
                point_marker = Sphere(radius=0.1, fill_opacity=1, resolution=(5, 5)).move_to(np.array([x, y, z]))
                antipodal_point_marker = Sphere(radius=0.1, fill_opacity=1, resolution=(5, 5)).move_to(np.array([-x, -y, -z]))
                point_marker.set_color(YELLOW)
                antipodal_point_marker.set_color(YELLOW)

                self.points_with_equivalent_temperatures[closest_theta_index] = (point_marker, antipodal_point_marker)
            
            all_highlighted_points = self.points_with_equivalent_temperatures.values()
            # flatten
            all_highlighted_points = [point for pair in all_highlighted_points for point in pair]

            return VGroup(earth, point, antipodal_point, *all_highlighted_points)
        
        return VGroup(earth, point, antipodal_point)
