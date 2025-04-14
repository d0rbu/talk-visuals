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


    def construct(self: Self) -> None:
        self.set_camera_orientation(phi=90 * DEGREES, theta=10 * DEGREES)

        point_theta = ValueTracker(0)
        point_phi = ValueTracker(0)

        theta_resolution = 80

        # Generate realistic-looking continuous noise (just a random walk lol)
        min_diff_phis = th.cumsum(th.normal(mean=0.0, std=0.05, size=(theta_resolution // 2,), generator=th.Generator().manual_seed(42)), dim=0)
        initial_min_diff_phi = min_diff_phis[0].clone()
        final_min_diff_phi = min_diff_phis[-1].clone()
        min_diff_phis += th.linspace(final_min_diff_phi, initial_min_diff_phi, theta_resolution // 2)
        min_diff_phis -= initial_min_diff_phi
        min_diff_phis -= final_min_diff_phi
        min_diff_phis += PI / 2
        min_diff_phis = th.cat([min_diff_phis, PI - min_diff_phis])  # enforce symmetry

        grid_theta = th.linspace(0, 2 * PI, theta_resolution + 1)[:-1]

        antipodal_points_opacity = ValueTracker(1)
        update_points_with_equivalent_temperatures = ValueTracker(True)
        self.points_with_equivalent_temperatures = {}

        antipodal_points = always_redraw(
            lambda: self.draw_points_on_surface(
                point_theta.get_value(),
                point_phi.get_value(),
                antipodal_points_opacity.get_value(),
                bool(update_points_with_equivalent_temperatures.get_value()),
                minimum_difference_phis=min_diff_phis,
                grid_theta=grid_theta,
            )
        )
        self.play(FadeIn(antipodal_points))

        self.wait(4)

        self.play(point_phi.animate.set_value(PI), run_time=4)

        self.wait(8)

        for i, theta in enumerate(grid_theta[1:grid_theta.shape[0] // 2]):
            point_theta.set_value(theta)
            point_phi.set_value(0)

            duration = 3 / (i + 1)

            self.play(
                point_phi.animate.set_value(PI),
                run_time=duration,
            )

        self.wait(2)

        # Draw and fade in the looped lines while fading out the antipodal points
        lines = VGroup()
        sorted_theta_indices = sorted(self.points_with_equivalent_temperatures.keys())
        n = len(sorted_theta_indices)

        for i in range(n):
            theta_i = sorted_theta_indices[i]

            theta_next = sorted_theta_indices[(i + 1) % n]
            point_i = self.points_with_equivalent_temperatures[theta_i][0]

            if i == n - 1:
                point_next = self.points_with_equivalent_temperatures[theta_next][1]
            else:
                point_next = self.points_with_equivalent_temperatures[theta_next][0]
            
            lines.add(Line(point_i.get_center(), point_next.get_center(), color=YELLOW))

        for i in range(n):
            theta_i = sorted_theta_indices[i]
            theta_next = sorted_theta_indices[(i + 1) % n]
            point_i = self.points_with_equivalent_temperatures[theta_i][1]
            
            if i == n - 1:
                point_next = self.points_with_equivalent_temperatures[theta_next][0]
            else:
                point_next = self.points_with_equivalent_temperatures[theta_next][1]
            
            lines.add(Line(point_i.get_center(), point_next.get_center(), color=YELLOW))

        self.play(FadeIn(lines))

        self.wait(6)

        self.move_camera(phi=0 * DEGREES, theta=10 * DEGREES, run_time=3)
        self.play(FadeOut(antipodal_points))
        
        self.wait(8)

    def draw_points_on_surface(
        self: Self,
        theta: float | th.Tensor,
        phi: float | th.Tensor,
        antipodal_points_opacity: float,
        update_points_with_equivalent_temperatures: bool,
        minimum_difference_phis: th.Tensor,
        grid_theta: th.Tensor,
    ) -> VGroup:

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

        closest_theta_index = int(th.argmin(th.abs(grid_theta - theta)).item())
        optimal_phi_for_this_theta = minimum_difference_phis[closest_theta_index]
        past_optimal_phi = (phi > optimal_phi_for_this_theta).item()

        if update_points_with_equivalent_temperatures:
            if past_optimal_phi and closest_theta_index not in self.points_with_equivalent_temperatures:
                optimal_x = self.SPHERE_RADIUS * th.sin(optimal_phi_for_this_theta) * th.cos(grid_theta[closest_theta_index])
                optimal_y = self.SPHERE_RADIUS * th.sin(optimal_phi_for_this_theta) * th.sin(grid_theta[closest_theta_index])
                optimal_z = self.SPHERE_RADIUS * th.cos(optimal_phi_for_this_theta)

                point_marker = Sphere(radius=0.1, fill_opacity=1, resolution=(5, 5)).move_to(np.array([optimal_x, optimal_y, optimal_z]))
                antipodal_point_marker = Sphere(radius=0.1, fill_opacity=1, resolution=(5, 5)).move_to(np.array([-optimal_x, -optimal_y, -optimal_z]))
                point_marker.set_color(YELLOW)
                antipodal_point_marker.set_color(YELLOW)

                self.points_with_equivalent_temperatures[closest_theta_index] = (point_marker, antipodal_point_marker)

            all_highlighted_points = self.points_with_equivalent_temperatures.values()
            all_highlighted_points = [point for pair in all_highlighted_points for point in pair]

            return VGroup(earth, point, antipodal_point, *all_highlighted_points)

        return VGroup(earth, point, antipodal_point)
