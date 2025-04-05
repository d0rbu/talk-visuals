from typing import Self

import numpy as np
import torch as th
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
    IN,
    OUT,
    DEGREES,
    ThreeDScene,
    FadeIn,
    FadeOut,
    Sphere,
    ValueTracker,
    VGroup,
    always_redraw,
)
from noise.hypersphere import hyperspheric_noise

class Intro(ThreeDScene):
    EARTH_RADIUS = 2

    def construct(self: Self) -> None:
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        self.begin_ambient_camera_rotation()

        earth = Sphere(radius=self.EARTH_RADIUS)
        earth.set_opacity(0.7)

        self.play(FadeIn(earth))

        point_theta = ValueTracker(0)
        point_phi = ValueTracker(0)

        self.wait(2)

        antipodal_points = always_redraw(
            lambda: self.draw_points_on_surface(point_theta, point_phi)
        )
        self.play(FadeIn(antipodal_points))

        self.wait(2)

        self.play(point_theta.animate.set_value(PI), point_phi.animate.set_value(PI / 2), run_time=2)

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

        point.set_color(RED)
        antipodal_point.set_color(RED)

        return VGroup(point, antipodal_point)
