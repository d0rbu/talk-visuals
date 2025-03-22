from typing import Self

from manim import (
    BLUE,
    DOWN,
    GREEN,
    PI,
    RED,
    UP,
    Arrow,
    Rotate,
    Circle,
    VGroup,
    Create,
    FadeOut,
    FadeIn,
    Line,
    NumberPlane,
    Polygon,
    ThreeDScene,
    always_rotate,
)


class OrientedHyperplane(ThreeDScene):
    def construct(self: Self) -> None:
        self.wait(6)

        number_plane = NumberPlane(
            x_range=(-24, 24),
            y_range=(-7, 7),
            background_line_style={
                "stroke_color": BLUE,
                "stroke_width": 3,
                "stroke_opacity": 0.5,
            },
        ).rotate(-5 * PI / 16, axis=UP)

        always_rotate(number_plane, rate=PI / 16, axis=UP)

        self.play(Create(number_plane), run_time=1)
        self.wait(4)

        # stop rotating
        number_plane.clear_updaters()

        self.wait()

        line = Line(start=(-10, 0, 0), end=(10, 0, 0), color=GREEN).rotate(
            angle=-PI / 4
        )
        self.play(Create(line))

        self.wait(1)

        self.play(line.animate.shift((0.5, 0, 0)), run_time=0.3)
        self.play(line.animate.shift((-1, 0, 0)), run_time=0.5)
        self.play(line.animate.shift((0.5, 0, 0)), run_time=0.3)
        self.play(line.animate.rotate(angle=PI / 4), FadeOut(number_plane), run_time=1)

        self.wait(1)

        point = Circle(radius=0.1, color=RED, fill_opacity=1)
        self.play(Create(point, run_time=0.5))

        self.wait(1)

        self.play(point.animate.shift((0.5, 0, 0)), run_time=0.3)
        self.play(point.animate.shift((-1, 0, 0)), run_time=0.5)
        self.play(point.animate.shift((0.5, 0, 0)), run_time=0.3)

        self.wait(2)

        self.play(FadeOut(point))

        self.wait(4)

        self.play(line.animate.rotate(angle=PI / 8), run_time=1)
        right_side_fill = Polygon(
            line.get_start(),
            line.get_end(),
            line.get_end() + 15 * DOWN,
            line.get_start() + 6 * DOWN,
            color=BLUE,
            fill_opacity=0.3,
        )
        left_side_fill = Polygon(
            line.get_start(),
            line.get_end(),
            line.get_end() + 6 * UP,
            line.get_start() + 15 * UP,
            color=RED,
            fill_opacity=0.3,
        )
        line.set_z_index(1)
        self.play(FadeIn(right_side_fill), FadeIn(left_side_fill))

        self.wait(9)

        colored_hyperplane = VGroup(line, right_side_fill, left_side_fill)

        self.play(Rotate(colored_hyperplane, angle=PI), run_time=0.6)

        self.wait(12)

        normal = Arrow(start=(0, 0, 0), end=(-2, 0, 0), color=BLUE, buff=0).rotate(-3 * PI / 8, about_point=(0, 0, 0))
        self.play(Create(normal))

        self.wait(7)

        self.play(FadeOut(left_side_fill), FadeOut(right_side_fill))

        self.wait(2)

        line_with_normal = VGroup(line, normal)
        self.play(Rotate(line_with_normal, angle=PI), run_time=0.6)

        self.wait(6)
