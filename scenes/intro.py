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
    Text,
    Arrow,
    Circle,
    FadeIn,
    FadeOut,
    Line,
    ThreeDScene,
    Polygon,
    Rectangle,
    ValueTracker,
    VGroup,
    always_redraw,
    rate_functions,
    ManimColor,
)
from noise.hypersphere import hyperspheric_noise

class Intro(ThreeDScene):
    def construct(self: Self) -> None:
        pass
