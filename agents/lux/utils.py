from enum import IntEnum
from typing import Literal

import sys
import numpy as np


Vector2 = np.ndarray[tuple[Literal[2]], np.dtype[np.int32]]


class Direction(IntEnum):
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


def direction_to(src: Vector2, target: Vector2) -> Direction:
    ds = target - src
    dx: int = ds[0]
    dy: int = ds[1]

    if dx == 0 and dy == 0:
        return Direction.CENTER

    if abs(dx) > abs(dy):
        if dx > 0:
            return Direction.RIGHT
        else:
            return Direction.LEFT
    else:
        if dy > 0:
            return Direction.DOWN
        else:
            return Direction.UP


def print_debug(*values: object, sep: str | None = " ", end: str | None = "\n") -> None:
    print(*values, sep=sep, end=end, file=sys.stderr)
