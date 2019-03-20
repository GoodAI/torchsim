from dataclasses import dataclass
from typing import NamedTuple


# TODO convert to @dataclass
class Size2D(NamedTuple):
    height: int
    width: int


@dataclass
class Point2D:
    y: int
    x: int
