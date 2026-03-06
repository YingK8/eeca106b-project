from dataclasses import dataclass


@dataclass
class CircularObstacle:
    cx: float
    cy: float
    radius: float


def default_obstacle_scene() -> list[CircularObstacle]:
    return [
        CircularObstacle(cx=1.5, cy=1.5, radius=0.5),
        CircularObstacle(cx=3.0, cy=3.0, radius=0.6),
        CircularObstacle(cx=4.0, cy=1.5, radius=0.4),
    ]
