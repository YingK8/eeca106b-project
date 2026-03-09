from dataclasses import dataclass

# Chairs are ~27in diameter
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

def cory105_obstacle_scene() -> list[CircularObstacle]:
    return [
        # First chair is 48in ahead of 0,0
        CircularObstacle(cx=1.2192, cy=0.0, radius=0.3429),
        # Second chair is 48in ahead and 53in to the left of 0,0
        CircularObstacle(cx=1.2192, cy=1.3462, radius=0.3429),
    ]