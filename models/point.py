import numpy as np


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def midpoint(self, other):
        return Point(
            self.x + other.x // 2,
            self.y + other.y // 2
        )

    def direction(self, other):
        return Point(
            other.x - self.x,
            other.y - self.y
        )

    def distance(self, other):
        direction = self.direction(other)
        return np.sqrt(direction.x ** 2 + direction.y ** 2)

    def rotation(self, other):
        direction = self.direction(other)
        return -np.arctan2(direction.y, direction.x)

    def __str__(self):
        return "x: %d - y: %d" % (self.x, self.y)
