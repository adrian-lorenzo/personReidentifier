from models.vector import Vector


class BoundingBox:
    def __init__(self, origin: Vector, end: Vector):
        self.origin = origin
        self.end = end

    def getImageFromBox(self, frame):
        return frame[
           max(self.origin.y, 0): min(self.end.y, frame.shape[0]),
           max(self.origin.x, 0): min(self.end.x, frame.shape[1])
        ]

    def __str__(self):
        return "Origin: " + str(self.origin) + "- End: " + str(self.end)
