from models.vector import Vector


class BoundingBox:
    def __init__(self, origin: Vector, end: Vector):
        self.origin = origin
        self.end = end

    def getImageFromBox(self, frame):
        return frame[
               self.origin.y: self.end.y,
               self.origin.x: self.end.x
               ]
    def __str__(self):
        return "Origin: " + str(self.origin) + "- End: " + str(self.end)