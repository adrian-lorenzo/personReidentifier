from models.point import Point


class BoundingBox:
    def __init__(self, origin: Point, end: Point):
        self.origin = origin
        self.end = end

    def getImageFromBox(self, frame):
        return frame[
               self.origin.y: self.end.y,
               self.origin.x: self.end.x
               ]
