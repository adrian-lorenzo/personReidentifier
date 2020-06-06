import cv2 as cv
import numpy as np

from models.boundingBox import BoundingBox
from models.vector import Vector


class Face:
    def __init__(self, boundingBox: BoundingBox, leftEye: Vector, rightEye: Vector, nose: Vector):
        self.boundingBox = boundingBox
        self.leftEye = leftEye
        self.rightEye = rightEye
        self.nose = nose

    def getAlignedFaceImage(self, frame):
        angle = self.leftEye.rotation(self.rightEye)
        eyesCenter = self.leftEye.midpoint(self.rightEye)
        direction = self.boundingBox.origin.direction(self.boundingBox.end)
        xRotated = ((direction.x - self.boundingBox.origin.x) * np.cos(angle)) - (
                (self.boundingBox.origin.y - direction.y) * np.sin(angle)) + self.boundingBox.origin.x
        yRotated = ((self.boundingBox.origin.y - direction.y) * np.cos(angle)) - (
                (direction.x - self.boundingBox.origin.x) * np.sin(angle)) + self.boundingBox.origin.y

        rotationMatrix = cv.getRotationMatrix2D((eyesCenter.x, eyesCenter.y), angle, 1)

        output = cv.warpAffine(
            frame,
            rotationMatrix,
            (frame.shape[0], frame.shape[1]),
            flags=cv.INTER_CUBIC)

        faceImage = output[self.boundingBox.origin.y:int(yRotated + self.boundingBox.origin.y),
                    self.boundingBox.origin.x:int(xRotated + self.boundingBox.origin.x)]

        return faceImage
