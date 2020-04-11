import cv2 as cv

from mtcnn import MTCNN
from models.vector import Vector
from models.face import Face
from models.boundingBox import BoundingBox


class FaceDetector():
    interpolationMethod = cv.INTER_AREA
    paddingMultiplier = 1.05

    def __init__(self):
        self.model = MTCNN()

    def getFaces(self, image):
        faces = self.model.detect_faces(image)
        if faces is not None:
            return [
                Face(
                    boundingBox=BoundingBox(
                        origin=Vector(
                            int(face["box"][0]),
                            int(face["box"][1])
                        ),
                        end=Vector(
                            int(face["box"][0] + face["box"][2]),
                            int(face["box"][1] + face["box"][3])
                        ),
                    ),

                    leftEye=Vector(
                        int(face["keypoints"]["left_eye"][0]),
                        int(face["keypoints"]["left_eye"][1])
                    ),

                    rightEye=Vector(
                        int(face["keypoints"]["right_eye"][0]),
                        int(face["keypoints"]["right_eye"][1])
                    ),

                    nose=Vector(
                        int(face["keypoints"]["nose"][0]),
                        int(face["keypoints"]["nose"][1])
                    )
                )

                for face in faces
            ]
