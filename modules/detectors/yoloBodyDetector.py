import cv2 as cv
import numpy as np

from models.boundingBox import BoundingBox
from models.vector import Vector
from modules.detectors.bodyDetector import BodyDetector


class YoloBodyDetector(BodyDetector):
    modelConf = "../pretrained_models/yolov3/yolov3.cfg"
    weights = "../pretrained_models/yolov3/yolov3.weights"
    inputDim = (416, 416)

    def __init__(self):
        self.model = cv.dnn.readNetFromDarknet(self.modelConf, self.weights)
        self.model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def getOutputLayers(self):
        layersNames = self.model.getLayerNames()
        return [layersNames[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]

    def getBoundingBoxes(self, image, threshold=0.7):
        blob = cv.dnn.blobFromImage(image, 1 / 255., self.inputDim, swapRB=True, crop=False)
        self.model.setInput(blob)
        outs = self.model.forward(self.getOutputLayers())

        boxes = []
        for out in outs:
            for detection in out:
                id = np.argmax(detection[5:])
                score = detection[5:][id]
                if id == 0 and score > threshold:
                    center = Vector(detection[0] * image.shape[1], detection[1] * image.shape[0])
                    width = detection[2] * image.shape[1]
                    height = detection[3] * image.shape[0]
                    origin = Vector(int(center.x - width / 2), int(center.x - height / 2))
                    boxes.append(BoundingBox(
                        origin,
                        Vector(
                            int(origin.x + width),
                            int(origin.y + height)
                        )
                    ))

        return image, boxes
