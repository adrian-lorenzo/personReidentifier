import cv2 as cv
import numpy as np

from models.boundingBox import BoundingBox
from models.vector import Vector
from modules.detectors.bodyDetector import BodyDetector


class YoloBodyDetector(BodyDetector):
    inputDim = (416, 416)
    nmsThreshold = 0.3

    def __init__(self, weights="../pretrained_models/yolov3/yolov3.weights",
                 modelConf="../pretrained_models/yolov3/yolov3.cfg"):
        self.model = cv.dnn.readNetFromDarknet(modelConf, weights)
        self.model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def getOutputLayers(self):
        layersNames = self.model.getLayerNames()
        return [layersNames[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]

    def getBoundingBoxes(self, image, threshold=0.5):
        blob = cv.dnn.blobFromImage(image, 1 / 255., self.inputDim, swapRB=False, crop=False)
        self.model.setInput(blob)
        outs = self.model.forward(self.getOutputLayers())

        boxVectors = []
        confidence = []
        for out in outs:
            for detection in out:
                id = np.argmax(detection[5:])
                score = detection[5:][id]
                if id == 0 and score > threshold:
                    boxVector = detection[:4] * np.array(
                        [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    boxVectors.append([
                        int(boxVector[0] - (boxVector[2] / 2)),
                        int(boxVector[1] - (boxVector[3] / 2)),
                        int(boxVector[2]),
                        int(boxVector[3])
                    ])

                    confidence.append(float(score))

        boxes = []
        if boxVectors:
            indexes = cv.dnn.NMSBoxes(boxVectors, confidence, threshold, self.nmsThreshold)
            boxes = [
                BoundingBox(
                    Vector(
                        boxVector[0],
                        boxVector[1]
                    ),
                    Vector(
                        boxVector[0] + boxVector[2],
                        boxVector[1] + boxVector[3]
                    )
                ) for boxVector in map(lambda i: boxVectors[i[0]], indexes)
            ]
        return image, boxes
