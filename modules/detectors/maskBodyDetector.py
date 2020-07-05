import os

import cv2 as cv

import modules.detectors.mrcnn.model as modellib
from models.boundingBox import BoundingBox
from models.vector import Vector
from modules.detectors.bodyDetector import BodyDetector
from modules.detectors.mrcnn import utils
from modules.detectors.mrcnn.config import Config


class InferenceConfig(Config):
    NAME = "coco"
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1
    NUM_CLASSES = 1 + 80


class MaskBodyDetector(BodyDetector):

    def __init__(self, weights="../pretrained_models/mask_rcnn_coco/mask_rcnn_coco.h5",
                 modelConf="../.logs"):
        if modelConf is None:
            modelConf = "../.logs"
        if not os.path.exists(weights):
            utils.download_trained_weights(weights)
        config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference", model_dir=modelConf, config=config)
        self.model.load_weights(weights, by_name=True)

    def preprocessImage(self, image):
        image = cv.normalize(image, image, 0, 255, cv.NORM_MINMAX)
        return image

    def getBoundingBoxes(self, image, threshold=0.8):
        bodies = self.model.detect([self.preprocessImage(image)], verbose=0)[0]
        boxes = []
        for id, score, box in zip(bodies["class_ids"], bodies["scores"], bodies["rois"]):
            if id == 1 and score > threshold:
                boxes.append(BoundingBox(
                    Vector(
                        int(box[1]),
                        int(box[0])
                    ),
                    Vector(
                        int(box[3]),
                        int(box[2])
                    )
                ))

        return image, boxes

    def getBoundingBoxesAndMasks(self, image, threshold=0.8):
        bodies = self.model.detect([self.preprocessImage(image)], verbose=0)[0]
        boxes = []
        masks = []
        for index, (id, score, box) in enumerate(zip(bodies["class_ids"], bodies["scores"], bodies["rois"])):
            if id == 1 and score > threshold:
                boxes.append(BoundingBox(
                    Vector(
                        int(box[1]),
                        int(box[0])
                    ),
                    Vector(
                        int(box[3]),
                        int(box[2])
                    )
                ))

                masks.append(bodies["masks"][:, :, index])

        return (boxes, masks)
