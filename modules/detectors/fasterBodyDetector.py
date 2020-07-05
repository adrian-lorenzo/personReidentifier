import cv2 as cv
import numpy as np
import tensorflow.compat.v1 as tf

from models.boundingBox import BoundingBox
from models.vector import Vector
from modules.detectors.bodyDetector import BodyDetector


class FasterBodyDetector(BodyDetector):

    def __init__(self, weights="../pretrained_models/faster_rcnn_coco/frozen_inference_graph.pb",
                 modelConf=None):

        self.modelConf = modelConf
        with tf.gfile.GFile(weights, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.Graph()
        tf.import_graph_def(graph_def, name="")
        self.graph = tf.get_default_graph()

        self.inputImage = self.graph.get_tensor_by_name('image_tensor:0')
        self.detectionClassesTensor = self.graph.get_tensor_by_name('detection_classes:0')
        self.numDetectionsTensor = self.graph.get_tensor_by_name('num_detections:0')
        self.detectionBoxesTensor = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detectionScoresTensor = self.graph.get_tensor_by_name('detection_scores:0')

    def preprocessImage(self, image):
        image = cv.normalize(image, image, 0, 255, cv.NORM_MINMAX)
        return image

    def getBoundingBoxes(self, image, threshold=0.7):
        with tf.Session(graph=self.graph) as sess:
            detectionClasses, numDetections, detectionBoxes, detectionScores = sess.run(
                [self.detectionClassesTensor, self.numDetectionsTensor,
                 self.detectionBoxesTensor, self.detectionScoresTensor],
                feed_dict={self.inputImage: np.expand_dims(self.preprocessImage(image), axis=0)}
            )

        boxes = []
        for i in range(int(numDetections[0])):
            if detectionClasses[0][i] == 1 and detectionScores[0][i] > threshold:
                boxes.append(BoundingBox(
                    Vector(
                        int(detectionBoxes[0][i][1] * image.shape[1]),
                        int(detectionBoxes[0][i][0] * image.shape[0])
                    ),
                    Vector(
                        int(detectionBoxes[0][i][3] * image.shape[1]),
                        int(detectionBoxes[0][i][2] * image.shape[0])
                    )
                ))

        return image, boxes
