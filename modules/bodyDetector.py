import tensorflow.compat.v1 as tf1
import numpy as np
from models.point import Point
from models.boundingBox import BoundingBox


class BodyDetector:
    path = "../pretrained_models/faster_rcnn_coco/frozen_inference_graph.pb"

    def __init__(self):
        with tf1.gfile.GFile(self.path, "rb") as f:
            graph_def = tf1.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf1.Graph()
        tf1.import_graph_def(graph_def, name="")
        self.graph = tf1.get_default_graph()

        self.inputImage = self.graph.get_tensor_by_name('image_tensor:0')
        self.detectionClassesTensor = self.graph.get_tensor_by_name('detection_classes:0')
        self.numDetectionsTensor = self.graph.get_tensor_by_name('num_detections:0')
        self.detectionBoxesTensor = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detectionScoresTensor = self.graph.get_tensor_by_name('detection_scores:0')

    def getBoundingBoxes(self, image, threshold=0.8):
        with tf1.Session(graph=self.graph) as sess:
            detectionClasses, numDetections, detectionBoxes, detectionScores = sess.run(
                [self.detectionClassesTensor, self.numDetectionsTensor,
                 self.detectionBoxesTensor, self.detectionScoresTensor],
                feed_dict={self.inputImage: np.expand_dims(image, axis=0)}
            )

        boxes = []
        for i in range(int(numDetections[0])):
            if detectionClasses[0][i] == 1 and detectionScores[0][i] > threshold:
                boxes.append(BoundingBox(
                    Point(
                        int(detectionBoxes[0][i][1] * image.shape[1]),
                        int(detectionBoxes[0][i][0] * image.shape[0])
                    ),
                    Point(
                        int(detectionBoxes[0][i][3] * image.shape[1]),
                        int(detectionBoxes[0][i][2] * image.shape[0])
                    )
                ))

        return boxes
