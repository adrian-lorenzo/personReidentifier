import cv2 as cv
import numpy as np

import os
import sys
stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
import tensorflow as tf
sys.stdout = stdout

class FaceEmbeddingGenerator():
    """
    WARNING - DEPRECATED!: This is a FaceNet implementation done to experiment with MTCNN.
    It is deprecated and not used in the current CLI implementation.
    It can be taken as reference to start working on face embedding generation.
    """
    inputDim = (160, 160)
    interpolationMethod = cv.INTER_AREA
    modelLocation = '../pretrained_models/facenet/facenet_keras.h5'
    weightsLocation = '../pretrained_models/facenet/facenet_keras_weights.h5'

    def __init__(self):
        self.model = tf.keras.models.load_model(self.modelLocation)
        self.model.load_weights(self.weightsLocation)

    def normalize(self, data):
        return (data - data.mean()) / data.std()

    def getDescriptor(self, face):
        face = cv.resize(face, self.inputDim, interpolation=self.interpolationMethod)
        face = self.normalize(face)
        return self.model.predict(np.expand_dims(face, axis=0))[0]
