#!/usr/local/bin/python3
import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN

from modules.bodyEmbeddingGenerator import BodyEmbeddingGenerator
from modules.faceDetector import FaceDetector
from modules.faceEmbeddingGenerator import FaceEmbeddingGenerator
from models.gallery import Gallery
from models.galleryDatabase import GalleryDatabase
from modules.bodyDetector import BodyDetector


class PersonIdentifier():
    # Image processing constants
    inputDim = (600, 600)
    presentationDim = (1200, 720)
    interpolationMethod = cv.INTER_AREA

    # Bounding box constants
    boundingBoxColor = (165, 214, 167)
    boundingBoxBorderSize = 1

    # Id text constants
    font = cv.FONT_HERSHEY_SIMPLEX
    fontSize = 0.5
    fontColor = (0, 255, 0)
    lineType = 2

    # DBSCAN constants
    epsilon = 1.5
    minSamplesToCluster = 10

    # Window constants
    frameLabel = "Person identifier"

    def __init__(self):
        self.faceDetector = FaceDetector()
        self.faceDescriptorGenerator = FaceEmbeddingGenerator()
        self.bodyEmbeddingGenerator = BodyEmbeddingGenerator()
        self.galleryDatabase = GalleryDatabase()
        self.clusteringSystem = DBSCAN(eps=self.epsilon, min_samples=self.minSamplesToCluster)
        self.bodyDetector = BodyDetector()

    def identify(self, descriptor):
        for key, gallery in self.galleryDatabase.database.items():
            scan = DBSCAN(eps=self.epsilon,
                          min_samples=min([max([1, len(gallery.descriptors)]), self.minSamplesToCluster]))
            if scan.fit_predict(np.vstack((gallery.descriptors, descriptor)))[-1] != -1:
                self.galleryDatabase.addToGallery(key, descriptor)
                return key
        return self.galleryDatabase.addNewGallery(Gallery(np.array([descriptor])))

    def drawBoundingBox(self, frame, boundingBox):
        cv.rectangle(frame,
                     (boundingBox.origin.x, boundingBox.origin.y),
                     (boundingBox.end.x, boundingBox.end.y),
                     self.boundingBoxColor,
                     self.boundingBoxBorderSize)
        return frame

    def drawId(self, frame, id, box):
        return cv.putText(frame,
                          "Id: %d" % id,
                          (box.origin.x, box.origin.y - 16),
                          self.font,
                          self.fontSize,
                          self.fontColor,
                          self.lineType)

    def getAllAlignedFaces(self, frame):
        frame = cv.resize(frame, self.inputDim, interpolation=self.interpolationMethod)
        faces = self.faceDetector.getFaces(frame)
        if faces is not None:
            return [face.getAlignedFaceImage(frame) for face in faces]
        return None

    def getAllFaces(self, frame):
        frame = cv.resize(frame, self.inputDim, interpolation=self.interpolationMethod)
        faces = self.faceDetector.getFaces(frame)
        if faces is not None:
            return [faces.boundingBox.getImageFromBox(frame) for face in faces]
        return None

    def identifyFaceInFrame(self, frame):
        frame = cv.resize(frame, self.inputDim, interpolation=self.interpolationMethod)
        faces = self.faceDetector.getFaces(frame)
        if faces is not None:
            for face in faces:
                faceImage = face.boundingBox.getImageFromBox(frame)
                id = self.identify(self.faceDescriptorGenerator.getDescriptor(faceImage))
                frame = self.drawBoundingBox(frame, face.boundingBox)
                frame = self.drawId(frame, id, face)
        return cv.resize(frame, self.presentationDim, interpolation=self.interpolationMethod)

    def identifyBodyInFrame(self, frame):
        #frame = cv.resize(frame, self.inputDim, interpolation=self.interpolationMethod)
        boxes = self.bodyDetector.getBoundingBoxes(frame)
        if boxes is not None:
            for box in boxes:
                bodyImage = box.getImageFromBox(frame)
                id = self.identify(self.bodyEmbeddingGenerator.getEmbedding(bodyImage))
                frame = self.drawBoundingBox(frame, box)
                frame = self.drawId(frame, id, box)
        return frame #cv.resize(frame, self.presentationDim, interpolation=self.interpolationMethod)

    def startIdentificationByVideo(self, videoPath):
        cap = cv.VideoCapture(videoPath)
        while (cap.isOpened()):
            _, frame = cap.read()
            newFrame = self.identifyBodyInFrame(frame)
            cv.imshow(self.frameLabel, newFrame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

    def startIdentificationByCam(self):
        cap = cv.VideoCapture(0)
        while True:
            _, frame = cap.read()
            newFrame = self.identifyBodyInFrame(frame)
            cv.imshow(self.frameLabel, newFrame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
