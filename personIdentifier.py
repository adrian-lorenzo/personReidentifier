#!/usr/local/bin/python3
import cv2 as cv

from models.galleryDatabase import GalleryDatabase
from modules.bodyEmbeddingGenerator import BodyEmbeddingGenerator
from modules.faceDetector import FaceDetector
from modules.faceEmbeddingGenerator import FaceEmbeddingGenerator
from modules.fasterBodyDetector import FasterBodyDetector
from modules.maskBodyDetector import MaskBodyDetector
from modules.yoloBodyDetector import YoloBodyDetector
from utils.detectors import Detector
from utils.imageUtils import drawBoundingBox, drawId, drawMask


class PersonIdentifier():
    # Image processing constants
    inputDim = (416, 416)
    presentationDim = (1200, 720)
    interpolationMethod = cv.INTER_AREA

    # Window constants
    frameLabel = "Person identifier"

    def __init__(self, detector = Detector.mask, detectionThreshold = 0.8, identificationThreshold = 1.5):
        self.faceDetector = FaceDetector()
        self.faceDescriptorGenerator = FaceEmbeddingGenerator()
        self.bodyEmbeddingGenerator = BodyEmbeddingGenerator()
        self.galleryDatabase = GalleryDatabase(identificationThreshold=identificationThreshold)
        self.detector = detector
        self.bodyDetector = self.selectDetector(detector)
        self.detectionThreshold = detectionThreshold

    def selectDetector(self, detector):
        return {
            Detector.mask : MaskBodyDetector(),
            Detector.faster : FasterBodyDetector(),
            Detector.yolo : YoloBodyDetector(),
            Detector.mtcnn : FaceDetector()
        }.get(detector, MaskBodyDetector())

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

    def identifyBodyInFrameWithMask(self, frame):
        if type(self.bodyDetector) is MaskBodyDetector:
            frame = cv.resize(frame, self.inputDim, interpolation=self.interpolationMethod)
            boxes, masks = self.bodyDetector.getBoundingBoxesAndMasks(frame, threshold=self.detectionThreshold)
            if boxes is not None:
                for box, mask in zip(boxes, masks):
                    frame = drawBoundingBox(frame, box)
                    frame = drawMask(frame, mask)
        return frame

    def identifyFaceInFrame(self, frame):
        frame = cv.resize(frame, self.inputDim, interpolation=self.interpolationMethod)
        boxes = self.bodyDetector.getBoundingBoxes(frame, threshold=self.detectionThreshold)
        if boxes is not None:
            for box in boxes:
                bodyImage = box.getImageFromBox(frame)
                id = self.galleryDatabase.getIdentity(self.bodyEmbeddingGenerator.getEmbedding(bodyImage))
                frame = drawBoundingBox(frame, box)
                frame = drawId(frame, id, box)
        return frame

    def identifyBodyInFrame(self, frame):
        if self.detector == Detector.yolo:
            frame = cv.resize(frame, self.inputDim, interpolation=self.interpolationMethod)
        frame, boxes = self.bodyDetector.getBoundingBoxes(frame, threshold=self.detectionThreshold)
        if boxes is not None:
            for box in boxes:
                bodyImage = box.getImageFromBox(frame)
                id = self.galleryDatabase.getIdentity(self.bodyEmbeddingGenerator.getEmbedding(bodyImage))
                frame = drawBoundingBox(frame, box)
                frame = drawId(frame, id, box)
        return cv.resize(frame, self.presentationDim, interpolation=self.interpolationMethod)

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
