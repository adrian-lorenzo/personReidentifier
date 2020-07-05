import os
from contextlib import redirect_stderr
from os import path

import cv2 as cv

from models.galleryDatabase import GalleryDatabase
from modules.detectors.yoloBodyDetector import YoloBodyDetector
from modules.detectors.fasterBodyDetector import FasterBodyDetector
from modules.embeddingGenerators.abdEmbeddingGenerator import AbdEmbeddingGenerator
from modules.embeddingGenerators.alignedReIdEmbeddingGenerator import AlignedReIdEmbeddingGenerator
from modules.embeddingGenerators.faceEmbeddingGenerator import FaceEmbeddingGenerator
from models.detector import Detector
from models.embeddingGenerator import EmbeddingGenerator
from utils.debugUtils import printv, printDone
from utils.imageUtils import drawBoundingBox, drawId, drawMask, deinterlaceImages

# Remove Keras Backend warnings
with redirect_stderr(open(os.devnull, "w")):
    from modules.detectors.faceDetector import FaceDetector
    from modules.detectors.maskBodyDetector import MaskBodyDetector

# Suppress Tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class PersonReidentifierService():
    # Image processing constants
    inputDim = (1280, 720)
    presentationDim = (1280, 720)
    interpolationMethod = cv.INTER_AREA

    # Window constants
    frameLabel = "Person identifier"

    def __init__(self, detector=Detector.mask,
                 detectionThreshold=0.8,
                 embeddingGenerator=EmbeddingGenerator.alignedReId,
                 detectorWeights=None,
                 detectorConf=None,
                 embeddingGeneratorWeights=None,
                 embeddingGeneratorNumClasses=None,
                 gallery=None,
                 databaseThreshold=1.5,
                 maxDescriptors=None):
        self.detector = detector

        if detector is Detector.mtcnn:
            self.faceDetector = FaceDetector()
            self.faceEmbeddingGenerator = FaceEmbeddingGenerator()
        else:
            self.bodyEmbeddingGenerator = self.selectEmbeddingGenerator(
                embeddingGenerator,
                embeddingGeneratorWeights,
                embeddingGeneratorNumClasses
            )
            self.bodyDetector = self.selectDetector(
                detector, detectorWeights, detectorConf
            )

        self.galleryDatabase = GalleryDatabase(
            database=gallery if gallery is not None else {},
            maxDescriptors=maxDescriptors,
            threshold=databaseThreshold
        )
        self.detectionThreshold = detectionThreshold

    def selectDetector(self, detector, detectorWeights, detectorConf):
        return {
            Detector.mask: MaskBodyDetector,
            Detector.faster: FasterBodyDetector,
            Detector.yolo: YoloBodyDetector,
            Detector.mtcnn: FaceDetector
        }.get(detector, MaskBodyDetector)(detectorWeights, detectorConf)

    def selectEmbeddingGenerator(self, embeddingGenerator, embeddingsGeneratorWeights, numClasses):
        return {
            EmbeddingGenerator.alignedReId: AlignedReIdEmbeddingGenerator,
            EmbeddingGenerator.abd: AbdEmbeddingGenerator
        }.get(embeddingGenerator, AlignedReIdEmbeddingGenerator)(embeddingsGeneratorWeights, numClasses)

    def getAllFaces(self, frame):
        frame = cv.resize(frame, self.inputDim, interpolation=self.interpolationMethod)
        faces = self.faceDetector.getFaces(frame)
        if faces is not None:
            return [face.boundingBox.getImageFromBox(frame) for face in faces]
        return None

    def getFacesFromImage(self, image):
        printv("Getting faces images from image...")
        boxes = [face.boundingBox for face in self.faceDetector.getFaces(image)]
        faces = [box.getImageFromBox(image) for box in boxes]
        printDone()
        return faces

    def getBodiesFromImage(self, image):
        printv("Getting body images from image...")
        boxes = self.bodyDetector.getBoundingBoxes(image)[1]
        bodies = [box.getImageFromBox(image) for box in boxes]
        printDone()
        return bodies

    def getEmbeddingFromBody(self, body):
        printv("Getting embedding from body image...")
        embedding = self.bodyEmbeddingGenerator.getEmbedding(body)
        printDone()
        return embedding

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
        faces = self.faceDetector.getFaces(frame)
        if faces is not None:
            for face in faces:
                faceImage = face.boundingBox.getImageFromBox(frame)
                id = self.galleryDatabase.getIdentity(self.bodyEmbeddingGenerator.getEmbedding(faceImage))
                frame = drawBoundingBox(frame, face.boundingBox)
                frame = drawId(frame, id, face.boundingBox)
        return frame

    def detectBodiesInFrame(self, frame):
        frame, boxes = self.bodyDetector.getBoundingBoxes(frame, threshold=self.detectionThreshold)
        if boxes is not None:
            for box in boxes:
                frame = drawBoundingBox(frame, box)
        return frame

    def detectFacesInFrame(self, frame):
        boxes = [face.boundingBox for face in self.faceDetector.getFaces(frame)]
        if boxes is not None:
            for box in boxes:
                frame = drawBoundingBox(frame, box)
        return frame

    def identifyBodyInFrame(self, frame):
        frame, boxes = self.bodyDetector.getBoundingBoxes(frame, threshold=self.detectionThreshold)
        if boxes is not None:
            for box in boxes:
                bodyImage = box.getImageFromBox(frame)
                id = self.galleryDatabase.getIdentity(self.bodyEmbeddingGenerator.getEmbedding(bodyImage))
                frame = drawBoundingBox(frame, box)
                frame = drawId(frame, id, box)
        return frame

    def startIdentificationByVideo(self, videoPath):
        printv("Starting identification by video...")
        if not path.exists(videoPath):
            raise FileNotFoundError("Video path does not exist")
        identify = self.identifyFaceInFrame if self.detector is Detector.mtcnn else self.identifyBodyInFrame
        cap = cv.VideoCapture(videoPath)
        printv("⚡️ Identification by video started, use key 'q' to quit the process.")
        while (cap.isOpened()):
            _, frame = cap.read()
            newFrame = identify(deinterlaceImages([frame])[0])
            cv.imshow(self.frameLabel, newFrame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
        printDone()

    def startIdentificationByCam(self):
        printv("Starting identification using webcam...")
        identify = self.identifyFaceInFrame if self.detector is Detector.mtcnn else self.identifyBodyInFrame
        cap = cv.VideoCapture(0)
        printv("⚡️ Identification using webcam started, use key 'q' to quit the process.")
        while True:
            _, frame = cap.read()
            newFrame = identify(deinterlaceImages([frame])[0])
            cv.imshow(self.frameLabel, newFrame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
        printDone()
