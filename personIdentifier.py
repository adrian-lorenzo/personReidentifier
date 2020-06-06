import cv2 as cv

from os import path
from models.galleryDatabase import GalleryDatabase
from modules.embeddingGenerators.abdEmbeddingGenerator import AbdEmbeddingGenerator
from modules.detectors.faceDetector import FaceDetector
from modules.embeddingGenerators.alignedReIdEmbeddingGenerator import AlignedReIdEmbeddingGenerator
from modules.embeddingGenerators.faceEmbeddingGenerator import FaceEmbeddingGenerator
from modules.detectors.fasterBodyDetector import FasterBodyDetector
from modules.detectors.maskBodyDetector import MaskBodyDetector
from modules.detectors.yoloBodyDetector import YoloBodyDetector
from utils.detector import Detector
from utils.embeddingGenerator import EmbeddingGenerator
from utils.imageUtils import drawBoundingBox, drawId, drawMask, deinterlaceImages
from utils.persistanceUtils import persistImage


class PersonIdentifier():
    # Image processing constants
    inputDim = (1280, 720)
    presentationDim = (1280, 720)
    interpolationMethod = cv.INTER_AREA

    # Window constants
    frameLabel = "Person identifier"

    def __init__(self, detector=Detector.mask, embeddingGenerator=EmbeddingGenerator.alignedReId, detectionThreshold=0.8):
        self.faceDetector = FaceDetector()
        self.faceDescriptorGenerator = FaceEmbeddingGenerator()
        self.bodyEmbeddingGenerator = self.selectEmbeddingGenerator(embeddingGenerator)
        self.galleryDatabase = GalleryDatabase()
        self.detector = detector
        self.bodyDetector = self.selectDetector(detector)
        self.detectionThreshold = detectionThreshold


    def selectDetector(self, detector):
        return {
            Detector.mask: MaskBodyDetector,
            Detector.faster: FasterBodyDetector,
            Detector.yolo: YoloBodyDetector,
            Detector.mtcnn: FaceDetector
        }.get(detector, MaskBodyDetector())()


    def selectEmbeddingGenerator(self, embeddingGenerator):
        return {
            EmbeddingGenerator.alignedReId : AlignedReIdEmbeddingGenerator,
            EmbeddingGenerator.abd : AbdEmbeddingGenerator
        }.get(embeddingGenerator, AlignedReIdEmbeddingGenerator)()


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

    def getBodiesFromImage(self, image):
        boxes = self.bodyDetector.getBoundingBoxes(image)[1]
        bodies = [box.getImageFromBox(image) for box in boxes]
        return bodies

    def getEmbeddingFromBody(self, body):
        return self.bodyEmbeddingGenerator.getEmbedding(body)

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
                #id = self.galleryDatabase.getIdentity(self.bodyEmbeddingGenerator.getEmbedding(faceImage))
                frame = drawBoundingBox(frame, face.boundingBox)
                #frame = drawId(frame, id, box)
        return frame


    def identifyBodyInFrame(self, frame):
        if self.detector == Detector.yolo:
            frame = cv.resize(frame, self.inputDim, interpolation=self.interpolationMethod)
        frame, boxes = self.bodyDetector.getBoundingBoxes(frame, threshold=self.detectionThreshold)
        if boxes is not None:
            for box in boxes:
                bodyImage = box.getImageFromBox(frame)
                #id = self.galleryDatabase.getIdentity(self.bodyEmbeddingGenerator.getEmbedding(bodyImage))
                frame = drawBoundingBox(frame, box)
                #frame = drawId(frame, id, box)
        return cv.resize(frame, self.presentationDim, interpolation=self.interpolationMethod)


    def startIdentificationByVideo(self, videoPath):
        pth = "/Users/adrianlorenzomelian/tfg/report/images/detection/yolo"
        if not path.exists(videoPath):
            raise FileNotFoundError("Video path does not exist")

        cap = cv.VideoCapture(videoPath)
        cap.set(cv.CAP_PROP_POS_MSEC, 30000)
        for i in range(10):
        #while (cap.isOpened()):
            _, frame = cap.read()
            #newFrame = deinterlaceImages([frame])[0]
            newFrame = self.identifyBodyInFrame(deinterlaceImages([frame])[0])
            #newFrame = self.identifyFaceInFrame(deinterlaceImages([frame])[0])
            persistImage("%s/%d.png" % (pth, i), newFrame)
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
