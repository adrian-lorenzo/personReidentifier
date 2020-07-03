from enum import Enum


class Detector(Enum):
    mask = 0
    mtcnn = 1
    yolo = 2
    faster = 3
