import cv2 as cv
import numpy as np

# Bounding box constants
boundingBoxColor = (165, 214, 167)
boundingBoxBorderSize = 1
maskColor = (145, 255, 117)

# Id text constants
font = cv.FONT_HERSHEY_SIMPLEX
fontSize = 0.5
fontColor = (0, 255, 0)
lineType = 2

def drawBoundingBox(frame, boundingBox):
    cv.rectangle(frame,
                 (boundingBox.origin.x, boundingBox.origin.y),
                 (boundingBox.end.x, boundingBox.end.y),
                 boundingBoxColor,
                 boundingBoxBorderSize)
    return frame


def drawMask(frame, mask):
    for c in range(3):
        frame[:, :, c] = np.where(mask == 1,
                                  frame[:, :, c] * 0.5 + 0.5 * maskColor[c] * 255,
                                  frame[:, :, c])
    return frame


def drawId(frame, id, box):
    return cv.putText(frame,
                      "Id: %d" % id,
                      (box.origin.x, box.origin.y - 16),
                      font,
                      fontSize,
                      fontColor,
                      lineType)