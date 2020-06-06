from pathlib import Path

import numpy as np
import cv2 as cv

from personIdentifier import PersonIdentifier
from utils.imageUtils import getImages, deinterlaceImages
from utils.persistanceUtils import persistEmbedding, persistEmbeddingsToDisk, getEmbeddingFromDisk


def getEmbeddingsFromDisk(personId=1, area=1, basePath="/Users/adrianlorenzomelian/embeddings", numSamples=4):
    return np.array([
        getEmbeddingFromDisk(
            "%s/%d/%d/%d.h5" % (basePath, personId, area, i)
        )
        for i in range(1, numSamples + 1)
    ])

def getEmbeddingsFromDiskNoArea(personId=1, basePath="/Users/adrianlorenzomelian/embeddings", numSamples=5):
    return np.array([
        getEmbeddingFromDisk(
            "%s/%d/%d.h5" % (basePath, personId, i)
        )
        for i in range(1, numSamples + 1)
    ])

def getSampleImagesNoArea(personId=1, basePath="/Users/adrianlorenzomelian/dataset", numSamples=4):
    return np.array([
        cv.cvtColor(cv.imread("%s/%d/%d.png" % (basePath, personId, i), 1), cv.COLOR_BGR2RGB)
        for i in range(1, numSamples + 1)
    ])

def persistEmbeddingsToDiskNoArea(embeddings, personId=1, basePath="/Users/adrianlorenzomelian/embeddings"):
    path = "%s/%d" % (basePath, personId)
    Path(path).mkdir(parents=True, exist_ok=True)
    for index, embedding in enumerate(embeddings):
        persistEmbedding("%s/%d.h5" % (path, index + 1), embedding)

def persistImagesToDiskNoArea(images, personId=1, basePath="/Users/adrianlorenzomelian/bodies"):
    path = "%s/%d" % (basePath, personId)
    Path(path).mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        cv.imwrite("%s/%d.png" % (path, index + 1), image[..., ::-1])


def saveEmbeddingsDisk():
    identifier = PersonIdentifier()
    for i in range(1, 101):
        for j in range(1, 3):
            bodies = getImages(personId=i, area=j, basePath="/Users/adrianlorenzomelian/bodies")
            persistEmbeddingsToDisk([identifier.getEmbeddingFromBody(body) for body in bodies], personId=i, area=j,
                                    basePath="/Users/adrianlorenzomelian/abd_embeddings")


def saveEmbeddingsDiskIds(ids):
    identifier = PersonIdentifier()
    for id in ids:
        bodies = deinterlaceImages(
            getSampleImagesNoArea(personId=id, basePath="/Users/adrianlorenzomelian/test-datasets/new-test-bodies",
                                  numSamples=5))
        # persistImagesToDiskNoArea(bodies, personId=id, basePath="/Users/adrianlorenzomelian/Desktop/test-datasets/new-test-bodies2")
        persistEmbeddingsToDiskNoArea([identifier.getEmbeddingFromBody(body) for body in bodies], personId=id,
                                      basePath="/Users/adrianlorenzomelian/test-datasets/new-test-embeddings2")