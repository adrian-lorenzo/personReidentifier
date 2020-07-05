import os
from pathlib import Path

import cv2 as cv
import h5py

embeddingDatasetName = 'embedding'


def persistEmbedding(path, embedding):
    h5f = h5py.File(path, 'w')
    h5f.create_dataset(embeddingDatasetName, data=embedding)
    h5f.close()


def getEmbeddingFromDisk(path):
    h5f = h5py.File(path, 'r')
    embedding = h5f[embeddingDatasetName][:]
    h5f.close()

    return embedding


def persistEmbeddingsToDisk(embeddings, personId, area, basePath):
    path = "%s/%d/%d" % (basePath, personId, area)
    Path(path).mkdir(parents=True, exist_ok=True)
    for index, embedding in enumerate(embeddings):
        persistEmbedding("%s/%d.h5" % (path, index + 1), embedding)


def loadEmbeddingsDataset(embPath, ids, locations, oneIndexed=False):
    idx = 1 if oneIndexed else 0
    dataset = {}
    for index, location in enumerate(locations):
        embeddings = {}
        for id in ids:
            embeddings[id] = []
            path = "%s/%d/%d" % (embPath, id + idx, index + idx)
            for filename in os.listdir(path):
                embeddings[id].append(
                    getEmbeddingFromDisk(
                        "%s/%s" % (path, filename)
                    )
                )
        dataset[location] = embeddings
    return dataset


def loadEmbeddings(embPath, ids, location, oneIndexed=False):
    idx = 1 if oneIndexed else 0
    embeddings = {}
    for id in ids:
        embeddings[id] = []
        path = "%s/%d/%d" % (embPath, id + idx, location + idx)
        for filename in os.listdir(path):
            embeddings[id].append(
                getEmbeddingFromDisk(
                    "%s/%s" % (path, filename)
                )
            )
    return embeddings


def persistImage(path, name, image):
    Path(path).mkdir(parents=True, exist_ok=True)
    cv.imwrite("%s/%s" % (path, name), image)


def loadImage(path):
    return cv.imread(path, 1)


def getImagesDataset(basePath, ids, locations, oneIndexed=False):
    idx = 1 if oneIndexed else 0
    dataset = {}
    for index, location in enumerate(locations):
        images = {}
        for id in ids:
            images[id] = []
            path = "%s/%d/%d" % (basePath, id, index + idx)
            for filename in os.listdir(path):
                images[id].append(
                    loadImage("%s/%s" % (path, filename))
                )
        dataset[location] = images

    return dataset
