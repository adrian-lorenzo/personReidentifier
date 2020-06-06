from pathlib import Path

import h5py
import cv2 as cv

embeddingDatasetName = 'embedding'

def persistEmbeddingsToDisk(embeddings, personId=1, area=1, basePath="/Users/adrianlorenzomelian/embeddings"):
    path = "%s/%d/%d" % (basePath, personId, area)
    Path(path).mkdir(parents=True, exist_ok=True)
    for index, embedding in enumerate(embeddings):
        persistEmbedding("%s/%d.h5" % (path, index + 1), embedding)

def persistEmbedding(path, embedding):
    h5f = h5py.File(path, 'w')
    h5f.create_dataset(embeddingDatasetName, data=embedding)
    h5f.close()

def getEmbeddingFromDisk(path):
    h5f = h5py.File(path, 'r')
    embedding = h5f[embeddingDatasetName][:]
    h5f.close()

    return embedding

def persistImage(path, image):
    cv.imwrite(path, image)

def loadImage(path):
    cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
