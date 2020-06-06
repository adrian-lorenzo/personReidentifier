from pathlib import Path

from personIdentifier import PersonIdentifier
from utils.embeddingGenerator import EmbeddingGenerator
from utils.imageUtils import deinterlaceImages
import cv2 as cv
import re
import os

from utils.persistanceUtils import getEmbeddingFromDisk, persistEmbedding

tgcLocations = ["Arucas", "Ayagaures", "ParqueSur", "PresaDeHornos", "Teror"]

def getTGCDataset(basePath, ids):
    dataset = {}
    for location in tgcLocations:
        images = {}
        for id in ids:
            images[id] = []
            path = "%s/%d" % (basePath, id)
            for filename in os.listdir(path):
                if re.match("\d+_%s_frame_\d+_\d+_\d+_\d+.jpg" % (location), filename):
                    images[id].append(
                        deinterlaceImages([cv.cvtColor(cv.imread("%s/%s" % (path, filename), 1), cv.COLOR_BGR2RGB)])[0]
                    )
        dataset[location] = images

    return dataset


def saveEmbeddingsTGC():
    identifier = PersonIdentifier(embeddingGenerator=EmbeddingGenerator.abd)
    basePath = "/Users/adrianlorenzomelian/tfg/datasets/tgc/TGC2020v0.3"
    embBasePath = "/Users/adrianlorenzomelian/tfg/datasets/tgc/abd_embeddings_all"

    #ids = getIds("/Users/adrianlorenzomelian/tfg/datasets/tgc/TGC2020v0.3/BibsineveryCPs.txt")
    ids = [int(name) for name in os.listdir(basePath) if os.path.isdir(os.path.join(basePath, name))]
    print(len(ids))
    dataset = getTGCDataset(basePath, ids)

    for id in ids:
        idPath = "%s/%d" % (embBasePath, id)
        for index, location in enumerate(tgcLocations):
            path = "%s/%d" % (idPath, index)
            Path(path).mkdir(parents=True, exist_ok=True)
            embeddings = [identifier.getEmbeddingFromBody(image) for image in dataset[location][id]]
            for num, embedding in enumerate(embeddings):
                persistEmbedding("%s/%d.h5" % (path, num), embedding)


def loadEmbeddingsTGC(embPath, ids):
    dataset = {}
    for index, location in enumerate(tgcLocations):
        embeddings = {}
        for id in ids:
            embeddings[id] = []
            path = "%s/%d/%d" % (embPath, id, index)
            for filename in os.listdir(path):
                embeddings[id].append(
                    getEmbeddingFromDisk(
                        "%s/%s" % (path, filename)
                    )
                )
        dataset[location] = embeddings
    return dataset
