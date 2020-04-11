import random as rand
import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN
from pathlib import Path

import matplotlib.pyplot as plt

from personIdentifier import PersonIdentifier
from utils.persistanceUtils import persistEmbedding, getEmbeddingFromDisk


def getSampleImages(personId=1, area=1, basePath="/Users/adrianlorenzomelian/dataset", numSamples=4):
    return np.array([
        cv.cvtColor(cv.imread("%s/%d/%d/%d.png" % (basePath, personId, area, i), 1), cv.COLOR_BGR2RGB)
        for i in range(1, numSamples + 1)
    ])

def getBodies(identifier, images):
    return [identifier.bodyDetector.getBoundingBoxes(image)[1][0]
                .getImageFromBox(image) for image in images]

def getEmbeddingsFromImages(identifier, bodies):
    return np.array([
        identifier.bodyEmbeddingGenerator.getEmbedding(body)
        for body in bodies
    ])

def getDistanceClustering(firstArea, secondArea, epsilon = 1.8):
    scan = DBSCAN(
        eps=epsilon,
        min_samples=4
    )
    array = scan.fit_predict(np.concatenate((firstArea, secondArea), axis = 0))
    print(array)
    return array

def persistEmbeddingsToDisk(embeddings, personId=1, area=1, basePath="/Users/adrianlorenzomelian/embeddings"):
    path = "%s/%d/%d" % (basePath, personId, area)
    Path(path).mkdir(parents=True, exist_ok=True)
    for index, embedding in enumerate(embeddings):
        persistEmbedding("%s/%d.h5" % (path, index+1), embedding)

def persistImagesToDisk(images, personId=1, area=1, basePath="/Users/adrianlorenzomelian/bodies"):
    path = "%s/%d/%d" % (basePath, personId, area)
    Path(path).mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        cv.imwrite("%s/%d.png" % (path, index+1), image)

def getEmbeddingsFromDisk(personId=1, area=1, basePath="/Users/adrianlorenzomelian/embeddings", numSamples=4):
    return np.array([
        getEmbeddingFromDisk(
            "%s/%d/%d/%d.h5" % (basePath, personId, area, i)
        )
        for i in range(1, numSamples + 1)
    ])

def cmc(galleries):
    embeddingCount = 0

    matches = np.zeros(len(galleries))
    for id, selectedGallery in enumerate(galleries):
        for index in range(len(selectedGallery)):
            gall = selectedGallery.copy()
            embedding = np.delete(gall, index, 0)

            distance = []
            for gallId, gallery in enumerate(galleries):
                dists = []
                for i, gallEmbedding in enumerate(gallery):
                    if i != index or gallId != id:
                        dists.append(np.linalg.norm(embedding - gallEmbedding))
                distance.append(
                    [gallId, np.amin(dists)]
                )

            distance.sort(key = lambda v: v[1])
            # Zero if no match 1 if match
            for i in range(0, len(distance)):
                if distance[i][0] == id:
                    matches[i] += 1
            embeddingCount += 1
    results = np.cumsum(matches) / embeddingCount
    return matches, results

def cmc2(galleries, galleries2):
    embeddingCount = 0

    matches = np.zeros(len(galleries))
    for id, selectedGallery in enumerate(galleries2):
        for index in range(len(selectedGallery)):
            gall = selectedGallery.copy()
            embedding = np.delete(gall, index, 0)

            distance = []
            for gallId, gallery in enumerate(galleries):
                dists = []
                for i, gallEmbedding in enumerate(gallery):
                        dists.append(np.linalg.norm(embedding - gallEmbedding))
                print(dists)
                distance.append(
                    [gallId, np.amin(dists)]
                )

            distance.sort(key = lambda v: v[1])
            print(distance)
            # Zero if no match 1 if match
            for i in range(0, len(distance)):
                if distance[i][0] == id:
                    print(distance[i][0], id, distance[i][1])
                    matches[i] += 1
            embeddingCount += 1
    results = np.cumsum(matches) / embeddingCount
    return matches, results

def plotCMC(result, ids):
    plt.plot(ids, result)
    plt.yticks(np.arange(0, 1.20, 0.20))
    plt.xticks(ids)
    plt.xlabel("Ranks")
    plt.ylabel("Probability")
    plt.title("CMC curve")
    plt.show()

def cmcTest():
    galleries = [getEmbeddingsFromDisk(personId=i, area=1) for i in range(9, 23)]
    ids = list(range(len(galleries)))
    plotCMC(cmc(galleries)[1], ids)


def cmcTest2():
    bodiesFirstArea = []
    bodiesSecondArea = []

    for i in range(9, 19):
        bodiesFirstArea.append(getEmbeddingsFromDisk(personId=i))
        bodiesSecondArea.append(getEmbeddingsFromDisk(personId=i, area=2))

    plotCMC(cmc2(bodiesFirstArea, bodiesSecondArea)[1], list(range(0, 10)))

def saveEmbeddingsDisk():
    identifier = PersonIdentifier()
    for i in range(100, 101):
        bodies1 = getBodies(identifier, getSampleImages(personId=i, area=1))
        persistImagesToDisk(bodies1, personId=i, area=1)
        persistEmbeddingsToDisk(getEmbeddingsFromImages(identifier, bodies1), personId=i, area=1)

        bodies2 = getBodies(identifier, getSampleImages(personId=i, area=2))
        persistImagesToDisk(bodies2, personId=i, area=2)
        persistEmbeddingsToDisk(getEmbeddingsFromImages(identifier, bodies2), personId=i,area=2)

def evaluateDifferentIdentityPairs(epsilon = 1.8):
    indexes = list(range(9, 23))

    numClustered = []
    while len(indexes) > 1:
        firstId = indexes.pop(rand.randint(0, len(indexes)-1))
        secondId = indexes.pop(rand.randint(0, len(indexes)-1))
        result = getDistanceClustering(
            getEmbeddingsFromDisk(personId=firstId, area=1),
            getEmbeddingsFromDisk(personId=secondId, area=1),
            epsilon=epsilon
        )
        numClustered.append((
            np.count_nonzero(result == 0),
            np.count_nonzero(result == 1)
        ))

    return numClustered, 8