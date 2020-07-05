from os import path

import matplotlib.pyplot as plt
import numpy as np

from utils.debugUtils import printv, printDone

defaultWindowSize = 30
defaultShiftProp = .1
defaultShiftStep = 1
defaultFps = 5.
defaultTimeout = 10.


def plotCMC(result):
    """
    Plots the given CMC curve result.
    """
    printv("Plotting the CMC curve...", False)
    plt.figure(figsize=(9, 9))
    top = range(1, len(result)+1)
    plt.plot(top, result)
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.xticks(top)
    plt.xlabel("Ranks")
    plt.ylabel("Probability")
    plt.title("CMC curve")
    plt.grid()
    plt.show()
    printDone()


def rankIds(gallery, embedding, possibleIds):
    """
    Ranks a set of given ids of a given gallery by the mininum difference
    of their embedding samples with a given embedding.

    Arguments:
        gallery -- the gallery set of embeddings.
        embedding -- the selected embedding.
        possibleIds -- the possible ids to be choosen.

    Returns:
        The CMC curve in a list.
    """

    ranking = [
        [
            gallId,
            # The minimum value is obtained
            np.amin([
                # Distance difference of the selected embedding with
                # the gallery possible ones
                np.linalg.norm(embedding - gallEmbedding)
                for gallEmbedding in gallery[gallId]
            ])
        ]
        for gallId in possibleIds
    ]

    # Ordenamos
    ranking.sort(key=lambda item: item[1])
    return ranking


def cmc(gallery, query, topNum):
    """
    Calculates the CMC curve.

    Arguments:
        gallery -- the gallery set of embeddings.
        query -- the query set of embeddings.
        topNum -- the top size.

    Returns:
        The CMC curve in a list.
    """

    printv("Calculating CMC curve...")
    embeddingCount = 0
    matches = np.zeros(topNum)

    # For each query identity, we iterate
    # for each embedding.
    for id in query:
        for embedding in query[id]:
            # We obtain the ranking
            ranking = rankIds(gallery, embedding, gallery.keys())

            # Evaluates the given ranking
            for rank in range(0, topNum):
                if ranking[rank][0] == id:
                    matches[rank] += 1
                    break
            embeddingCount += 1

    # Obtains the accumulated probability
    results = np.cumsum(matches) / embeddingCount
    print(results)
    printDone()
    return matches, results


def cmcTimeHeuristic(gallery, query, topNum, raceRank, windowSize=defaultWindowSize, shiftProp=defaultShiftProp,
                     shiftStep=defaultShiftStep):
    """
    Calculates the CMC curve using time heuristic.

    Arguments:
        gallery -- the gallery set of embeddings.
        query -- the query set of embeddings.
        topNum -- the top size.
        raceRank -- the previous checkpoint rank.
        windowSize -- the size of the window for time heuristic.
        shiftProp -- the position difference proportion to consider shifting for time heuristic.
        shiftStep -- the size of the step for time heuristic.

    Returns:
        The CMC curve in a list.
    """

    printv("Calculating CMC curve with time heuristic...")
    embeddingCount = 0
    windowStart = 0

    shiftSize = shiftProp * windowSize

    matches = np.zeros(topNum)

    # For each query identity, we iterate
    # for each embedding.
    for id in query:
        for embedding in query[id]:
            # Obtains the ranking position that will be
            # used in this iteration
            windowPositions = list(range(
                max(windowStart - windowSize, 0),
                min(windowStart + windowSize + 1, len(raceRank))
            ))

            # Obtains the ids of each positions
            possibleIds = [raceRank[position] for position in windowPositions]

            # Obtains the ranking
            ranking = rankIds(gallery, embedding, possibleIds)

            # If the previous id rank was lower than the window
            # position plus the shift size, the window shifts
            if raceRank.index(ranking[0][0]) > (windowStart + shiftSize):
                windowStart += shiftStep

            # Evaluates the obtained rank
            for rank in range(0, topNum):
                if ranking[rank][0] == id:
                    matches[rank] += 1
                    break
            embeddingCount += 1

    # Obtains the accumulated probability
    results = np.cumsum(matches) / embeddingCount
    print(results)
    printDone()
    return matches, results


def cmcSpaceHeuristic(gallery, query, topNum, raceRank, fps=defaultFps, timeout=defaultTimeout):
    """
    Calculates the CMC curve using space heuristic.

    Arguments:
        gallery -- the gallery set of embeddings.
        query -- the query set of embeddings.
        topNum -- the top size.
        raceRank -- the previous checkpoint rank.
        fps -- the image feeder frame per second value for space heuristic.
        timeout -- the amount time elapsed to consider an identity as not possible for space heuristic.

    Returns:
        The CMC curve in a list.
    """

    printv("Calculating CMC curve with space heuristic...")
    embeddingCount = 0
    matches = np.zeros(topNum)
    ids = list(gallery.keys())

    seenDict = {}
    idCounter = 0

    # For each query identity, we iterate
    # for each embedding.
    for id in query:
        for embedding in query[id]:
            # Seen identities with 0-value timeout
            # get filtered.
            possibleIds = filter(
                lambda id: id not in list(seenDict.keys()) or seenDict[id] > 0.,
                ids
            )

            ranking = rankIds(gallery, embedding, possibleIds)

            if ranking[0][0] not in seenDict.keys():
                idCounter += 1

            # Winner timeout is reset
            seenDict[ranking[0][0]] = timeout + (
                    timeout * max(raceRank.index(ranking[0][0]) - idCounter, 0)
            )

            # Seen identities counter gets updated
            for seenId in seenDict.keys():
                if seenId != ranking[0][0]:
                    seenDict[seenId] = max(seenDict[seenId] - (1. / fps), 0)

            # Evaluates the obtained rank
            for rank in range(0, min(topNum, len(ranking))):
                if ranking[rank][0] == id:
                    matches[rank] += 1
                    break
            embeddingCount += 1

    # Obtains the accumulated probability
    results = np.cumsum(matches) / embeddingCount
    print(results)
    printDone()
    return matches, results


def cmcTimeSpaceHeuristic(gallery, query, topNum, raceRank,
                          windowSize=defaultWindowSize, shiftProp=defaultShiftProp, shiftStep=defaultShiftStep,
                          fps=defaultFps, timeout=defaultTimeout):
    """
    Calculates the CMC curve using time and space heuristic.

    Arguments:
        gallery -- the gallery set of embeddings.
        query -- the query set of embeddings.
        topNum -- the top size.
        raceRank -- the previous checkpoint rank.
        windowSize -- the size of the window for time heuristic.
        shiftProp -- the position difference proportion to consider shifting for time heuristic.
        shiftStep -- the size of the step for time heuristic.
        fps -- the image feeder frame per second value for space heuristic.
        timeout -- the amount time elapsed to consider an identity as not possible for space heuristic.

    Returns:
        The CMC curve in a list.
    """

    print("Calculating CMC curve with time and space heuristic...")
    embeddingCount = 0
    matches = np.zeros(topNum)

    windowStart = 0
    shiftSize = shiftProp * windowSize

    idCounter = 0
    seenDict = {}

    # For each query identity, we iterate
    # for each embedding.
    for id in query:
        for embedding in query[id]:
            # Obtains the ranking position that will be
            # used in this iteration
            windowPositions = list(range(
                windowStart,
                min(windowStart + windowSize, len(gallery))
            ))

            # Obtains the ids of each ranking positions, removing
            # the ones that have been seen with timeout 0
            possibleIds = set(
                [raceRank[position] for position in windowPositions]
            ).union(
                dict(filter(lambda item: item[1] > 0, seenDict.items())).keys()
            )

            ranking = rankIds(gallery, embedding, possibleIds)

            # If the previous id rank was lower than the window
            # position plus the shift size, the window shifts
            if ranking[0][0] not in seenDict.keys():
                if raceRank.index(ranking[0][0]) > (windowStart + shiftSize):
                    windowStart += shiftStep
                idCounter += 1

            # Winner timeout is reset
            seenDict[ranking[0][0]] = timeout + (
                    timeout * max(raceRank.index(ranking[0][0]) - idCounter, 0)
            )

            # Seen identities counter gets updated
            for i in seenDict.keys():
                if i != ranking[0][0] and seenDict[i] != 0:
                    seenDict[i] = max(seenDict[i] - 1 / fps, 0)

            # Evaluates the obtained rank
            for rank in range(0, min(topNum, len(ranking))):
                if ranking[rank][0] == id:
                    matches[rank] += 1
                    break
            embeddingCount += 1

    # Obtains the accumulated probability
    results = np.cumsum(matches) / embeddingCount
    print(results)
    printDone()
    return matches, results


def getIds(filePath):
    """
    Gets ids from file.
    The ids must be formatted as one per row.

    Arguments:
        fileName -- the file string with the ids.

    Returns:
        The id list.
    """

    if not path.exists(filePath):
        FileNotFoundError("Ids file not exists")

    with open(filePath) as file:
        return [int(line) for line in file]
