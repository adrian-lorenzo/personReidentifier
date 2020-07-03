from os import path

import matplotlib.pyplot as plt
import numpy as np

defaultWindowSize = 20
defaultShiftProp = .3
defaultShiftStep = 1
defaultFps = 3
defaultTimeout = 10


def plotCMC(result, ids):
    plt.figure(figsize=(9, 9))
    plt.plot(ids, result)
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.xticks(ids)
    plt.xlabel("Ranks")
    plt.ylabel("Probability")
    plt.title("CMC curve")
    plt.grid()
    plt.show()


def rankIds(gallery, embedding, possibleIds):
    ranking = [
        [
            gallId,
            # Se obtiene el valor mínimo
            np.amin([
                # Diferencia de distancias del embedding
                # seleccionado con los posibles de la galería
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
    embeddingCount = 0
    matches = np.zeros(topNum)

    # Por cada identidad de la consulta,
    # iteramos por cada espacio embebido
    for id in query:
        for embedding in query[id]:
            # Obtenemos el ranking
            ranking = rankIds(gallery, embedding, gallery.keys())

            # Evaluamos la clasificación obtenida
            for rank in range(0, topNum):
                if ranking[rank][0] == id:
                    matches[rank] += 1
                    break
            embeddingCount += 1

    # Obtenemos la precisión acumulada
    results = np.cumsum(matches) / embeddingCount
    print(results)
    print(matches)
    return matches, results


def cmcTimeHeuristic(gallery, query, topNum, raceRank, windowSize=defaultWindowSize, shiftProp=defaultShiftProp,
                     shiftStep=defaultShiftStep):
    embeddingCount = 0
    windowStart = 0

    shiftSize = shiftProp * windowSize

    matches = np.zeros(topNum)

    # Por cada identidad de la consulta,
    # iteramos por cada espacio embebido
    for id in query:
        for embedding in query[id]:
            # Obtenemos las posiciones del ranking que
            # se utilizarán en esta iteración
            windowPositions = list(range(
                max(windowStart - windowSize, 0),
                min(windowStart + windowSize + 1, len(raceRank))
            ))

            # Obtenemos las ids de esas posiciones
            possibleIds = [raceRank[position] for position in windowPositions]

            # Obtenemos el ranking
            ranking = rankIds(gallery, embedding, possibleIds)

            # Si la clasificación anterior era mayor a la posición
            # de la ventana más el tamaño para desplazar,
            # se desplaza la ventana
            if raceRank.index(ranking[0][0]) > (windowStart + shiftSize):
                windowStart += shiftStep

            # Evaluamos la clasificación obtenida
            for rank in range(0, topNum):
                if ranking[rank][0] == id:
                    matches[rank] += 1
                    break
            embeddingCount += 1

    # Obtenemos la precisión acumulada
    results = np.cumsum(matches) / embeddingCount
    print(results)
    print(matches)
    return matches, results


def cmcSpaceHeuristic(gallery, query, topNum, raceRank, fps=defaultFps, timeout=defaultTimeout):
    embeddingCount = 0
    matches = np.zeros(topNum)
    ids = list(gallery.keys())

    seenDict = {}
    idCounter = 0

    # Por cada identidad de la consulta,
    # iteramos por cada espacio embebido
    for id in query:
        for embedding in query[id]:
            # Se filtran aquellas identidades ya vistas
            # con timeout a 0.
            possibleIds = filter(
                lambda id: id not in list(seenDict.keys()) or seenDict[id] > 0.,
                ids
            )

            ranking = rankIds(gallery, embedding, possibleIds)

            # Ordenamos las distancias de menor a mayor
            ranking.sort(key=lambda v: v[1])

            if ranking[0][0] not in seenDict.keys():
                idCounter += 1

            # Se restablece la cuenta atrás del ganador
            seenDict[ranking[0][0]] = timeout + (
                    timeout * max(raceRank.index(ranking[0][0]) - idCounter, 0)
            )

            # Se reduce el contador de los ya vistos
            for seenId in seenDict.keys():
                if seenId != ranking[0][0]:
                    seenDict[seenId] = max(seenDict[seenId] - (1. / fps), 0)

            # Evaluamos la clasificación obtenida
            for rank in range(0, min(topNum, len(ranking))):
                if ranking[rank][0] == id:
                    matches[rank] += 1
                    break
            embeddingCount += 1

    # Obtenemos la precisión acumulada
    results = np.cumsum(matches) / embeddingCount
    print(results)
    print(matches)
    return matches, results


def cmcTimeSpaceHeuristic(gallery, query, topNum, raceRank,
                          windowSize=defaultWindowSize, shiftProp=defaultShiftProp, shiftStep=defaultShiftStep,
                          fps=defaultFps, timeout=defaultTimeout):
    embeddingCount = 0
    matches = np.zeros(topNum)

    windowStart = 0
    shiftSize = shiftProp * windowSize

    idCounter = 0
    seenDict = {}

    # Por cada identidad de la consulta,
    # iteramos por cada espacio embebido
    for id in query:
        for embedding in query[id]:
            # Obtenemos las posiciones del ranking que
            # se utilizarán en esta iteración
            windowPositions = list(range(
                windowStart,
                min(windowStart + windowSize, len(gallery))
            ))

            # Se añaden las posiciones de la clasificación, eliminando
            # las que ya han aparecido con cuenta atrás finalizada
            possibleIds = set(
                [raceRank[position] for position in windowPositions]
            ).union(
                dict(filter(lambda item: item[1] > 0, seenDict.items())).keys()
            )

            ranking = rankIds(gallery, embedding, possibleIds)

            # Ordenamos las distancias de menor a mayor
            ranking.sort(key=lambda v: v[1])

            # Si la clasificación anterior era mayor a la posición
            # de la ventana más el tamaño para desplazar,
            # se desplaza la ventana
            if ranking[0][0] not in seenDict.keys():
                if raceRank.index(ranking[0][0]) > (windowStart + shiftSize):
                    windowStart += shiftStep
                idCounter += 1

            # Se restablece la cuenta atrás del ganador
            seenDict[ranking[0][0]] = timeout + (
                    timeout * max(raceRank.index(ranking[0][0]) - idCounter, 0)
            )

            # Se reduce el contador de los ya vistos
            for i in seenDict.keys():
                if i != ranking[0][0] and seenDict[i] != 0:
                    seenDict[i] = max(seenDict[i] - 1 / fps, 0)

            # Evaluamos la clasificación obtenida
            for rank in range(0, min(topNum, len(ranking))):
                if ranking[rank][0] == id:
                    matches[rank] += 1
                    break
            embeddingCount += 1

    # Obtenemos la precisión acumulada
    results = np.cumsum(matches) / embeddingCount
    print(results)
    print(matches)
    return matches, results


def getIds(fileName):
    if not path.exists(fileName):
        FileNotFoundError("Ids file not exists")

    with open(fileName) as file:
        return [int(line) for line in file]
