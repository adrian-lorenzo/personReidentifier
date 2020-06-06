from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


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


def cmcIds(gallery, query, topNum):
    embeddingCount = 0
    matches = np.zeros(topNum)

    for id in query:
        for embedding in query[id]:
            distance = []
            for gallId in gallery:
                dists = []

                for gallEmbedding in gallery[gallId]:
                    dists.append(np.linalg.norm(embedding - gallEmbedding))

                distance.append(
                    [gallId, np.amin(dists)]
                )

            distance.sort(key=lambda v: v[1])

            for i in range(0, topNum):
                if distance[i][0] == id:
                    matches[i] += 1
                    break
            embeddingCount += 1
    results = np.cumsum(matches) / embeddingCount

    return matches, results

def cmc(gallery, query, topNum):
    embeddingCount = 0

    matches = np.zeros(topNum)

    # Seleccionamos una de las galerías de la segunda área
    for id, selectedGallery in enumerate(query):

        # Seleccionamos de la galería un embedding
        for index in range(len(selectedGallery)):
            gall = selectedGallery.copy()

            embedding = np.delete(gall, index, 0)

            distance = []
            # Se itera por las galerías de la primera área
            for gallId in range(len(gallery)):
                dists = []

                # En este caso no hay que filtrar, ya que los embeddings siempre son diferentes.
                for gallEmbedding in gallery[gallId]:
                    # Calculamos la distancia entre el embedding de la consulta y el seleccionado.
                    dists.append(np.linalg.norm(embedding - gallEmbedding))

                # De cada identidad, seleccionamos el embedding de menor distancia
                distance.append(
                    [gallId, np.amin(dists)]
                )

            # Ordenamos las distancias de menor a mayor
            distance.sort(key=lambda v: v[1])

            # Iteramos en el ranking
            for i in range(0, topNum):
                # Si las dos identidades coinciden, ese es el rank obtenido. Aumentamos el contador.
                if distance[i][0] == id:
                    matches[i] += 1
                    break
            embeddingCount += 1
    results = np.cumsum(matches) / embeddingCount
    return matches, results

def cmcOneGallery(gallery, topNum, windowEnd=10):
    embeddingCount = 0
    windowStart = 0

    matches = np.zeros(topNum)
    for id, selectedGallery in enumerate(gallery):

        for index in range(len(selectedGallery)):
            gall = selectedGallery.copy()

            embedding = np.delete(gall, index, 0)

            distance = []

            for gallId in range(windowStart, min(windowStart + windowEnd, len(gallery))):
                dists = []

                for i, gallEmbedding in enumerate(gallery[gallId]):
                    if i != index or gallId != id:
                        dists.append(np.linalg.norm(embedding - gallEmbedding))

                distance.append(
                    [gallId, np.amin(dists)]
                )

            distance.sort(key=lambda v: v[1])

            for i in range(0, topNum):
                if distance[i][0] == id:
                    if distance[i][0] > (windowStart + 3):
                        windowStart += 1
                    matches[i] += 1
                    break
            embeddingCount += 1

    results = np.cumsum(matches) / embeddingCount
    return matches, results


def cmcTimeHeuristic(gallery, query, topNum, windowEnd=15):
    embeddingCount = 0
    windowStart = 0

    matches = np.zeros(topNum)

    # Seleccionamos una de las galerías de la segunda área
    for id, selectedGallery in enumerate(query):

        # Seleccionamos de la galería un embedding
        for index in range(len(selectedGallery)):
            gall = selectedGallery.copy()

            embedding = np.delete(gall, index, 0)

            distance = []
            # Se itera por las galerías de la primera área
            for gallId in range(windowStart, min(windowStart + windowEnd, len(gallery))):
                dists = []

                # En este caso no hay que filtrar, ya que los embeddings siempre son diferentes.
                for i, gallEmbedding in enumerate(gallery[gallId]):
                    # Calculamos la distancia entre el embedding de la consulta y el seleccionado.
                    dists.append(np.linalg.norm(embedding - gallEmbedding))

                # De cada identidad, seleccionamos el embedding de menor distancia
                distance.append(
                    [gallId, np.amin(dists)]
                )

            # Ordenamos las distancias de menor a mayor
            distance.sort(key=lambda v: v[1])

            # Iteramos en el ranking
            for i in range(0, topNum):
                # Si las dos identidades coinciden, ese es el rank obtenido. Aumentamos el contador.
                if distance[i][0] == id:
                    if distance[i][0] > (windowStart + 3):
                        windowStart += 1
                    matches[i] += 1
                    break
            embeddingCount += 1
    results = np.cumsum(matches) / embeddingCount
    return matches, results


def cmcTimespatialHeuristic(gallery, query, topNum, windowSize=12, fps=4):
    embeddingCount = 0
    seenDict = {}

    matches = np.zeros(topNum)

    identities = list(range(len(gallery)))
    seen = []

    # Seleccionamos una de las galerías de la segunda área
    for id, selectedGallery in enumerate(query):

        # Seleccionamos de la galería un embedding
        for index in range(len(selectedGallery)):
            gall = selectedGallery.copy()

            embedding = np.delete(gall, index, 0)

            distance = []

            # REFERENCIADOS
            for gallId in seen:
                if seenDict[gallId] <= 0:
                    continue

                dists = []

                # En este caso no hay que filtrar, ya que los embeddings siempre son diferentes.
                for i, gallEmbedding in enumerate(gallery[gallId]):
                    # Calculamos la distancia entre el embedding de la consulta y el seleccionado.
                    dists.append(np.linalg.norm(embedding - gallEmbedding))

                # De cada identidad, seleccionamos el embedding de menor distancia
                distance.append(
                    [gallId, np.amin(dists)]
                )

            window = 0

            for gallId in identities:
                dists = []

                # En este caso no hay que filtrar, ya que los embeddings siempre son diferentes.
                for i, gallEmbedding in enumerate(gallery[gallId]):
                    # Calculamos la distancia entre el embedding de la consulta y el seleccionado.
                    dists.append(np.linalg.norm(embedding - gallEmbedding))

                # De cada identidad, seleccionamos el embedding de menor distancia
                distance.append(
                    [gallId, np.amin(dists)]
                )

                window += 1

                if window >= windowSize:
                    break

            # Ordenamos las distancias de menor a mayor
            distance.sort(key=lambda v: v[1])

            if distance[0][0] in seen:
                seenDict[distance[0][0]] = seenDict[distance[0][0]] + 10
            else:
                seenDict[distance[0][0]] = fps * 4
                identities.remove(distance[0][0])
                seen.append(distance[0][0])

            for i in seen:
                if i != distance[0][0]:
                    seenDict[i] = max(seenDict[i] - 1, 0)

            # Iteramos en el ranking
            for i in range(0, min(topNum, len(distance))):
                # Si las dos identidades coinciden, ese es el rank obtenido. Aumentamos el contador.
                if distance[i][0] == id:
                    matches[i] += 1
                    break
            embeddingCount += 1
    results = np.cumsum(matches) / embeddingCount

    return matches, results


def cmcReranking(gallery, query, topNum, k1=5, k2=3, lambda_value=0.3):
    final_dist = rerank(gallery, query, k1, k2, lambda_value)

    ids = np.array([[i for _ in range(4)] for i in range(400)]).flatten()
    embeddingCount = 0
    matches = np.zeros(topNum)

    for id, distances in zip(ids, final_dist):
        dist = []
        for id2, value in zip(ids, distances):
            dist.append([id2, value])
        dist.sort(key=lambda v: v[1])
        for i in range(0, topNum):
            if dist[i][0] == id:
                matches[i] += 1
                break
        embeddingCount += 1
    results = np.cumsum(matches) / embeddingCount

    return matches, results


def rerank(gallery, query, k1, k2, lambda_value):
    gallery = np.array(gallery)
    gallery = gallery.reshape(gallery.shape[0] * gallery.shape[1], -1)
    query = np.array(query)
    query = query.reshape(query.shape[0] * query.shape[1], -1)
    query_num = query.shape[0]
    all_num = query_num + gallery.shape[0]
    feat = np.append(query, gallery, axis=0)
    feat = feat.astype(np.float16)

    original_dist = np.zeros(shape=[all_num, all_num], dtype=np.float16)
    original_dist = cdist(feat, feat).astype(np.float16)
    original_dist = np.power(original_dist, 2).astype(np.float16)
    del feat

    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def getIds(fileName):
    if not path.exists(fileName):
        FileNotFoundError("Ids file not exists")

    with open(fileName) as file:
        return [int(line) for line in file]


def plotAllLocationsCMC(alignedReIdDataset, abdDataset, galleryLocation, locations, saveLocation):
    numIds = len(list(alignedReIdDataset[galleryLocation].keys()))
    for location in locations:
        if galleryLocation != location:
            alignedReIdCmc = cmcIds(alignedReIdDataset[galleryLocation], alignedReIdDataset[location], numIds)[1]
            abdCmc = cmcIds(abdDataset[galleryLocation], abdDataset[location], numIds)[1]

            ranks = np.arange(0, numIds, 1)
            plt.figure(figsize=(9, 9))
            plt.plot(ranks, alignedReIdCmc)
            plt.plot(ranks, abdCmc)

            plt.yticks(np.arange(0, 1.05, 0.05))
            plt.xticks(np.arange(0, numIds, 5))
            plt.xlabel("Ranks")
            plt.ylabel("Accuracy")
            plt.title("CMC curve - %s as gallery, %s as query" % (galleryLocation, location))
            plt.legend(['AlignedReId', 'ABD'])
            plt.grid()
            plt.savefig("%s/%s/cmc_%s_%s.png" % (saveLocation, galleryLocation, galleryLocation, location))