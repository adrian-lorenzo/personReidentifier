import numpy as np

from models.gallery import Gallery


class GalleryDatabase():
    def __init__(self, database={}, maxDescriptors=None, threshold=None):
        self.database = database
        self.maxDescriptors = maxDescriptors
        self.threshold = threshold
        self.count = 0

    def addNewGallery(self, gallery):
        self.database[self.count] = gallery
        self.count += 1
        return self.count

    def addToGallery(self, id, descriptor):
        self.database[id].addDescriptor(descriptor)

    def getAllGalleries(self):
        return self.database.values()

    def getAllIds(self):
        return list(self.database.keys())

    def clearDatabase(self):
        self.database = {}

    def getIdentity(self, descriptor):
        if descriptor is None: return -1
        distance = []
        for gallId in self.database:
            dists = []

            for gallEmbedding in self.database[gallId].descriptors:
                dists.append(np.linalg.norm(descriptor - gallEmbedding))

            distance.append(
                [gallId, np.amin(dists)]
            )

        if not distance:
            return self.addNewGallery(Gallery([descriptor], maxDescriptors=self.maxDescriptors))

        bestId, minDistance = min(distance, key=lambda v: v[1])

        if self.threshold is None or minDistance < self.threshold:
            self.addToGallery(bestId, descriptor)
            return bestId
        else:
            return self.addNewGallery(Gallery([descriptor], maxDescriptors=self.maxDescriptors))
