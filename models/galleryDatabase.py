import numpy as np
from sklearn.cluster import DBSCAN

from models.gallery import Gallery


class GalleryDatabase():
    minSamplesToCluster = 10

    def __init__(self, database={}, identificationThreshold=1.5):
        self.database = database
        self.idCount = 0
        self.epsilon = identificationThreshold

    def addNewGallery(self, gallery):
        id = self.idCount
        self.database[id] = gallery
        self.idCount += 1
        return id

    def addToGallery(self, id, descriptor):
        self.database[id].addDescriptor(descriptor)

    def getAllGalleries(self):
        return self.database.values()

    def getAllDescriptors(self):
        return np.array([gallery.descriptors for gallery in self.getAllGalleries()]).flatten()

    def clearDatabase(self):
        self.database = {}

    def getIdentity(self, descriptor):
        for key, gallery in self.database.items():
            scan = DBSCAN(
                eps=self.epsilon,
                min_samples=min(
                    [max([1, len(gallery.descriptors)]), self.minSamplesToCluster])
            )
            if scan.fit_predict(np.vstack([gallery.descriptors, descriptor]))[-1] != -1:
                self.addToGallery(key, descriptor)
                return key
        return self.addNewGallery(Gallery(np.array([descriptor])))
