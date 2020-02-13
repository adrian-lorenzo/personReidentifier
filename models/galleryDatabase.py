import numpy as np


class GalleryDatabase():
    def __init__(self, database={}):
        self.database = database
        self.idCount = 0

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
