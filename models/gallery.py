import numpy as np


class Gallery():
    def __init__(self, descriptors=np.empty([0, 16384]), maxDescriptors=20):
        self.descriptors = descriptors
        self.maxDescriptors = maxDescriptors

    def addDescriptor(self, descriptor):
        self.descriptors = np.vstack((self.descriptors, descriptor))
        self.compensateGallery()

    def compensateGallery(self):
        if len(self.descriptors) > self.maxDescriptors:
            self.descriptors = self.descriptors[1:]
