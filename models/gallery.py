class Gallery:
    def __init__(self, bodies: list, maxDescriptors: int = None):
        self.descriptors = bodies
        self.maxDescriptors = maxDescriptors

    def addDescriptor(self, descriptor):
        self.descriptors.append(descriptor)
        self.compensateGallery()

    def compensateGallery(self):
        if self.maxDescriptors is not None and len(self.descriptors) > self.maxDescriptors:
            self.descriptors = self.descriptors[1:]
