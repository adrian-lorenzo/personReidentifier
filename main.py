#!/usr/local/bin/python3
#from personIdentifier import PersonIdentifier

#from utils.detectors import Detector
from utils.evaluationUtils import cmcTest, cmcTest2, saveEmbeddingsDisk

if __name__ == '__main__':
    #id = PersonIdentifier(detector=Detector.mask)
    #id.startIdentificationByVideo("/Users/adrianlorenzomelian/meta.MTS")
    saveEmbeddingsDisk()