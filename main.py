#!/usr/local/bin/python3
import argparse

from personIdentifier import PersonIdentifier
from os import path

from utils.detector import Detector
from utils.persistanceUtils import loadImage

argparser = argparse.ArgumentParser()

argparser.add_argument("video", type=str, help="Runs system with specified file video")
argparser.add_argument("camera", help="Runs system with system camera")
argparser.add_argument("embedding", type=str, help="Returns embedding of the given body")
argparser.add_argument("detection", type=str, help="Returns bodies of the given image in the specified folder")

argparser.add_argument("cmc", help="Display the CMC of the given query and gallery")
argparser.add_argument("query", type=str, help="The query embeddings directory")
argparser.add_argument("gallery", type=str, help="The gallery embeddings directory")
argparser.add_argument("ids", type=str, help="The ids list files")

if __name__ == '__main__':
    # id = PersonIdentifier(detector=Detector.mask)
    # id.startIdentificationByCam()
    # AbdBodyEmbeddingGenerator()
    # saveEmbeddingsDisk()
    # saveEmbeddingsDiskIds([2, 3, 5, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 23, 25, 37])
    # saveEmbeddingsDiskIds([18, 19, 23, 25, 32, 37, 38])
    # saveEmbeddingsTGC()
    # saveCMC()

    id = PersonIdentifier(detector=Detector.yolo)
    id.startIdentificationByVideo("/Users/adrianlorenzomelian/tfg/race-videos/LPATRail_Meta_mp4/01132.mp4")

    # args = argparser.parse_args()
    #
    # identifier = PersonIdentifier()
    #
    # if args.video:
    #     identifier.startIdentificationByVideo(args.video)
    # elif args.camera:
    #     identifier.startIdentificationByCam()
    # elif args.embedding:
    #     if not path.exists(args.embedding):
    #         raise FileNotFoundError("Body image does not exist")
    #     body = loadImage(args.embedding)
    #     identifier.getEmbeddingFromBody(body)

