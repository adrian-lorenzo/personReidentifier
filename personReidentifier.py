#!/usr/local/bin/python3
import time
from os import path

from utils.args import buildArgumentParser
from utils.cmdUtils import buildCmcViewer, buildPersonIdentifier, checkPath
from utils.debugUtils import printv
from utils.persistanceUtils import loadImage, persistImage, persistEmbedding

startTime = time.time()
if __name__ == '__main__':
    args = buildArgumentParser().parse_args()
    verbose = args.silent

    if args.cmc:
        buildCmcViewer(args)
    else:
        identifier = buildPersonIdentifier(args)
        if args.video is not None:
            identifier.startIdentificationByVideo(args.video)
        elif args.camera:
            identifier.startIdentificationByCam()
        elif args.embedding is not None:
            checkPath(args.embedding, "Body image", True)
            if args.mtcnn:
                raise ValueError(
                    "You can't generate an embedding using MTCNN. Please, use a body detector and a body embedding generator.")
            body = loadImage(args.embedding)
            embedding = identifier.getEmbeddingFromBody(body)
            if args.savePath is None:
                raise FileNotFoundError("Save path is missing. You can set it using --savePath option.")
            printv("Saving embedding at %s" % (args.savePath))
            persistEmbedding(args.savePath, embedding)
        elif args.detection is not None:
            checkPath(args.saveFolder, "Detection save folder", False)
            checkPath(args.detection, "Image", True)
            basename = path.splitext(path.basename(args.detection))[0]
            image = loadImage(args.detection)
            if args.boundingBox:
                detect = identifier.detectFacesInFrame if args.mtcnn else identifier.detectBodiesInFrame
                persistImage(
                    args.saveFolder,
                    "%s.png" % (basename),
                    detect(image)
                )
            else:
                detect = identifier.getFacesFromImage if args.mtcnn else identifier.getBodiesFromImage
                images = detect(image)
                for idx, img in enumerate(images):
                    persistImage(
                        args.saveFolder,
                        "%s-%d.png" % (basename, idx),
                        img
                    )

    printv("Task finished 🚀. Time elapsed: %s seconds." % (time.time() - startTime))
