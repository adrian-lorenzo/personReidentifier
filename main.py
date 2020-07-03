#!/usr/local/bin/python3

from os import path

from personIdentifier import PersonIdentifier
from utils.args import buildArgumentParser
from utils.detector import Detector
from utils.embeddingGenerator import EmbeddingGenerator
from utils.evaluationUtils import cmcTimeHeuristic, cmc, cmcSpaceHeuristic, cmcTimeSpaceHeuristic, getIds, \
    defaultWindowSize, defaultShiftStep, defaultShiftProp, defaultFps, defaultTimeout
from utils.persistanceUtils import loadImage, persistImage, loadEmbeddings


def checkPath(path, name, isFile):
    if not path:
        raise ValueError("%s is not set." % (name))
    if not path.exists(args.saveFolder):
        raise FileNotFoundError("%s is not found" % (name))
    if isFile is not path.isdir(args.saveFolder):
        raise ValueError("%s is not a %s" % (name, "file" if isFile else "directory"))


def choosePersonDetector(args):
    if args.maskrcnn:
        return Detector.mask
    elif args.yolo:
        return Detector.yolo
    elif args.faster:
        return Detector.faster

    return Detector.mask


def chooseBodyEmbeddingGenerator(args):
    if args.alignedreid:
        return EmbeddingGenerator.alignedReId
    elif args.abd:
        return EmbeddingGenerator.abd

    return EmbeddingGenerator.alignedReId


def getDetectionThreshold(args):
    return min(max(0, args.detectionThreshold), 1) if args.detectionThreshold else 0.8


def buildPersonIdentifier(args):
    return PersonIdentifier(
        detector=choosePersonDetector(args),
        embeddingGenerator=chooseBodyEmbeddingGenerator(args),
        detectionThreshold=getDetectionThreshold(args)
    )


def buildCmcViewer(args):
    checkPath(
        args.ids,
        "Ids set file",
        True
    )

    checkPath(
        args.query,
        "Query set folder",
        False
    )

    checkPath(
        args.gallery,
        "Gallery set folder",
        False
    )

    if not args.queryLocation:
        raise ValueError("No query location specified. It should be specified with --location option.")
    if not args.galleryLocation:
        raise ValueError("No query location specified. It should be specified with --location option.")

    ids = getIds(args.ids)
    query = loadEmbeddings(args.query, ids, args.queryLocation, args.oneIndexed)
    gallery = loadEmbeddings(args.query, ids, args.galleryLocation, args.oneIndexed)

    topNum = args.topNum if args.topNum else len(ids)

    if args.timeHeuristic or args.spaceHeuristic:
        windowSize = defaultWindowSize if not args.windowSize else args.windowSize
        shiftSize = defaultShiftStep if not args.shiftSize else args.shiftSize
        shiftProp = defaultShiftProp if not args.shiftProp else args.shiftProp
        fps = defaultFps if not args.fps else args.fps
        timeout = defaultTimeout if not args.timeout else args.timeout

        checkPath(
            args.raceRanking,
            "Previous race rank file",
            True
        )
        raceRanking = getIds(args.raceRanking)

        if args.timeHeuristic:
            if args.spaceHeuristic:
                cmcTimeSpaceHeuristic(gallery, query, topNum, raceRanking, windowSize, shiftProp, shiftSize, fps,
                                      timeout)
            else:
                cmcTimeHeuristic(gallery, query, topNum, raceRanking, windowSize, shiftProp, shiftSize)
        elif args.spaceHeuristic:
            cmcSpaceHeuristic(gallery, query, topNum, raceRanking, fps, timeout)
    else:
        cmc(gallery, query, topNum)


if __name__ == '__main__':
    args = buildArgumentParser().parse_args()
    identifier = buildPersonIdentifier(args)

    if args.video:
        identifier.startIdentificationByVideo(args.video)
    elif args.camera:
        identifier.startIdentificationByCam()
    elif args.embedding:
        checkPath(args.embedding, "Body image", True)
        body = loadImage(args.embedding)
        print(identifier.getEmbeddingFromBody(body))
    elif args.detection:
        checkPath(args.saveFolder, "Detection save folder", False)
        checkPath(args.detection, "Image", True)
        basename = path.basename(args.detection)
        image = loadImage(args.detection)
        persistImage(
            args.saveFolder,
            basename,
            identifier.detectBodyInFrame(image)
        )
    elif args.cmc:
        buildCmcViewer(args)
