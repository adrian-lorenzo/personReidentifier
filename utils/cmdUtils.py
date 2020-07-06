from os import path

from models.gallery import Gallery
from services.personReidentifierService import PersonReidentifierService
from models.detector import Detector
from models.embeddingGenerator import EmbeddingGenerator
from utils.evaluationUtils import getIds, defaultWindowSize, defaultShiftStep, defaultShiftProp, defaultFps, \
    defaultTimeout, cmcTimeSpaceHeuristic, cmcTimeHeuristic, cmcSpaceHeuristic, cmc, plotCMC
from utils.persistanceUtils import loadEmbeddings

def checkPath(pathStr, name, isFile):
    if pathStr is None:
        raise ValueError("%s is not set." % (name))
    if not path.exists(pathStr):
        raise FileNotFoundError("%s is not found" % (name))
    if isFile is path.isdir(pathStr):
        raise ValueError("%s is not a %s" % (name, "file" if isFile else "directory"))


def choosePersonDetector(args):
    if args.mask:
        return Detector.mask
    elif args.yolo:
        return Detector.yolo
    elif args.faster:
        return Detector.faster
    elif args.mtcnn:
        return Detector.mtcnn

    return Detector.mask


def chooseBodyEmbeddingGenerator(args):
    if args.alignedreid:
        return EmbeddingGenerator.alignedReId
    elif args.abd:
        return EmbeddingGenerator.abd

    return EmbeddingGenerator.alignedReId


def getDetectionThreshold(args):
    if args.detectionThreshold is None:
        return 0.8
    else:
        return min(max(0, args.detectionThreshold), 1)


def buildPersonIdentifier(args):
    detector = choosePersonDetector(args)
    bodyEmbeddingGenerator = chooseBodyEmbeddingGenerator(args)

    if detector is not Detector.mtcnn:
        checkPath(
            args.bodyDetectorWeights,
            "Body Detector weights file",
            True
        )

        if detector is Detector.yolo:
            checkPath(
                args.bodyDetectorConfig,
                "Body detector configuration file",
                True
            )

        checkPath(
            args.embeddingGeneratorWeights,
            "Embedding generator weights file",
            True
        )

        if args.embeddingGeneratorClasses is None:
            args.embeddingGeneratorClasses = 1041 if bodyEmbeddingGenerator is EmbeddingGenerator.alignedReId else 751

    gallery = None

    if args.gallery is not None:
        checkPath(
            args.bodyDetectorWeights,
            "Gallery weights file",
            True
        )

        checkPath(
            args.ids,
            "Ids set file",
            True
        )

        ids = getIds(args.ids)
        gallery = loadEmbeddings(args.gallery, ids, args.galleryLocation, args.oneIndexed)
        gallery = { id: Gallery(embeddings) for id, embeddings in gallery.items() }

        if args.galleryLocation is None:
            raise ValueError("No query location specified. It should be specified with --galleryLocation option.")

    return PersonReidentifierService(
        detector=detector,
        detectionThreshold=getDetectionThreshold(args),
        embeddingGenerator=bodyEmbeddingGenerator,
        detectorWeights=args.bodyDetectorWeights,
        detectorConf=args.bodyDetectorConfig,
        embeddingGeneratorWeights=args.embeddingGeneratorWeights,
        embeddingGeneratorNumClasses=args.embeddingGeneratorClasses,
        gallery=gallery,
        databaseThreshold=args.openWorldThreshold,
        maxDescriptors=args.maxDescriptors
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

    if args.queryLocation is None:
        raise ValueError("No query location specified. It should be specified with --queryLocation option.")
    if args.galleryLocation is None:
        raise ValueError("No query location specified. It should be specified with --galleryLocation option.")

    ids = getIds(args.ids)
    query = loadEmbeddings(args.query, ids, args.queryLocation, args.oneIndexed)
    gallery = loadEmbeddings(args.gallery, ids, args.galleryLocation, args.oneIndexed)

    topNum = args.topNum if args.topNum else len(ids)
    results = None

    if args.timeHeuristic or args.spaceHeuristic:
        windowSize = defaultWindowSize if args.windowSize is None else args.windowSize
        shiftSize = defaultShiftStep if args.shiftSize is None else args.shiftSize
        shiftProp = defaultShiftProp if args.shiftProp is None else args.shiftProp
        fps = defaultFps if args.fps is None else args.fps
        timeout = defaultTimeout if args.timeout is None else args.timeout

        checkPath(
            args.raceRanking,
            "Previous race rank file",
            True
        )
        raceRanking = getIds(args.raceRanking)

        if args.timeHeuristic:
            if args.spaceHeuristic:
                _, results = cmcTimeSpaceHeuristic(gallery, query, topNum, raceRanking, windowSize, shiftProp,
                                                   shiftSize, fps,
                                                   timeout)
            else:
                _, results = cmcTimeHeuristic(gallery, query, topNum, raceRanking, windowSize, shiftProp, shiftSize)
        elif args.spaceHeuristic:
            _, results = cmcSpaceHeuristic(gallery, query, topNum, raceRanking, fps, timeout)
    else:
        _, results = cmc(gallery, query, topNum)

    if args.plot:
        plotCMC(results)