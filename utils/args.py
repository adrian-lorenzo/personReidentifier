import argparse


def buildArgumentParser():
    argparser = argparse.ArgumentParser()

    argparser.add_argument("video", type=str, help="Runs system with specified file video.")
    argparser.add_argument("camera", help="Runs system with system camera.")

    argparser.add_argument("embeddingGeneratorWeights", type=str, help="Embedding generator weights file.")
    argparser.add_argument("embeddingGeneratorClasses", type=int, help="Embedding generator number of classes.")

    argparser.add_argument("bodyDetectorWeights", type=str, help="Body detector weights file.")
    argparser.add_argument("bodyDetectorConfFile", type=str, help="Body detector configuration file.")

    argparser.add_argument("embedding", type=str, help="Returns embedding of the given body.")
    argparser.add_argument("detection", type=str,
                           help="Returns bodies of the given image in the specified folder using --saveFolder option.")
    argparser.add_argument("detectionThreshold", type=float, help="Specifies the detection threshold.")
    argparser.add_argument("saveFolder", type=str, help="Specifies the save folder for the --detection option.")

    argparser.add_argument("yolo", help="Uses YOLOv3 for person detection.")
    argparser.add_argument("faster", help="Uses FasterRCNN for person detection.")
    argparser.add_argument("mask", help="Uses MaskRCNN for person detection.")

    argparser.add_argument("alignedreid", help="Uses AlignedReId++ for body embedding generation.")
    argparser.add_argument("abd", help="Uses Attentive but Diverse Network for body embedding generation.")

    argparser.add_argument("cmc", help="Display the CMC curve of the given query and gallery.")
    argparser.add_argument("queryLocation", type=int, help="The choosen query location in the dataset.")
    argparser.add_argument("galleryLocation", type=int, help="The choosen gallery location in the dataset.")
    argparser.add_argument("oneIndexed", type=bool, help="Check this option if the dataset indexes starts at 1.")
    argparser.add_argument("topNum", type=int, help="Top number of CMC curve.")

    argparser.add_argument("timeHeuristic",
                           help="Display the CMC curve using time heuristic of the given query and gallery.")
    argparser.add_argument("raceRanking", type=str, help="Previous race ranking file for CMC curve for heuristics.")
    argparser.add_argument("windowSize", type=int, help="Window size for time heuristic.")
    argparser.add_argument("shiftSize", type=int, help="Shift size for time heuristic.")
    argparser.add_argument("shiftProp", type=float, help="Shift proportion for time heuristic.")

    argparser.add_argument("spaceHeuristic",
                           help="Display the CMC curve using space heuristic of the given query and gallery.")
    argparser.add_argument("fps", type=int, help="FPS for space heuristic.")
    argparser.add_argument("timeout", type=int, help="Timeout for time heuristic.")

    argparser.add_argument("query", type=str, help="The query embeddings directory.")
    argparser.add_argument("gallery", type=str, help="The gallery embeddings directory.")
    argparser.add_argument("ids", type=str, help="The ids list files.")

    return argparser
