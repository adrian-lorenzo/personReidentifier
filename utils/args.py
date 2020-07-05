import argparse


def buildArgumentParser():
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--silent", action='store_false', help="Runs program in silent mode (no verbose).")

    argparser.add_argument("--embedding", type=str, metavar="image_file",
                           help="Returns embedding of the given body. Default embedding generator: AlignedReId++")
    argparser.add_argument("--alignedreid", action='store_true',
                           help="Uses AlignedReId++ for body embedding generation.")
    argparser.add_argument("--abd", action='store_true',
                           help="Uses Attentive but Diverse Network for body embedding generation.")
    argparser.add_argument("--embeddingGeneratorWeights", type=str, metavar="weights_file",
                           help="Embedding generator weights file.")
    argparser.add_argument("--embeddingGeneratorClasses", type=int, metavar="classes_number",
                           help="Embedding generator number of classes.")
    argparser.add_argument("--savePath", type=str, metavar="save_embedding_path", help="Embedding save path (use .h5 extension)")

    argparser.add_argument("--detection", type=str, metavar="image_path",
                           help="Returns bodies of the given image in the specified folder using --saveFolder option. Default body detector: MaskRCNN")
    argparser.add_argument("--mtcnn", action='store_true', help="Uses MTCNN for face detection.")
    argparser.add_argument("--yolo", action='store_true', help="Uses YOLOv3 for person detection.")
    argparser.add_argument("--faster", action='store_true', help="Uses FasterRCNN for person detection.")
    argparser.add_argument("--mask", action='store_true', help="Uses MaskRCNN for person detection.")
    argparser.add_argument("--bodyDetectorWeights", type=str, metavar="weights_path",
                           help="Body detector weights file.")
    argparser.add_argument("--bodyDetectorConfig", type=str, metavar="configuration_file_path",
                           help="Body detector configuration file. Only needed for YOLOv3.")
    argparser.add_argument("--detectionThreshold", type=float, metavar="threshold_float",
                           help="Specifies the detection threshold.")
    argparser.add_argument("--boundingBox", action='store_true', help="Draws bounding boxes in image")
    argparser.add_argument("--saveFolder", type=str, metavar="folder_path",
                           help="Specifies the save folder for the --detection option.")

    argparser.add_argument("--cmc", action='store_true',
                           help="Returns the CMC curve discrete values of the given query and gallery.")
    argparser.add_argument("--plot", action='store_true', help="Plots the calculated CMC curve.")
    argparser.add_argument("--query", type=str, metavar="query_directory_path", help="The query embeddings directory.")
    argparser.add_argument("--gallery", type=str, metavar="gallery_directory_path",
                           help="The gallery embeddings directory.")
    argparser.add_argument("--ids", type=str, metavar="ids_file_path", help="The ids list files.")
    argparser.add_argument("--queryLocation", type=int, metavar="location_number",
                           help="The choosen query location in the dataset.")
    argparser.add_argument("--galleryLocation", type=int, metavar="location_number",
                           help="The choosen gallery location in the dataset.")
    argparser.add_argument("--oneIndexed", action='store_true',
                           help="Check this option if the dataset indexes starts at 1.")
    argparser.add_argument("--topNum", type=int, metavar="top_number", help="Top number of CMC curve.")

    argparser.add_argument("--timeHeuristic", action='store_true',
                           help="Display the CMC curve using time heuristic of the given query and gallery.")
    argparser.add_argument("--raceRanking", type=str, metavar="rank_file_path",
                           help="Previous race ranking file for CMC curve for heuristics.")
    argparser.add_argument("--windowSize", type=int, metavar="integer", help="Window size for time heuristic.")
    argparser.add_argument("--shiftSize", type=int, metavar="integer", help="Shift size for time heuristic.")
    argparser.add_argument("--shiftProp", type=float, metavar="float", help="Shift proportion for time heuristic.")

    argparser.add_argument("--spaceHeuristic", action='store_true',
                           help="Display the CMC curve using space heuristic of the given query and gallery.")
    argparser.add_argument("--fps", type=int, metavar="integer", help="FPS for space heuristic.")
    argparser.add_argument("--timeout", type=int, metavar="float", help="Timeout for time heuristic.")

    argparser.add_argument("--video", type=str, help="Runs system with specified file video.")
    argparser.add_argument("--camera", action='store_true', help="Runs system with system camera.")
    argparser.add_argument("--openWorldThreshold", type=float, metavar="float",
                           help="The minimum value to consider an identity as equal as one in the gallery, used in the open world assumption.")
    argparser.add_argument("--maxDescriptors", type=int, metavar="integer",
                           help="The maximum number of descriptors per gallery.")

    return argparser
