#!/bin/zsh

# Save detection of specified image using YOLOv3
./personReidentifier.py --yolo --abd \
    --detection ../../report/images/detection/deint_frames/8.png \
    --bodyDetectorWeights ../pretrained_models/yolov3/yolov3.weights \
    --bodyDetectorConf ../pretrained_models/yolov3/yolov3.cfg \
    --embeddingGeneratorWeights ../pretrained_models/abd/market_checkpoint_best.pth.tar \
    --saveFolder ../../ \
    --detectionThreshold 0.5

: '
# Calculates embedding of specified image using ABD-Net.
./personReidentifier.py --yolo --abd \
    --embedding ../../8-2.png \
    --bodyDetectorWeights ../pretrained_models/yolov3/yolov3.weights \
    --bodyDetectorConf ../pretrained_models/yolov3/yolov3.cfg \
    --embeddingGeneratorWeights ../pretrained_models/abd/market_checkpoint_best.pth.tar \
    --savePath ../../8-2.h5


# Computes CMC curve of given gallery and query sets using time and space heuristic with default values.
./personReidentifier.py --cmc --timeHeuristic --spaceHeuristic \
    --gallery ../../datasets/lpatrail/deint_embeddings \
    --galleryLocation 1 \
    --query ../../datasets/lpatrail/deint_embeddings \
    --queryLocation 2 \
    --ids ../../datasets/lpatrail/deint_embeddings/ids.txt \
    --raceRanking ../../datasets/lpatrail/deint_embeddings/ids.txt \
    --topNum 30 \
    --plot


# Runs the system using YOLOv3 and AlignedReId++ with the given gallery.
./personReidentifier.py --camera --yolo --alignedreid \
    --bodyDetectorWeights ../pretrained_models/yolov3/yolov3.weights \
    --bodyDetectorConf ../pretrained_models/yolov3/yolov3.cfg \
    --embeddingGeneratorWeights ../pretrained_models/alignedreid/checkpoint_ep500.pth.tar \
    --embeddingGeneratorClasses 100 \
    --ids ../../datasets/lpatrail/deint_embeddings/ids.txt \
    --gallery ../../datasets/lpatrail/deint_embeddings \
    --galleryLocation 1
'