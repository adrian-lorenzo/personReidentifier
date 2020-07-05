#!/bin/zsh

python3 personReidentifier.py --yolo --abd \
    --detection /Users/adrianlorenzomelian/Desktop/adrian/photos/photo2.jpeg \
    --bodyDetectorWeights ../pretrained_models/yolov3/yolov3.weights \
    --bodyDetectorConf ../pretrained_models/yolov3/yolov3.cfg \
    --embeddingGeneratorWeights ../pretrained_models/abd/market_checkpoint_best.pth.tar \
    --saveFolder /Users/adrianlorenzomelian/Desktop/