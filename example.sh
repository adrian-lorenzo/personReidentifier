#!/bin/zsh
./personReidentifier.py --yolo --abd \
    --detection /Users/adrianlorenzomelian/Downloads/test_image.jpeg \
    --bodyDetectorWeights ../pretrained_models/yolov3/yolov3.weights \
    --bodyDetectorConf ../pretrained_models/yolov3/yolov3.cfg \
    --embeddingGeneratorWeights ../pretrained_models/abd/market_checkpoint_best.pth.tar \
    --saveFolder /Users/adrianlorenzomelian/Desktop/

: '
./personReidentifier.py --yolo --abd \
    --embedding /Users/adrianlorenzomelian/Desktop/test_image-2.png \
    --bodyDetectorWeights ../pretrained_models/yolov3/yolov3.weights \
    --bodyDetectorConf ../pretrained_models/yolov3/yolov3.cfg \
    --embeddingGeneratorWeights ../pretrained_models/abd/market_checkpoint_best.pth.tar \
    --savePath /Users/adrianlorenzomelian/Desktop/test_image-2.h5


./personReidentifier.py --cmc \
    --gallery /Users/adrianlorenzomelian/tfg/datasets/lpatrail/deint_embeddings \
    --galleryLocation 1 \
    --query /Users/adrianlorenzomelian/tfg/datasets/lpatrail/deint_embeddings \
    --queryLocation 2 \
    --ids /Users/adrianlorenzomelian/tfg/datasets/lpatrail/deint_embeddings/ids.txt


./personReidentifier.py --camera --yolo --alignedreid \
    --bodyDetectorWeights ../pretrained_models/yolov3/yolov3.weights \
    --bodyDetectorConf ../pretrained_models/yolov3/yolov3.cfg \
    --embeddingGeneratorWeights ../pretrained_models/alignedreid/checkpoint_ep500.pth.tar \
    --embeddingGeneratorClasses 100 \
    --ids /Users/adrianlorenzomelian/tfg/datasets/lpatrail/deint_embeddings/ids.txt \
    --gallery /Users/adrianlorenzomelian/tfg/datasets/lpatrail/deint_embeddings \
    --galleryLocation 1
''