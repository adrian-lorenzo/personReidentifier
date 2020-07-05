# Person Reidentifier

**Implementation with CLI interface of the work done for my degree final project `"Appereance and context information combination 
for re-identificaton in sports competitions"`** (`"Combinación de apariencia e información de contexto para la re-identificación en competiciones deportivas").

With this implementation you can:

* Run the studied body detectors and **get the body images or bounding boxes of the bodies in a given image.**
* **Generate a representative embedding from a body image** using studied body embedding generators.
* **Compute the CMC curve of a given gallery set and query set, using one of the proposed heuristics.**
* **Test the whole system from a video feed, using a given gallery or query set.**

## Requirements

* Python `>= 3.7`.
* Pip `>= 20`.

## Project set up

In your system or virtual `Python` environment, install the requirements/dependencies:

```shell
pip install -r requirements.txt
```

Then you can use it by executing the `personIdentifier.py` script:

```shell
./personIdentifier.py
```

## CLI interface manual

### Program configuration

| Option | Description |
|---|---|
| `-h`, `--help` | Shows the help message and exit. |
| `--silent` | Runs program in silent mode (no verbose). |

### Embedding generation configuration

| Option | Description |
|---|---|
| `--alignedreid` | Uses `AlignedReId++` for body embedding generation. |
| `--abd` | Uses `Attentive but Diverse Network` for body embedding generation. |
| `--embeddingGeneratorWeights` `weights_file` | Embedding generator weights file. |
| `--embeddingGeneratorClasses` `classes_number` | Embedding generator number of classes. |

### Detection configuration

| Option | Description |
|---|---|
| `--mtcnn` | Uses `MTCNN` for face detection. |
| `--yolo` | Uses `YOLOv3` for person detection. |
| `--faster` | Uses `FasterRCNN` for person detection. |
| `--mask` | Uses `MaskRCNN for person detection. |
| `--bodyDetectorWeights` `weights_path` | Body detector weights file. |
| `--bodyDetectorConfig` `configuration_file_path` | Body detector configuration file. Only needed for `YOLOv3`. |
| `--detectionThreshold` `threshold_float` | Specifies the detection threshold. Default value: `0.8` |

### Features

#### Embedding generation

| Option | Description |
|---|---|
| `--embedding` `image_file`  | Returns embedding of the given body. Default embedding generator: `AlignedReId++`. |
| `--savePath` `save_embedding_path` | Embedding save path (use `.h5` extension). |

#### Body detection

| Option | Description |
|---|---|
| `--detection` `image_path` |  Returns bodies of the given image in the specified folder using `--saveFolder` option. Default body detector: `MaskRCNN`. |
| `--boundingBox` | Draws bounding boxes in image instead of getting the body image. |
| `--saveFolder` `folder_path` | Specifies the save folder for the `--detection` option. |

#### System evaluation: CMC curves, heuristics

##### CMC curve

Dataset must follow the next structure:

```
  Root
  |
  |
  |     Id
  |--   0
  |     |
  |     |   
  |     |     Location
  |     |--   0
  |     |     |
  |     |     |
  |     |     |     Image number
  |     |     |--   0
  |     |     |
  |     |     |--   1
  |     |     |
  |     |     |--   2
  |     |     |
  |     |     |--   ...
  |     |
  |     |
  |     |
  |     |--   1
  |     |
  |     |--   2
  |     |
  |     |--   ...
  |
  |
  |
  |--   1
  |
  |--   2
  |
  |--   3
  |
  |--   ...
```

| Option | Description |
|---|---|
| `--cmc` | Returns the CMC curve discrete values of the given query and gallery. |
| `--plot` | Plots the calculated CMC curve. |
| `--ids` `ids_file_path` | The ids list files. |
| `--query` `query_directory_path` | The query embeddings directory. |
| `--gallery` `gallery_directory_path` | The gallery embeddings directory. |
| `--queryLocation` `location_number` | The choosen query location in the dataset. |
| `--galleryLocation` `location_number` | The choosen gallery location in the dataset. |
| `--oneIndexed` | Check this option if the dataset indexes starts at 1. |
| `--topNum` `top_number` | Top number of CMC curve. |

##### Heuristic configuration

###### Time heuristic

| Option | Description |
|---|---|
| `--timeHeuristic` | Display the CMC curve using time heuristic of the given query and gallery. |
| `--raceRanking` `rank_file_path` | Previous race ranking file for CMC curve for heuristics. |
| `--windowSize` `integer` | Window size for time heuristic. Default value: `30`. |
| `--shiftSize` `integer` | Shift size for time heuristic. Default value: `1`. |
| `--shiftProp` `float` | Shift proportion for time heuristic. Default value: `1`. |

###### Space heuristic

| Option | Description |
|---|---|
| ` --spaceHeuristic` | Display the CMC curve using space heuristic of the given query and gallery. |
| `--fps` `integer` | FPS for space heuristic. Default value: `5`. |
| `--timeout` `float` | Timeout for space heuristic. Default value: `10`. |

#### Complete system test

| Option | Description |
|---|---|
| `--video` `video_path` | Runs system with specified video file. |
| `--camera` | Runs system using system camera. |

##### Open world assumption parameters (experimental)

This features has not been tested and were implemented for experimental intentions only. Further work in this area can be made, as told in the project report.

| Option | Description |
|---|---|
| `--openWorldThreshold` `float` | The minimum value to consider an identity as equal as one in the gallery, used in the open world assumption. |
| `--maxDescriptors` `integer` | The maximum number of descriptors per gallery. |
                                 
## Credits:

### Models

* [MaskRCNN implementation and pre-trained weights.](https://github.com/matterport/Mask_RCNN)
* [FasterRCNN frozen graph and weights.](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
* [YOLOv3 weights and configuration file.](https://pjreddie.com/darknet/yolo/)
* [AlignedReId++ implementation and pre-trained weights.](https://github.com/michuanhaohao/AlignedReID)
* [ABD-Net implementation and pre-trained weights.](https://github.com/TAMU-VITA/ABD-Net)

### Libraries

* [Tensorflow](https://www.tensorflow.org)
* [PyTorch](https://pytorch.org)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Pillow](https://pillow.readthedocs.io/en/stable/)
* [Keras](https://keras.io)
* [OpenCV](https://opencv.org)
* [Matplotlib](https://matplotlib.org)
* [Numpy](https://numpy.org)
* [MTCNN](https://github.com/ipazc/mtcnn)

### Tools

* [Git](https://git-scm.com)
* [PyCharm](https://www.jetbrains.com/es-es/pycharm/)
* [Visual Studio Code](https://code.visualstudio.com)


                        
            
     
         
  
                        
    
     
       
       
           
  
           
                
  
                        
  
                        
