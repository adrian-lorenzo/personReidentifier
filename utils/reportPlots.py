import matplotlib.pyplot as plt
import numpy as np

persons = [21, 20, 20, 21, 20, 19, 20, 21, 20, 20]
maskTP = [16, 14, 14, 12, 13, 12, 15, 14, 14, 12]
maskFP = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
fasterTP = [9, 9, 8, 9, 10, 10, 9, 8, 9, 7]
fasterFP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
yoloFiveTP = [10, 7, 7, 10, 10, 8, 8, 8, 7, 7]
yoloFiveFP = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
yoloSevenTP = [5, 3, 2, 4, 4, 4, 4, 2, 2, 2]
yoloSevenFP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mtcnnTP = [2, 2, 2, 1, 1, 1, 1, 1, 1, 1]
mtcnnFP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ms = [136.48, 225.47, 235.96, 2543.64, 9477.34]


def detectorsFP():
    fotograms = range(1, 11)
    plt.figure(figsize=(9, 9))
    titleFont = {'fontname': 'Arial', 'size': '16', 'weight': 'bold'}
    axisFont = {'fontname': 'Arial', 'size': '15'}
    plt.plot(fotograms, mtcnnFP, "-.")
    plt.plot(fotograms, yoloFiveFP, "-.")
    plt.plot(fotograms, yoloSevenFP, "-.")
    plt.plot(fotograms, fasterFP, "-.")
    plt.plot(fotograms, maskFP, "-.")
    plt.yticks(np.arange(0, 2, 1))
    plt.xticks(np.arange(0, 11, 1))
    plt.xlabel("Fotograma", **axisFont)
    plt.ylabel("Falsos positivos", **axisFont)
    plt.title("Gráfica comparativa de los falsos positivos generados en la detección", **titleFont)
    plt.legend(['MTCNN', 'YOLOv3 (conf = 0.5)', 'YOLOv3 (conf = 0.7)', 'Faster RCNN', 'Mask RCNN'], loc=1,
               prop={"size": '12'})
    plt.grid()
    plt.savefig("/Users/adrianlorenzomelian/Desktop/comparativeFP.png")


def detectorsTP():
    fotograms = range(1, 11)
    plt.figure(figsize=(9, 9))
    titleFont = {'fontname': 'Arial', 'size': '16', 'weight': 'bold'}
    axisFont = {'fontname': 'Arial', 'size': '15'}
    plt.plot(fotograms, persons, "--")
    plt.plot(fotograms, mtcnnTP, "-.")
    plt.plot(fotograms, yoloFiveTP, "-.")
    plt.plot(fotograms, yoloSevenTP, "-.")
    plt.plot(fotograms, fasterTP, "-.")
    plt.plot(fotograms, maskTP, "-.")
    plt.yticks(np.arange(0, 24, 1))
    plt.xticks(np.arange(0, 11, 1))
    plt.xlabel("Fotograma", **axisFont)
    plt.ylabel("Personas detectadas", **axisFont)
    plt.title("Gráfica comparativa de los detectores de personas integrados", **titleFont)
    plt.legend(['Ground truth', 'MTCNN', 'YOLOv3 (conf = 0.5)', 'YOLOv3 (conf = 0.7)', 'Faster RCNN', 'Mask RCNN'],
               loc=1, prop={"size": '12'})
    plt.grid()
    plt.savefig("/Users/adrianlorenzomelian/Desktop/comparativeTP.png")


def detectorsTiming():
    x = np.arange(0, 5, 1)
    plt.figure(figsize=(12, 9))
    titleFont = {'fontname': 'Arial', 'size': '16', 'weight': 'bold'}
    axisFont = {'fontname': 'Arial', 'size': '15'}
    barlist = plt.bar(x, ms)
    barlist[0].set_color('b')
    barlist[1].set_color('y')
    barlist[2].set_color('g')
    barlist[3].set_color('r')
    barlist[4].set_color('m')
    plt.xticks(x, ['MTCNN', 'YOLOv3 (conf = 0.5)', 'YOLOv3 (conf = 0.7)', 'Mask RCNN', 'Faster RCNN'])
    plt.xlabel("Detector", **axisFont)
    plt.ylabel("Milisegundos", **axisFont)
    plt.title("Gráfica comparativa de la duración media en inferencia por cada detector", **titleFont)
    plt.grid()
    plt.savefig("/Users/adrianlorenzomelian/Desktop/comparativeTime.png")
