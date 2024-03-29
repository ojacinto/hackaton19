#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import time
import cv2
import os


class mlmodel(object):
    team = 'Print Team'
    mame = 'Yolo'

    def __init__(self, image_path=None):
        self.image_path = image_path

    def predict(self, image_path):
        (net, color, label) = self.loadYolo()
        (label, bounding, accuracy) = self.runYolo(net,
                image_path, COLORS=color, LABELS=label)
        detect_fruits = [label, accuracy, bounding]
        return detect_fruits

    def loadYolo(self):
        fullpath = os.path.dirname(os.path.abspath(__file__))

        from pathlib import Path
        mypath = Path().absolute()

        labelsPath = './Datos/fruitnames.names'

        LABELS = open(labelsPath).read().strip().split('\n')

        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                                   dtype='uint8')

        truew_path = os.path.abspath('Datos/1900.weights')
        truec_path = os.path.abspath('Datos/fruit_conf.cfg')        

        # truew_path = os.path.sep.join([fullpath, "datos","fruit_conf.cfg"])
        # truec_path = os.path.sep.join([fullpath, "datos","1900.weights"])
        # truew_path=str(truew_path).replace("\\","/")
        # truec_path=str(truec_path).replace("\\","/")
        # import ipdb; ipdb.set_trace()

        print('[INFO] loading YOLO from disk...')

        net = cv2.dnn.readNetFromDarknet(truec_path, truew_path)
        return (net, COLORS, LABELS)

    def runYolo(
        self,
        net,
        image,
        COLORS,
        LABELS,
        confidencep=.4,
        ):

        image = cv2.imread(image)
        (H, W) = image.shape[:2]

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        print('[INFO] YOLO took {:.6f} seconds'.format(end - start))

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidencep:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype('int'
                            )

                    x = int(centerX - width / 2)
                    y = int(centerY - height / 2)

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # import ipdb; ipdb.set_trace();

        (label, bounding, accuracy) = ([], [], [])
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidencep, .1)
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = '{}: {:.4f}'.format(LABELS[classIDs[i]],
                        confidences[i])
                cv2.putText(
                    image,
                    text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    )
                label.append(LABELS[classIDs[i]])
                accuracy.append(confidences[i])
                bb = dict(top=x, right=x + w, bottom=y, left=y + w)
                bounding.append(bb)
            #cv2.imshow("Image",image)
            #cv2.waitKey(0)
        # label,bounding,accuracy

        return (label, bounding, accuracy)

