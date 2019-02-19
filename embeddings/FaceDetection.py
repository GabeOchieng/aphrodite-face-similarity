import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN

class FaceDetection:
    def __init__(self):
        self.margin = 44
        self.im_size = 160
        self.min_size = 20
        self.factor = 0.709
        self.threshold = [0.6, 0.7, 0.7]
        self.detector = MTCNN(
            min_face_size=self.min_size,
            steps_threshold=self.threshold,
            scale_factor=self.factor)

    def get_faces_mtcnn(self, im_path):
        faces = []
        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.detector.detect_faces(image)

        if len(results) > 0:

            for res in results:
                box = res['box']
                keypoints = res['keypoints']
                confidence = res['confidence']

                if confidence >= 0.58:
                    x_start = box[0]
                    y_start = box[1]
                    x_end = x_start + box[2]
                    y_end = y_start + box[3]

                    cropped = image[y_start:y_end, x_start:x_end, :]

                    resized = cv2.resize(
                        cropped, 
                        (self.im_size, self.im_size),
                        interpolation=cv2.INTER_LINEAR)

                    faces.append({
                        'rect': resized, 
                        'bbox': box
                    })        

        return faces
