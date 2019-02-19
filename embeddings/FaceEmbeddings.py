import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from facenet import Facenet
from FaceDetection import FaceDetection

class FaceEmbeddings:
    def __init__(self, model_dir):
        self.threshold = 1.08
        self.embeddings = None
        self.images_placeholder = None
        self.phase_train_placeholder = None
        self.model_dir = model_dir
        self.face_detector = FaceDetection()

    def _get_embeddings(self, sess, processed_image):
        reshaped = np.expand_dims(processed_image, axis=0)

        feed_dict = {
            self.image_placeholder: reshaped,
            self.phase_train_placeholder: False
        }

        feature_vector = sess.run(
            self.embeddings, feed_dict=feed_dict)

        return feature_vector

    def _euclidean_distance(self, input_1, input_2, eps=0):
        distance = np.sqrt(eps + np.sum(
            np.square(np.subtract(input_1, input_2))))
        return distance

    def convert_embeddings(self, im_path):
        extracted = []

        with tf.Session() as sess:
            graph = tf.get_default_graph()
            Facenet.load_model(self.model_dir)

            self.image_placeholder = graph.get_tensor_by_name('input:0')
            self.phase_train_placeholder = graph.get_tensor_by_name('phase_train:0')
            self.embeddings = graph.get_tensor_by_name('embeddings:0')

            faces = self.face_detector.get_faces_mtcnn(im_path)

            for face in faces:
                rect = face['rect']
                plt.imshow(rect)
                plt.show()
                prewhitened = Facenet.prewhiten(rect)
                embeddings = self._get_embeddings(sess, prewhitened)
                extracted.append(embeddings)

        return extracted

    def compare_2_faces(self, embeddings_1, embeddings_2):
        distance = self._euclidean_distance(
            embeddings_1, embeddings_2)
        if distance <= self.threshold:
            return 'Same Person'
        else:
            return 'Not Same Person'
