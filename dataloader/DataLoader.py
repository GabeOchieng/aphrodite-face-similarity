import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.images_train = None
        self.images_test = None
        self.labels_train = None
        self.labels_test = None
        self.unique_train_label = None
        self.map_train_label_indices = dict()

    def read_images(self, filenames, im_size, 
                    should_invert=True):
        images = []
        labels = []

        for i in range(0, len(filenames)):
            img = cv2.imread(filenames[i])

            if img is None:
                print(filenames[i])
                pass

            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if should_invert:
                grayscale = cv2.bitwise_not(grayscale)

            if grayscale.shape != (im_size, im_size):
                grayscale = cv2.resize(grayscale, (im_size, im_size))

            grayscale = np.expand_dims(grayscale, axis=3)

            images.append(grayscale)

        return np.array(images)

    def get_dataset(self, dataset_path):
        filenames = []

        with open(dataset_path, 'r') as f:
            for line in tqdm(f, desc=dataset_path):
                filenames.append(line)

        images = [fname.split(' ')[0] \
            for fname in filenames]
        labels = [fname.split(' ')[1] \
            for fname in filenames]

        num_samples = len(images)

        return images, labels, num_samples

    def get_batch(self, trainable=True, random=True):

        X = self.images_train \
            if trainable else self.images_test
        y = self.labels_train \
            if trainable else self.images_test

        num_files = self.num_train \
            if trainable else self.num_val
        end_index = self.current_index + self.batch_size

        if random:
            self.indices = np.random.choice(
                num_files,
                self.batch_size)
        else:
            self.indices = np.arange(
                self.current_index,
                end_index)

            if end_index > self.num_files:
                self.indices[self.indices >= num_files] = np.arange(
                    0, np.sum(self.indices >= self.num_files))

        image_batch = self.read_images(
            [X[index] for index in self.indices])
        label_batch = [y[index] for index in self.indices]

        image_batch = np.reshape(
            np.squeeze(
                np.stack(
                    [image_batch])),
            newshape=(self.batch_size, self.im_size, self.im_size, 3))

        label_batch = np.stack(label_batch)

        self.current_index = end_index

        if self.current_index > num_files:
            self.current_index = self.current_index - num_files

        return image_batch, label_batch
