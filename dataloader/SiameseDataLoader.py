import os
import random
import numpy as np
from tqdm import tqdm
from dataloader.DataLoader import DataLoader

# TODO
class SiameseDataLoader(DataLoader):
    def __init__(self, config):
        super(SiameseDataLoader, self).__init__(config)
        self.im_size = config.image.size
        self.batch_size = 16
        self.total_epochs = 0
        self.current_index = 0
        self.iteration = 0
        self.initialize_dataset()

    def initialize_dataset(self):
        train_package = self.get_dataset(
            self.config.dataloader.train_dataset)
        val_package = self.get_dataset(
            self.config.dataloader.val_dataset)

        self.images_train = np.array(train_package[0])
        self.images_test = np.array(val_package[0])
        self.labels_train = np.expand_dims(np.array(train_package[1]), axis=1)
        self.labels_test = np.expand_dims(np.array(val_package[1]), axis=1)

        self.unique_train_label = np.unique(self.labels_train)
        self.unique_test_label = np.unique(self.labels_test)
        
        self.num_train = train_package[2]
        self.num_val = val_package[2]

        self.map_train_label_indices = {
            label: np.flatnonzero(
                self.labels_train == label) 
            for label in self.unique_train_label}
        
        self.map_test_label_indices = {
            label: np.flatnonzero(
                self.labels_test == label) 
            for label in self.unique_test_label}

    def get_siamese_similar_pair(self, trainable=True):
        source_unique_label = self.unique_train_label \
            if trainable else self.unique_test_label
        source_map_label = self.map_train_label_indices \
            if trainable else self.map_test_label_indices

        label = np.random.choice(
            source_unique_label)
        l, r = np.random.choice(
            source_map_label[label], 
            size=2, 
            replace=False)
        return l, r, 0

    def get_siamese_disimilar_pair(self, trainable=True):
        source_unique_label = self.unique_train_label \
            if trainable else self.unique_test_label
        source_map_label = self.map_train_label_indices \
            if trainable else self.map_test_label_indices

        label_l, label_r = np.random.choice(
            source_unique_label, 
            size=2, 
            replace=False)
        l = np.random.choice(
            source_map_label[label_l])
        r = np.random.choice(
            source_map_label[label_r])
        return l, r, 1

    def get_siamese_pair(self, trainable=True):
        if np.random.random() < 0.5:
            return self.get_siamese_similar_pair(trainable)
        else:
            return self.get_siamese_disimilar_pair(trainable)

    def get_siamese_batch(self, n, trainable=True):
        source = self.images_train \
            if trainable else self.images_test 

        idx_l = []
        idx_r = []
        labels = []

        for _ in range(n):
            l, r, x = self.get_siamese_pair(trainable)
            idx_l.append(l)
            idx_r.append(r)
            labels.append(x)

        batch_l = source[idx_l]
        batch_r = source[idx_r]
        labels = np.array(labels)

        inputs_l = self.read_images(
            batch_l, self.im_size)
        inputs_r = self.read_images(
            batch_r, self.im_size)
        
        return inputs_l, inputs_r, labels

    @staticmethod
    def create_data_txt(self, dataset_path, dataset_dir):
        fnames = os.listdir(dataset_dir)

        def isdir(x):
            if os.path.isdir(os.path.join(dataset_dir, x)):
                return True

        dirnames = [name for name in fnames if isdir(name)]

        with open(dataset_path, 'w') as f:
            lines = []
            label = 0
            for idx, dir_ in enumerate(dirnames):
                temp = dir_.split('s')
                label = temp[1]
                images_path = os.listdir(
                    os.path.join(dataset_dir, dir_))

                desc = 'Writing aphrodite-dataset.txt for input dataset'
                for path in tqdm(images_path, desc=desc):
                    if os.path.isfile(os.path.join(dataset_dir, dir_, path)):
                        fpath = os.path.join(
                            dataset_dir, dir_, path)
                        line = ''.join(
                            fpath + ' ' + str(label) + ' ' + '\n')
                        lines.append(line)

            random.shuffle(lines)

            for line in lines:
                f.write(line)
