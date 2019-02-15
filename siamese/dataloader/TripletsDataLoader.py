import os
import random
import numpy as np
from dataloader.DataLoader import DataLoader

class TripletsDataLoader(DataLoader):
    def __init__(self, config):
        super(TripletsDataLoader, self).__init__(config)
