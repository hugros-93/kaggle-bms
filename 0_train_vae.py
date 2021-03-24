import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))


from tensorflow import get_logger
from model import VAE
from image import ImageSetObject
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import random
from tqdm import tqdm
import pandas as pd
import numpy as np

################################
# Folders
folders = '0123456789abcdef'

# Random seed
random_state = 0

# Parameters
local_epochs = 1
global_epochs = 1000
# batch_size = 64
lr = 1e-3
name = f'BMS_VAE'
new_shape = [128, 128]
latent_dim = 64
################################

''''
Load image data, create target and model
'''

# Optimizer
optimizer = Adam(learning_rate=lr)

# VAE Model
input_shape = [None, new_shape[0], new_shape[1], 1]
model = VAE(name, latent_dim, input_shape)

'''
Train
'''

# Images data
dataset = 'train'

for epoch in tqdm(range(global_epochs)):
    for i in folders[:1]:
        for j in folders[:1]:
            for k in tqdm(folders):

                path = f'bms-molecular-translation/{dataset}/{i}/{j}/{k}/'

                # Files
                list_names = os.listdir(path)
                list_path = [path]*len(list_names)

                # Image data
                ImageSet = ImageSetObject(list_names, list_path)
                ImageSet.prepare_data(new_shape, filtering=False, adjust=True)

                batch_size = ImageSet.shape[0]
                train_dataset = Dataset.from_tensor_slices(
                    ImageSet.X).batch(batch_size)

                # Train
                model = model.train(
                    optimizer,
                    train_dataset,
                    local_epochs,
                    batch_size)
