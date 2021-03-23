import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))

import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.data import Dataset

from image import ImageSetObject
from model import VAE

from tensorflow import get_logger
get_logger().setLevel('WARNING')

################################
# Folders
folders = '0123456789abcdef'

# Random seed
random_state = 0

# Parameters
local_epochs = 1
global_epochs = 1
lr = 1e-3
name = f'gsk_VAE'
new_shape = [128, 128]
latent_dim = 8
################################

# Train labels
train_labels = pd.read_csv("bms-molecular-translation/train_labels.csv")
train_labels['InChI'] = train_labels['InChI'].apply(
    lambda x: x.replace('InChI=', ''))
train_labels = train_labels.set_index("image_id")
print(f"Size training set: {len(train_labels)}")

# Text processing
text = ''.join(train_labels['InChI'].values)

# Vocab
vocab = [' '] + sorted(set(text))
vocab_size = len(vocab)

# Mapping
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Max length
max_len = max([len(x) for x in train_labels['InChI']])

''''
Load image data, create target and model
'''

# Optimizer
optimizer = Adam(learning_rate=lr)

# VAE Model
input_shape = [None, 128, 128, 1]
model = VAE(name, latent_dim, input_shape)

'''
Train
'''

# Images data
dataset = 'train'

for epoch in tqdm(range(global_epochs)):
    for i in tqdm(folders[:1]):
        for j in tqdm(folders[:1]):
            for k in tqdm(folders):

                path = f'bms-molecular-translation/{dataset}/{i}/{j}/{k}/'

                # Files
                list_names = os.listdir(path)
                list_path = [path]*len(list_names)

                # Image data
                ImageSet = ImageSetObject(list_names, list_path)
                ImageSet.prepare_data(new_shape, filtering=False, adjust=True)

                batch_size = ImageSet.X.shape[0]
                train_dataset = Dataset.from_tensor_slices(ImageSet.X).batch(batch_size)

                # Train
                model = model.train(optimizer, 
                    train_dataset, 
                    local_epochs,
                    batch_size)