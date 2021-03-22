import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

from image import ImageSetObject
from model import get_model, get_text_from_predict, score, loss_function

################################
# Folders
folders = '0123456789abcdef'

# Random seed
random_state=0

# Parameters
epochs = 1 # 1000
batch_size = 128
lr=1e-3
name=f'gsk'
new_shape=[128, 128]
################################


# Train labels
train_labels = pd.read_csv("bms-molecular-translation/train_labels.csv")
train_labels['InChI'] = train_labels['InChI'].apply(lambda x: x.replace('InChI=', ''))
train_labels = train_labels.set_index("image_id")
print(f"Size training set: {len(train_labels)}")

# Text processing
text = ''.join(train_labels['InChI'].values)

# Vocab
vocab = [' '] + sorted(set(text))
vocab_size = len(vocab)

# Mapping
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Max length
max_len = max([len(x) for x in train_labels['InChI']])

''''
Load image data, create target and model
'''

# Optimizer
optimizer = Adam(learning_rate=lr)

# Model 
model = get_model(max_len, vocab)
model.compile(optimizer=optimizer, loss=loss_function)
model.summary()

'''
Train
'''

# Images data 
dataset = 'train'

i = folders[0]
j = folders[0]
# for i in folders:
#     for j in folders:
 
for k in tqdm(folders):

    path = f'bms-molecular-translation/{dataset}/{i}/{j}/{k}/'

    # Files
    list_names = os.listdir(path)
    list_paths = [path for _ in list_names]

    # Image data
    ImageSet = ImageSetObject(list_names, list_paths)
    ImageSet.load_set(new_shape)
    data = ImageSet.X

    # Text targets
    list_id = [x.split('.')[0] for x in ImageSet.list_names]
    targets = train_labels.loc[list_id, 'InChI'].values
    targets = [[char2idx[x] for x in target] for target in targets]
    targets = pad_sequences(targets, padding='post', maxlen=max_len)

    # Train
    history = model.fit(data, targets, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save_weights(f'outputs/{name}.h5')

    # Score 
    y_true=[''.join([idx2char[y] for y in yy]) for yy in targets]
    y_predict=get_text_from_predict(model, data, idx2char)

# Last train score
print(f"\t> Score: {score(y_true, y_predict)}")

'''
Validation
'''

# Images data 
dataset = 'train'

i = folders[0]
j = folders[1]
k = folders[0]

print(f"### {i} - {j} - {k} ###")

path = f'bms-molecular-translation/{dataset}/{i}/{j}/{k}/'

# Files
list_names = os.listdir(path)
list_paths = [path for _ in list_names]

# Image data
ImageSet = ImageSetObject(list_names, list_paths)
ImageSet.load_set(new_shape)
data_validation = ImageSet.X

# Text targets
list_id = [x.split('.')[0] for x in ImageSet.list_names]
targets = train_labels.loc[list_id, 'InChI'].values
targets = [[char2idx[x] for x in target] for target in targets]
targets = pad_sequences(targets, padding='post', maxlen=max_len)

# Predict
y_val_true=[''.join([idx2char[int(y)] for y in yy]) for yy in targets]
y_val_predict=get_text_from_predict(model, data_validation, idx2char)
print(f"\t> Score: {score(y_val_true, y_val_predict)}")
