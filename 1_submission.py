
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
lr=1e-3
name=f'gsk'
new_shape=[128, 128]
################################

'''
Submission
'''

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

# Optimizer
optimizer = Adam(learning_rate=lr)

# Model 
model = get_model(max_len, vocab)
model.compile(optimizer=optimizer, loss=loss_function)
model.summary()

# Load weights
model.load_weights(f'outputs/{name}.h5')
print("Loaded model from disk")
model.compile(optimizer=optimizer, loss=loss_function)

# Sample_submission
sample_submission = pd.read_csv("bms-molecular-translation/sample_submission.csv")
sample_submission = sample_submission.set_index('image_id')

# Images data 
dataset = 'test'

for i in tqdm(folders[0:1]):
    for j in tqdm(folders[0:1]):
        for k in tqdm(folders):

            path = f'bms-molecular-translation/{dataset}/{i}/{j}/{k}/'

            # Files
            list_names = os.listdir(path)
            list_paths = [path for _ in list_names]

            # Image data
            ImageSet = ImageSetObject(list_names, list_paths)
            ImageSet.load_set(new_shape)
            data_test = ImageSet.X

            # Text targets
            list_id = [x.split('.')[0] for x in ImageSet.list_names]

            # Predict
            y_test_predict=get_text_from_predict(model, data_test, idx2char)
            y_test_predict=['InChI='+x for x in y_test_predict]

            # Prepare df
            df_y_test_predict = pd.DataFrame([list_id, y_test_predict], index = ['image_id','InChI']).transpose().set_index('image_id')
            sample_submission.loc[df_y_test_predict.index, 'InChI'] = df_y_test_predict['InChI']

            # Export
            sample_submission.reset_index().to_csv('outputs/submission.csv', index=False)