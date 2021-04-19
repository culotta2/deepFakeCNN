import numpy as np
import numpy.random as rn
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def get_images(n: int, dtype: str = 'train', seed: int = None):
    '''
    Returns n randomly selected testing, training, or validation data.

    Takes ~13 sec / 100 iter with n = 100
    Takes ~118 sec / 1000 iter with n = 100
    '''

    # Make sure train param is valid
    if dtype not in ['train', 'valid', 'test']:
        raise Exception("dtype argument must be train, valid, or test.")

    # Load labeled dataframes
    PATHDIR = Path('data')
    df = pd.read_csv(PATHDIR / f'{dtype}.csv', header=0).drop(
        ['original_path', 'Unnamed: 0', 'label_str'], axis=1)

    # Get the number of files in the directory of interest
    n_files = {"train": 50000, "valid": 10000, "test": 10000}[dtype]

    # Make sure you don't want more pictures than we have
    if n > n_files:
        raise Exception(f'There are not {n} files in the {dtype} folder')

    # Create the paths to the data
    datapath = Path('.') / 'data' / 'real_vs_fake' / 'real-vs-fake' / dtype
    fakepath = datapath / 'fake'
    realpath = datapath / 'real'

    # Set a seed if present
    if seed is not None:
        rn.seed(seed)

    # Get n random ids
    sample_ids = rn.choice(df['id'].to_numpy(), size=n)

    # Get the labels and image paths from the ids
    sample_df = df.copy()
    sample_df = sample_df[sample_df['id'].isin(sample_ids)]

    return sample_df


def prep_for_train(sample_df):
    """
    Gets the images from the get_image family of functions into a format
    that the model can understand.
    """
    # Save the labels
    y = sample_df['label'].to_numpy()

    # Path to the data
    DATADIR = Path('data/') / 'real_vs_fake' / 'real-vs-fake'

    # Load the sample images
    n = sample_df.shape[0]
    X = np.empty(shape=(n, 256, 256, 3))

    # Load in the images to be trained on
    for img_idx, img_path in enumerate(sample_df['path']):
        img = plt.imread(DATADIR / img_path)
        # plt.imshow(img)
        # plt.show()
        X[img_idx, :, :, :] = img / 255.0

    return X, y[np.newaxis].reshape(-1, 1)
