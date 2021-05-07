import numpy as np
import numpy.random as rn
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def get_images(n: int, dsplit: str = 'train', seed: int = None):
    '''
    Returns n randomly selected testing, training, or validation data.

    Takes ~13 sec / 100 iter with n = 100
    Takes ~118 sec / 1000 iter with n = 100
    '''

    # Make sure train param is valid
    if dsplit not in ['train', 'valid', 'test']:
        raise Exception("dtype argument must be train, valid, or test.")

    # Load labeled dataframes
    PATHDIR = Path('data')
    df = pd.read_csv(PATHDIR / f'{dsplit}.csv', header=0).drop(
        ['original_path', 'Unnamed: 0', 'label_str'], axis=1)

    df_real = df[df['label'] == 0]
    df_fake = df[df['label'] == 1]

    # Get the number of files in the directory of interest
    n_files = {"train": 50000, "valid": 10000, "test": 10000}[dsplit]

    # Make sure you don't want more pictures than we have
    if n > n_files:
        raise Exception(f'There are not {n} files in the {dsplit} folder')

    # Set a seed if present
    if seed is not None:
        rn.seed(seed)

    # Get n balanced random ids
    sample_ids_real = pd.DataFrame(
        {'id': rn.choice(df_real['id'].to_numpy(), size=int(n / 2))})
    sample_ids_fake = pd.DataFrame(
        {'id': rn.choice(df_fake['id'].to_numpy(), size=int(n / 2))})

    sample_ids = sample_ids_real.append(sample_ids_fake)

    # Get the labels and image paths from the ids
    sample_df = df.copy()
    sample_df = sample_df[sample_df['id'].isin(sample_ids['id'].to_numpy())]

    shuffled_sample = sample_df.sample(frac=1)

    return shuffled_sample


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
        X[img_idx, :, :, :] = img / 255.0

    return X, y[np.newaxis].reshape(-1, 1)


def imgs_to_numpy(dsplit):
    """
    Saves the images to prevent them having to be loaded
    as numpy arrays later
    """
    # Check that the split exists
    if dsplit not in ('train', 'valid', 'test'):
        raise Exception('dsplit must be `train`, `test`, or `valid`')

    # Load labeled dataframes
    PATHDIR = Path('data')
    DATADIR = Path('data/') / 'real_vs_fake' / 'real-vs-fake'
    SAVEPATH = PATHDIR / 'data_array'

    df = pd.read_csv(PATHDIR / f'{dsplit}.csv', header=0).drop(
        ['original_path', 'Unnamed: 0', 'label_str'], axis=1)

    # Create containers for the image data
    n = df.shape[0]
    X = np.empty(shape=(n, 256, 256, 3))
    y = df['label'].to_numpy()[np.newaxis].reshape(-1, 1)

    # Load in the images
    for img_idx, img_path in enumerate(df['path']):
        img = plt.imread(DATADIR / img_path)
        X[img_idx, :, :, :] = img / 255.0

    # Save the images as numpy arrays
    with open(SAVEPATH / f'X_{dsplit}.npy', 'wb') as file:
        np.save(file, X)

    with open(SAVEPATH / f'y_{dsplit}.npy', 'wb') as file:
        np.save(file, y)


def main():
    """
    Function for testing
    """
    # imgs_to_numpy('train')

    # with open('data/data_array/X_train.npy', 'rb') as f:
    #     X = np.load(f)

    # with open('data/data_array/y_train.npy', 'rb') as f:
    #     y = np.load(f)

    # plt.imshow(X[1, :, :, :])
    # print(X[1, :, :, :].shape)
    # plt.show()


if __name__ == '__main__':
    main()
