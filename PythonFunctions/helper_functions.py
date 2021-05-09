from shutil import copyfile
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

def create_quiz(n: int, dsplit: str = 'train', seed: int = None, name: str = None):
    # Check dsplit variable
    if dsplit not in {'train', 'valid', 'test'}:
        raise Exception('dsplit must be "train", "valid", or "test"')
    
    # Create directories
    QUIZDIR = Path('.') / 'Quizzes' 
    DATADIR = Path('data') / 'real_vs_fake' / 'real-vs-fake' 

    # Check name variable
    upper_bound = 9999999
    bound_padding = int(np.log10(upper_bound)) + 1
    if name is None:
        name = str(rn.randint(0, upper_bound)).zfill(bound_padding)

    # Create save directory
    SAVEDIR = QUIZDIR / name / name / 'images'

    # Make sure save directory is unique
    while SAVEDIR.is_dir():
        name = str(rn.randint(0, upper_bound)).zfill(bound_padding)
        SAVEDIR = QUIZDIR / name / 'images' 
    SAVEDIR.mkdir(parents=True, exist_ok=False)

    # Copy the README.txt file
    copyfile(QUIZDIR / 'README.txt', QUIZDIR / name / name / 'README.txt')

    # Randomly select n images
    shuffled_sample = get_images(n, dsplit, seed).reset_index()

    # Save the answer key
    y = shuffled_sample['label'].to_numpy()
    
    # Copy the images to a new location
    padding = int(np.log10(n)) + 1
    paths = shuffled_sample['path'].to_numpy()

    for img_idx, img_path in enumerate(paths):
        idx_str = str(img_idx + 1).zfill(padding)
        img_src = DATADIR / img_path
        img_dest = SAVEDIR / f'img{idx_str}.jpg'
        
        copyfile(img_src, img_dest)

    # csv to fill in
    fill = pd.DataFrame({'image': shuffled_sample.index.to_numpy() + 1, 'type': [''] * n})
    fill.to_csv(SAVEDIR / 'quiz.csv', index=False)

    # Answers
    answers = pd.DataFrame({'image': shuffled_sample.index.to_numpy() + 1, 'type': shuffled_sample['label'].to_numpy()})
    answers.to_csv(QUIZDIR / name / 'answers.csv', index=False)

    # Alert
    print(f'Created quiz for {name}')

def load_quizzes():
    '''
    Function to load in all of the paths with quiz answers
    '''
    # Location of the guesses

    # Location of the answers
    RESULTSDIR = Path('.') / 'QuizzesResults' / 'answers'
    QUIZDIR = Path('.') / 'Quizzes'

    # Get all the csv files
    quiz_files = RESULTSDIR.glob('*.csv')
    
    # Load in the master file
    master_quiz = pd.read_csv(Path('.') / 'QuizzesResults' / 'master_quiz.csv' )
    names = set(master_quiz['name'])

    for quiz_file in quiz_files:
        # Extract the id
        name = str(quiz_file).split('/')[2].split('_')[0]

        # Check if the name is already accounted for
        if name in names:
            continue

        # Create the dataframe for the new answers
        new_df = pd.read_csv(quiz_file)
        
        guesses = new_df['type']
        img_id = new_df['image']
        name_col = [name] * len(guesses)
        ans = pd.read_csv(QUIZDIR / name / 'answers.csv')['type']
        
        df = pd.DataFrame({
            'name'  : name_col,
            'img_id': img_id,
            'guess' : guesses,
            'true'  : ans
        })

        # Append the master df
        master_quiz = master_quiz.append(df)

    # Write the quizzes to the file
    master_quiz.to_csv(Path('.') / 'QuizzesResults' / 'master_quiz.csv', index = False)

def get_quiz_stats():
    '''
    Returns the quiz average. Will return more stats
    '''
    # Make sure the master is updated
    load_quizzes()

    master_df = pd.read_csv(Path('.') / 'QuizzesResults' / 'master_quiz.csv' )
    
    diff = np.abs(master_df['guess'].to_numpy() - master_df['true'].to_numpy())

    avg = 1 - np.sum(diff) / len(diff)
    
    return diff

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
    get_quiz_stats()

if __name__ == '__main__':
    main()
