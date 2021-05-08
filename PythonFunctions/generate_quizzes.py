#!venv/bin/python3

import sys
from helper_functions import create_quiz

def main():
    '''
    Function to create n number of quizzes. Names must be not specified OR unique. 

    Example call from terminal (from deepFakeCNN):
        python3 PythonFunctions/generate_quizzes.py 5 Dominic Matthew Josh

        - Creates 5 quizzes, 3 of which have the names specified, and 2 of which are labeled with 
          random numbers
    '''
    # Specify the number of images
    NUM_IMGS = 30

    # Get the number of input args
    n_args = len(sys.argv)

    # Check if the number was input
    if n_args < 2:
        raise Exception('Must specify the number of quizzes to generate. Can specify names as well.')

    # Make sure the input is numeric
    n = int(sys.argv[1])

    # Check if names were provided
    names = [None] * n
    if n_args > 2:
        names = sys.argv[2:]

        # Make sure there are enough names for the number of quizzes
        while n > len(names):
            names.append(None)

    # Generate the quizzes
    for i in range(n):
        create_quiz(NUM_IMGS, name=names[i])


if __name__ == '__main__':
    main()
