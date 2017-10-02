import glob
import numpy as np
import os
import tqdm

# Import params
from params import *

#############################################################
# LOAD IMAGE DIRS AND WORD NUMBERS
#############################################################

def load_image_dirs_and_word_numbers(trainValSpeakersList = [1, 2, 3, 4, 5, 6, 7, 9],
                                        valSplit = 0.1,
                                        siList = [10, 11]):
    # TRAIN AND VAL
    trainDirs = []
    trainWordNumbers = []
    valDirs = []
    valWordNumbers = []
    np.random.seed(29)

    # For each speaker
    for speaker in sorted(tqdm.tqdm(trainValSpeakersList)):
        speakerDir = os.path.join(rootDir, 's' + '{0:02d}'.format(speaker))
        # List of all videos for each speaker
        vidDirs = sorted(glob.glob(os.path.join(speakerDir, '*/')))
        totalNumOfImages = len(vidDirs)
        # To shuffle directories before splitting into train and validate
        fullListIdx = list(range(totalNumOfImages))
        np.random.shuffle(fullListIdx)
        # Append training directories
        for i in fullListIdx[:int((1 - valSplit) * totalNumOfImages)]:
            for j in range(wordsPerVideo):
                trainDirs.append(vidDirs[i])
                trainWordNumbers.append(j)
        # Append val directories
        for i in fullListIdx[int((1 - valSplit) * totalNumOfImages):]:
            for j in range(wordsPerVideo):
                valDirs.append(vidDirs[i])
                valWordNumbers.append(j)

    # Numbers
    print("No. of training words: " + str(len(trainDirs)))
    print("No. of val words: " + str(len(valDirs)))

    # SPEAKER INDEPENDENT
    siDirs = []
    siWordNumbers = []
    for speaker in sorted(tqdm.tqdm(siList)):
        speakerDir = os.path.join(rootDir, 's' + '{0:02d}'.format(speaker))
        vidDirs = sorted(glob.glob(os.path.join(speakerDir, '*/')))
        for i in fullListIdx:
                for j in range(wordsPerVideo):
                    siDirs.append(vidDirs[i])
                    siWordNumbers.append(j)

    # Numbers
    print("No. of speaker-independent words: " + str(len(siDirs)))

    # Return
    return trainDirs, trainWordNumbers, valDirs, valWordNumbers, siDirs, siWordNumbers
