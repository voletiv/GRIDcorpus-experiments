import numpy as np
import os
import sys

#############################################################
# PARAMS TO BE SET
#############################################################

rootDir = '/home/voletiv/Datasets/GRIDcorpus'
saveDir = rootDir
# rootDir = '/Neutron6/voleti.vikram/GRIDcorpus'
# # saveDir = '/Neutron6/voleti.vikram/GRIDcorpusResults/1-h32-d1-Adam-2e-4-s0107-s0920-s2227'
# saveDir = rootDir


#############################################################
# IMPORT
#############################################################

if 'ROOT_DIR' not in dir():
	ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


if 'LIPREADER_DIR' not in dir():
    LIPREADER_DIR = os.path.join(ROOT_DIR, 'lipreader')

if LIPREADER_DIR not in sys.path:
    sys.path.append(LIPREADER_DIR)

#############################################################
# TRUE PARAMS
#############################################################

framesPerVid = 75
wordsPerVideo = 6
framesPerWord = 14
nOfMouthPixels = 1600
mouthW = 40
mouthH = 40
nOfUniqueWords = 53     # including silent and short pause
# excluding 'sil' and 'sp'
# wordsVocabSize = (nOfUniqueWords - 2) + 1
wordsVocabSize = nOfUniqueWords - 2

# # Unique Words Idx
# uniqueWordsFile = os.path.join('uniqueWords.npy')
# uniqueWords = np.load(uniqueWordsFile)
# # num of unique words, including 'sil' and 'sp'
# nOfUniqueWords = len(uniqueWords)
# # Remove silent
# uniqueWords = np.delete(
#     uniqueWords, np.argwhere(uniqueWords == 'sil'))
# # Remove short pauses
# uniqueWords = np.delete(
#     uniqueWords, np.argwhere(uniqueWords == 'sp'))
# # Vocabulary size
# # excluding 'sil' and 'sp', +1 for padding
# wordsVocabSize = len(uniqueWords) + 1
# # Word indices
# wordIdx = {}
# # Start word indices from 1 (0 for padding)
# for i, word in enumerate(uniqueWords):
#     wordIdx[word] = i+1

# np.save("wordIdx", wordIdx)

currDir = os.path.dirname(os.path.realpath(__file__))

# wordIdx dictionary
wordIdx = np.load(os.path.join(currDir, "wordIdx.npy")).item()

# grid_vocab list
GRID_VOCAB_LIST_FILE = os.path.join(ROOT_DIR, 'grid_vocabulary.txt')


#############################################################
# LOAD VOCAB LIST
#############################################################

def load_gridcorpus_vocab_list(GRID_VOCAB_LIST_FILE):
    grid_vocab = []
    with open(GRID_VOCAB_LIST_FILE) as f:
        for line in f:
            word = line.rstrip().split()[-1]
            grid_vocab.append(word)
    return grid_vocab

GRID_VOCAB = load_gridcorpus_vocab_list(GRID_VOCAB_LIST_FILE)
