import os
import sys

#############################################################
# DIRECTORIES
#############################################################

# GRID directory
if 'GRID_DIR' not in dir():
    GRID_DIR = os.path.dirname(os.path.realpath(__file__))

if GRID_DIR not in sys.path:
    sys.path.append(GRID_DIR)

if 'GEN_DIR' not in dir():
    GEN_DIR = os.path.join(GRID_DIR, "gen-images-and-words")

if GEN_DIR not in sys.path:
    sys.path.append(GEN_DIR)

# GRIDcorpus dataset directory
# GRID_DATA_DIR = '/media/voletiv/01D2BF774AC76280/Datasets/GRIDcorpus'
GRID_DATA_DIR = '/home/voletiv/Datasets/GRIDcorpus'

#############################################################
# PARAMETERS FOR GRIDCORPUS
#############################################################

FRAMES_PER_VIDEO = 75

WORDS_PER_VIDEO = 6

MAX_FRAMES_PER_WORD = 30

NUM_OF_MOUTH_PIXELS = 1600

MOUTH_W = 40

MOUTH_H = 40

NUM_OF_UNIQUE_WORDS = 53     # including silent and short pause

# excluding 'sil' and 'sp'
# wordsVocabSize = (nOfUniqueWords - 2) + 1
GRID_VOCAB_SIZE = NUM_OF_UNIQUE_WORDS - 2

TRAIN_VAL_SPEAKERS_LIST = [1, 2, 3, 4, 5, 6, 7, 10]
SI_SPEAKERS_LIST = [13, 14]

# grid_vocab list
GRID_VOCAB_LIST_FILE = os.path.join(GRID_DIR, 'grid_vocabulary.txt')


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
