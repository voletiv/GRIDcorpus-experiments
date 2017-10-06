import os
import sys

#############################################################
# PARAMS TO BE SET
#############################################################

GRID_DATA_DIR = '/home/voletiv/Datasets/GRIDcorpus'
GRID_SAVE_DIR = GRID_DATA_DIR

#############################################################
# PARAMS
#############################################################

SIAMESE_DIR = os.path.dirname(os.path.realpath(__file__))

if SIAMESE_DIR not in sys.path:
    sys.path.append(SIAMESE_DIR)

ROOT_DIR = os.path.normpath(os.path.join(SIAMESE_DIR, '../'))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from params import *

#############################################################
# TRUE PARAMS
#############################################################

FRAMES_PER_VIDEO = 75
WORDS_PER_VIDEO = 6
MAX_FRAMES_PER_WORD = 14
NUM_OF_MOUTH_PIXELS = 1600
MOUTH_W = 40
MOUTH_H = 40
GRID_VOCAB_SIZE = 51

TRAIN_VAL_SPEAKERS_LIST = [1, 2, 3, 4, 5, 6, 7, 10]
SI_SPEAKERS_LIST = [13, 14]
