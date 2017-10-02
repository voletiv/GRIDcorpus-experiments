# BEST LIP READER MODEL in https://arxiv.org/abs/1601.08188
import numpy as np

from grid_params import *
from best_lipreader_functions import *

########################################
# PARAMS
########################################

test_split = 0.2

load_data = False

batch_size = 128

num_of_epochs = 100

########################################
# Load data: dirs, word_numbers, word_idx
########################################

train_test_data = load_GRIDcorpus_speakers_data(TRAIN_VAL_SPEAKERS_LIST)

si_data = load_GRIDcorpus_speakers_data(SI_SPEAKERS_LIST)


########################################
# Split train_test into train and test
########################################

train_data, test_data = split_train_and_test_data(train_test_data, test_split)

########################################
# Load or Generate data
########################################

if load_data is True:
    ########################################
    # Load data
    ########################################
    # Train
    train_mouth_features, train_one_hot_words \
        = load_GRIDcorpus_mouth_features_and_one_hot_words(train_data)
    # Test
    test_mouth_features, train_one_hot_words \
        = load_GRIDcorpus_mouth_features_and_one_hot_words(test_data)
    # Speaker-Independent
    si_mouth_features, si_one_hot_words \
        = load_GRIDcorpus_mouth_features_and_one_hot_words(si_data)
else:
    ########################################
    # To generate data
    ########################################
    # Train
    gen_train_mouth_features_and_one_hot_words \
        = gen_GRIDcorpus_batch_mouth_features_and_one_hot_words(train_data, batch_size)
    train_steps = len(train_data["dirs"]) // batch_size
    # Test
    gen_test_mouth_features_and_one_hot_words \
        = gen_GRIDcorpus_batch_mouth_features_and_one_hot_words(test_data, batch_size)
    test_steps = len(test_data["dirs"]) // batch_size
    # Speaker-Independent
    gen_si_mouth_features_and_one_hot_words \
        = gen_GRIDcorpus_batch_mouth_features_and_one_hot_words(si_data, batch_size)
    si_steps = len(si_data["dirs"]) // batch_size

########################################
# Make Lipreader Model
########################################

LSTM_lipreader, LSTM_lipreader_encoder = make_best_LSTM_lipreader_model()

########################################
# Callbacks
########################################

early_stop = EarlyStopping(patience=10, verbose=1)

check_si_and_make_plots = CheckSIAndMakePlots(
    GRID_DATA_DIR, gen_si_mouth_features_and_one_hot_words, 1,
    file_name_pre="best-LSTM-lipreader")

########################################
# Train Lipreader Model
########################################

LSTM_lipreader_history \
    = LSTM_lipreader.fit_generator(gen_train_mouth_features_and_one_hot_words,
                                   steps_per_epoch=2,
                                   epochs=num_of_epochs, verbose=True,
                                   callbacks=[
                                       check_si_and_make_plots, early_stop],
                                   validation_data=gen_test_mouth_features_and_one_hot_words,
                                   validation_steps=1)
