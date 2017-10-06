from siamese_critic_params import *
from siamese_critic_functions import *

########################################
# Load LipReader
########################################

LSTMLipreaderModel, LSTMLipreaderEncoder = load_LSTM_lipreader_and_encoder()

########################################
# Get FULL Data
########################################

train_val_dirs, train_val_word_numbers, train_val_word_idx, \
    = load_GRIDcorpus_speakers_dirs_wordnums_wordidx_lists(TRAIN_VAL_SPEAKERS_LIST)

si_dirs, si_word_numbers, si_word_idx \
    = load_GRIDcorpus_speakers_dirs_wordnums_wordidx_lists(SI_SPEAKERS_LIST)

# create training+test positive and negative pairs
train_val_word_indices = [np.where(train_val_word_idx == i)[0] for i in range(GRID_VOCAB_SIZE)]

n = min([len(train_val_word_indices[d]) for d in range(GRID_VOCAB_SIZE)]) - 1



train_val_pairs, train_val_y = create_pairs(train_val_dirs, train_val_word_numbers, train_val_word_indices)



si_word_indices = [np.where(train_val_word_idx == i)[0] for i in range(GRID_VOCAB_SIZE)]
te_pairs, te_y = create_pairs(x_test, digit_indices)




########################################
# Make FULL features and one_hot_words
########################################

train_val_features, train_val_one_hot_words \
    = make_GRIDcorpus_features_and_one_hot_words_siamese_pairs(
        train_val_dirs, train_val_word_numbers, train_val_word_idx,
        LSTMLipreaderEncoder)

si_features, si_one_hot_words = make_GRIDcorpus_features_and_one_hot_words(
    si_dirs, si_word_numbers, si_word_idx, LSTMLipreaderEncoder)

