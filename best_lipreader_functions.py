# MAKE WORDS AS FRAME OUTPUTS FOR MANY-TO-MANY SETTING

import cv2
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm

from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import Input, Masking, Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from grid_params import *

#############################################################
# MAKE BEST LSTM LIPREADER MODEL
#############################################################


def make_best_LSTM_lipreader_model(fc1_units=128,
                                   LSTM1_units=128,
                                   LSTM2_units=128,
                                   vocab_size=51):
    # According to https://arxiv.org/abs/1601.08188
    # Input
    my_input = Input(shape=(MAX_FRAMES_PER_WORD, NUM_OF_MOUTH_PIXELS))
    # Masking
    x = Masking(mask_value=0.0)(my_input)
    # One feed-forward layer
    x = TimeDistributed(Dense(fc1_units,
                              kernel_initializer='random_uniform',
                              bias_initializer='random_uniform'))(x)
    # LSTM 1
    x = LSTM(LSTM1_units, return_sequences=True,
             kernel_initializer='random_uniform',
             bias_initializer='random_uniform')(x)
    # LSTM 2
    encoded = LSTM(LSTM2_units, return_sequences=False,
                   kernel_initializer='random_uniform',
                   bias_initializer='random_uniform')(x)
    # Outputs
    my_output = Dense(vocab_size, activation='softmax',
                      kernel_initializer='random_uniform',
                      bias_initializer='random_uniform')(encoded)
    # Model
    LSTM_lipreader = Model(inputs=my_input, outputs=my_output)
    LSTM_lipreader_encoder = Model(inputs=my_input, outputs=encoded)
    # Compile
    optim = SGD(lr=0.02)
    LSTM_lipreader.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Return
    return LSTM_lipreader, LSTM_lipreader_encoder

#############################################################
# CALLBACKS
#############################################################


class CheckSIAndMakePlots(Callback):

    # Init
    def __init__(self, GRID_DATA_DIR, gen_si_mouth_features_and_one_hot_words,
                 si_steps, file_name_pre="best-LSTM-lipreader"):
        self.GRID_DATA_DIR = GRID_DATA_DIR
        self.gen_si = gen_si_mouth_features_and_one_hot_words
        self.si_steps = si_steps
        self.file_name_pre = file_name_pre

    # On train start
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.test_losses = []
        self.si_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.si_accuracies = []

    # At every epoch
    def on_epoch_end(self, epoch, logs={}):

        # Get
        tl = logs.get('loss')
        ta = logs.get('acc')
        vl = logs.get('val_loss')
        va = logs.get('val_acc')
        print(tl, ta, vl, va)

        # Speaker-Independent
        print("Calculating speaker-independent loss and acc...")
        sil, sia = self.calc_sil_and_sia()
        print(sil, sia)

        # Append losses and accs
        self.train_losses.append(tl)
        self.test_losses.append(vl)
        self.si_losses.append(sil)
        self.train_accuracies.append(ta)
        self.test_accuracies.append(va)
        self.si_accuracies.append(sia)

        # Save model
        self.save_model_checkpoint(epoch, tl, ta, vl, va, sil, sia)

        # Plot graphs
        self.plot_and_save_losses_and_accuracies(epoch)

    # Calculate speaker-independent loss and accuracy
    def calc_sil_and_sia(self):
        if self.gen_si is not None:
            [sil, sia] = self.model.evaluate_generator(
                self.gen_si, self.si_steps)
        else:
            sil, sia = -1, -1
        return sil, sia

    # Save model checkpoint
    def save_model_checkpoint(self, epoch, tl, ta, vl, va, sil, sia):
        model_file_path = os.path.join(self.GRID_DATA_DIR,
            self.file_name_pre + "-epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-sil{5:.4f}-sia{6:.4f}.hdf5".format(epoch, tl, ta, vl, va, sil, sia))
        print("Saving model", model_file_path)
        self.model.save_weights(model_file_path)

    # Plot and save losses and accuracies
    def plot_and_save_losses_and_accuracies(self, epoch):
        print("Saving plots for epoch " + str(epoch))
        plt.subplot(121)
        plt.plot(self.train_losses, label='train_loss', c='C0', linestyle='--')
        plt.plot(self.test_losses, label='val_loss', c='C0', linestyle='-')
        plt.plot(self.si_losses, label='si_loss', c='C0', linestyle='-.')
        leg = plt.legend(loc='best', fontsize=11, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title("Loss")
        plt.subplot(122)
        plt.plot(self.train_accuracies, label='train_acc',
                 c='C0', linestyle='--')
        plt.plot(self.test_accuracies, label='test_acc', c='C0', linestyle='-')
        plt.plot(self.si_accuracies, label='si_acc', c='C0', linestyle='-.')
        leg = plt.legend(loc='best', fontsize=11, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.tick_params(axis='y', which='both',
                        labelleft='on', labelright='on')
        plt.gca().yaxis.grid(True)
        plt.title("Accuracy")
        plt.tight_layout()
        # plt.subplots_adjust(top=0.85)
        plt.suptitle(self.file_name_pre, fontsize=10)
        plt.savefig(os.path.join(self.GRID_DATA_DIR,
                                 self.file_name_pre + "-Plots.png"))
        plt.close()

#############################################################
# GENERATE FEATURES AND ONE_HOT_WORDS
#############################################################


def gen_GRIDcorpus_batch_mouth_features_and_one_hot_words(data,
                                                          batch_size=128,
                                                          shuffle=True):
    # Data
    dirs = data["dirs"]
    word_numbers = data["word_numbers"]
    word_idx = data["word_idx"]

    # Init
    all_dirs = np.array(dirs)
    all_word_numbers = np.array(word_numbers)
    all_word_idx = np.array(word_idx)

    # Full index list for shuffling
    if shuffle is True:
        np.random.seed(29)
        fullIdx = list(range(len(all_dirs)))

    # Looping for generation
    while 1:

        # Shuffling input list
        if shuffle is True:
            np.random.shuffle(fullIdx)
            all_dirs = all_dirs[fullIdx]
            all_word_numbers = all_word_numbers[fullIdx]
            all_word_idx = all_word_idx[fullIdx]

        # For each batch of batch_size number of words:
        for batch_index in range(0, len(all_dirs), batch_size):

            # Batch outputs
            batch_mouth_features = np.zeros(
                (batch_size, MAX_FRAMES_PER_WORD, NUM_OF_MOUTH_PIXELS))
            batch_one_hot_words = np.zeros((batch_size, GRID_VOCAB_SIZE))

            # Batch inputs
            batch_dirs = all_dirs[batch_index:batch_index + batch_size]
            batch_word_numbers = all_word_numbers[
                batch_index:batch_index + batch_size]
            batch_word_idx = all_word_idx[batch_index:batch_index + batch_size]

            # For each data point in batch
            for i, (vid_dir, word_num, word_index) in enumerate(zip(batch_dirs, batch_word_numbers, batch_word_idx)):

                # GET SEQUENCE OF MOUTH IMAGES
                # align file
                alignFile = vid_dir[:-1] + '.align'
                # Word-Time data
                wordTimeData = open(alignFile).readlines()
                # Get the max time of the video
                maxClipDuration = float(wordTimeData[-1].split(' ')[1])
                # Remove Silent and Short Pauses
                for line in wordTimeData:
                    if 'sil' in line or 'sp' in line:
                        wordTimeData.remove(line)
                # Find the start and end frame for this word
                wordStartFrame = math.floor(int(wordTimeData[word_num].split(' ')[
                                            0]) / maxClipDuration * FRAMES_PER_VIDEO)
                wordEndFrame = math.floor(int(wordTimeData[word_num].split(' ')[
                                          1]) / maxClipDuration * FRAMES_PER_VIDEO)
                # All mouth file names of video
                mouthFiles = sorted(
                    glob.glob(os.path.join(vid_dir, '*Mouth*.jpg')))
                # Note the file names of the word
                wordMouthFiles = mouthFiles[
                    wordStartFrame:wordEndFrame + 1]
                # Initialize the array of images for this word
                wordImages = np.zeros(
                    (MAX_FRAMES_PER_WORD, NUM_OF_MOUTH_PIXELS))
                # For each frame of this word
                for f, wordMouthFrame in enumerate(wordMouthFiles[:MAX_FRAMES_PER_WORD]):
                    # in reverse order of frames. eg. If there are 7 frames:
                    # 0 0 0 0 0 0 0 7 6 5 4 3 2 1
                    wordImages[-f - 1] = np.reshape(cv2.imread(wordMouthFrame,
                                                               0) / 255., (NUM_OF_MOUTH_PIXELS,))

                # MAKE FEATURES
                batch_mouth_features[i] = wordImages

                # MAKE ONE HOT WORDS
                batch_one_hot_words[i][word_index] = 1

            # Yield
            yield batch_mouth_features, batch_one_hot_words


#############################################################
# LOAD FEATURES AND ONE_HOT_WORDS
#############################################################


def load_GRIDcorpus_mouth_features_and_one_hot_words(data):
    # Data
    dirs = data["dirs"]
    word_numbers = data["word_numbers"]
    word_idx = data["word_idx"]

    # Outputs
    mouth_features = np.zeros(
        (len(dirs), MAX_FRAMES_PER_WORD, NUM_OF_MOUTH_PIXELS))
    one_hot_words = np.zeros((len(dirs), GRID_VOCAB_SIZE))

    # For each data point
    for i, (vid_dir, word_num, word_index) in tqdm.tqdm(enumerate(zip(dirs, word_numbers, word_idx)), total=len(dirs)):

        # GET SEQUENCE OF MOUTH IMAGES
        # align file
        alignFile = vid_dir[:-1] + '.align'
        # Word-Time data
        wordTimeData = open(alignFile).readlines()
        # Get the max time of the video
        maxClipDuration = float(wordTimeData[-1].split(' ')[1])
        # Remove Silent and Short Pauses
        for line in wordTimeData:
            if 'sil' in line or 'sp' in line:
                wordTimeData.remove(line)
        # Find the start and end frame for this word
        wordStartFrame = math.floor(int(wordTimeData[word_num].split(' ')[
                                    0]) / maxClipDuration * FRAMES_PER_VIDEO)
        wordEndFrame = math.floor(int(wordTimeData[word_num].split(' ')[
                                  1]) / maxClipDuration * FRAMES_PER_VIDEO)
        # All mouth file names of video
        mouthFiles = sorted(
            glob.glob(os.path.join(vid_dir, '*Mouth*.jpg')))
        # Note the file names of the word
        wordMouthFiles = mouthFiles[
            wordStartFrame:wordEndFrame + 1]
        # Initialize the array of images for this word
        wordImages = np.zeros((MAX_FRAMES_PER_WORD, NUM_OF_MOUTH_PIXELS))
        # For each frame of this word
        for f, wordMouthFrame in enumerate(wordMouthFiles[:MAX_FRAMES_PER_WORD]):
            # in reverse order of frames. eg. If there are 7 frames:
            # 0 0 0 0 0 0 0 7 6 5 4 3 2 1
            wordImages[-f - 1] = np.reshape(cv2.imread(wordMouthFrame,
                                                       0) / 255., (NUM_OF_MOUTH_PIXELS,))

        # MAKE FEATURES
        mouth_features[i] = wordImages

        # MAKE ONE HOT WORDS
        one_hot_words[i][word_index] = 1

    # Return
    return mouth_features, one_hot_words

#############################################################
# SPLIT TRAIN AND TEST DATA
#############################################################


def split_train_and_test_data(train_test_data, test_split):
    # Choose some data points of train_test as test data
    test_idx = np.sort(np.random.choice(len(train_test_data["dirs"]), int(
        test_split * len(train_test_data["dirs"])), replace=False))

    test_dirs = train_test_data["dirs"][test_idx]
    test_word_numbers = train_test_data["word_numbers"][test_idx]
    test_word_idx = train_test_data["word_idx"][test_idx]

    # Make the rest as train_data
    train_idx = np.delete(np.arange(len(train_test_data["dirs"])), test_idx)

    train_dirs = train_test_data["dirs"][train_idx]
    train_word_numbers = train_test_data["word_numbers"][train_idx]
    train_word_idx = train_test_data["word_idx"][train_idx]

    train_data = {}
    train_data["dirs"] = train_dirs
    train_data["word_numbers"] = train_word_numbers
    train_data["word_idx"] = train_word_idx

    test_data = {}
    test_data["dirs"] = test_dirs
    test_data["word_numbers"] = test_word_numbers
    test_data["word_idx"] = test_word_idx

    return train_data, test_data

#############################################################
# LOAD SPEAKER_DIRS, WORD_NUMBERS, WORDS
#############################################################


def load_GRIDcorpus_speakers_data(speakers_list=TRAIN_VAL_SPEAKERS_LIST):
    # TRAIN AND VAL
    dirs = []
    word_numbers = []
    word_idx = []

    # For each speaker
    for speaker in tqdm.tqdm(sorted((speakers_list))):
        speaker_dir = os.path.join(
            GRID_DATA_DIR, 's' + '{0:02d}'.format(speaker))

        # List of all videos for each speaker
        vid_dirs_list = sorted(glob.glob(os.path.join(speaker_dir, '*/')))

        # Append training directories
        for vid_dir in vid_dirs_list:
            # Words
            align_file = vid_dir[:-1] + '.align'
            words = []
            with open(align_file) as f:
                for line in f:
                    if 'sil' not in line and 'sp' not in line:
                        words.append(line.rstrip().split()[-1])
            # Append
            for word_num in range(WORDS_PER_VIDEO):
                # print(word_num)
                dirs.append(vid_dir)
                word_numbers.append(word_num)
                word_idx.append(GRID_VOCAB.index(words[word_num]))

    data = {}
    data["dirs"] = np.array(dirs)
    data["word_numbers"] = np.array(word_numbers)
    data["word_idx"] = np.array(word_idx)

    # Return
    return data
