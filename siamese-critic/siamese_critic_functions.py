import glob
import numpy as np
import os
import random as rn
import tensorflow as tf
import tqdm

# with tf.device('/cpu:0'):
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense

# Import params
from siamese_critic_params import *
from LSTM_lipreader_function import *

######################################################################
# Simaese Critic MODEL function with all inputs and 1 output Hidden layer
######################################################################


def simaese_critic():
    
    # Video input
    videoInput = Input(shape=(MOUTH_W, MOUTH_H, FRAMES_PER_WORD,))

    # Video process
    videoNetwork = vgg_m_network()
    videoFeatures = videoNetwork(vidInput)

    # Word input
    wordInput = Input(shape=(GRID_VOCAB_SIZE,))

    # Word process
    wordNetwork = word_network()
    wordFeatures = wordNetwork(wordInput)

    distance = Lambda(euclidean_distance,
        output_shape=eucl_dist_output_shape)([videoFeatures, wordFeatures])

    model = Model([videoInput, wordInput], distance)

    model.compile(loss=contrastive_loss,
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def vgg_m_network(fc7=256):
    # Manual seeds
    os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
    np.random.seed(29)
    rn.seed(29)
    tf.set_random_seed(29)
    # MODEL
    model = Sequential()
    # 40x40x30
    model.add(Conv2D(96, 3, padding='same', name='conv1'),
                     input_shape=(MOUTH_W, MOUTH_H, FRAMES_PER_WORD,))
    # 40x40x96
    model.add(Conv2D(256, 3, padding='same', name='conv2'))
    # 40x40x256
    model.add(MaxPooling2D(3, name='pool2'))
    # 14x14x256
    model.add(Conv2D(512, 3, padding='same', name='conv3'))
    # 14x14x512
    model.add(Conv2D(512, 3, padding='same', name='conv4'))
    model.add(Conv2D(512, 3, padding='same', name='conv5'))
    model.add(MaxPooling2D(3, padding='same', name='pool5'))
    # 5x5x512
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc6'))
    # 4096
    model.add(Dense(fc7, activation='softmax', name='fc7'))
    # 256
    return model


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def word_network(fc1=300, fc2=256):
    # Manual seeds
    os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
    np.random.seed(29)
    rn.seed(29)
    tf.set_random_seed(29)
    # MODEL
    model = Sequential()
    model.add(Dense(fc1, activation='relu', name='word_fc1'))
    model.add(Dense(fc2, activation='relu', name='word_fc2'))
    return model


######################################################################
# Creating Word Pairs
######################################################################

def make_GRIDcorpus_vids_and_one_hot_words_siamese_pairs(dirs,
                                                         word_numbers,
                                                         word_idx,
                                                         LSTMLipreaderEncoder):
    features = np.zeros((len(dirs), LIPREADER_ENCODED_DIM))
    one_hot_words = np.zeros((len(dirs), GRID_VOCAB_SIZE))
    # For each data point
    for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(dirs, word_numbers, word_idx)), total=len(dirs)):
        # GET SEQUENCE OF MOUTH IMAGES
        # align file
        alignFile = vidDir[:-1] + '.align'
        # Word-Time data
        wordTimeData = open(alignFile).readlines()
        # Get the max time of the video
        maxClipDuration = float(wordTimeData[-1].split(' ')[1])
        # Remove Silent and Short Pauses
        for line in wordTimeData:
            if 'sil' in line or 'sp' in line:
                wordTimeData.remove(line)
        # Find the start and end frame for this word
        wordStartFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
                                    0]) / maxClipDuration * FRAMES_PER_VIDEO)
        wordEndFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
                                  1]) / maxClipDuration * FRAMES_PER_VIDEO)
        # All mouth file names of video
        mouthFiles = sorted(glob.glob(os.path.join(vidDir, '*Mouth*.jpg')))
        # Note the file names of the word
        wordMouthFiles = mouthFiles[
            wordStartFrame:wordEndFrame + 1]
        # Initialize the array of images for this word
        wordImages = np.zeros((1, FRAMES_PER_WORD, NUM_OF_MOUTH_PIXELS))
        # For each frame of this word
        for f, wordMouthFrame in enumerate(wordMouthFiles[:FRAMES_PER_WORD]):
            # in reverse order of frames. eg. If there are 7 frames:
            # 0 0 0 0 0 0 0 7 6 5 4 3 2 1
            wordImages[0][-f - 1] = np.reshape(cv2.imread(wordMouthFrame,
                                                          0) / 255., (NUM_OF_MOUTH_PIXELS,))
        # MAKE FEATURES
        features[i] = LSTMLipreaderEncoder.predict(wordImages)
        # MAKE ONE HOT WORDS
        one_hot_words[i][wordIndex] = 1
    # Return
    return features, one_hot_words




#############################################################
# LOAD SPEAKER_DIRS, WORD_NUMBERS, WORDS
#############################################################


def load_GRIDcorpus_speakers_dirs_wordnums_wordidx_lists(
    speakers_list=TRAIN_VAL_SPEAKERS_LIST
):
    # TRAIN AND VAL
    all_dirs = []
    all_word_numbers = []
    all_wordidx = []
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
                # print(wordNum)
                all_dirs.append(vid_dir)
                all_word_numbers.append(word_num)
                all_wordidx.append(GRID_VOCAB.index(words[word_num]))
    # Return
    return np.array(all_dirs), np.array(all_word_numbers), np.array(all_wordidx)


#############################################################
# LOAD LSTM LIP READER MODEL
#############################################################


def load_LSTM_lipreader_and_encoder():
    LSTMLipReaderModel, LSTMLipreaderEncoder, _ = make_LSTM_lipreader_model()
    lipreader_filelist = os.listdir(os.path.join(GRID_DATA_DIR, "TRAINED-WEIGHTS"))
    for file in lipreader_filelist:
        if ('lipreader' in file or 'Lipreader' in file or 'LipReader' in file or 'LIPREADER' in file) and '.hdf5' in file:
            LIPREADER_MODEL_FILE = os.path.join(GRID_DATA_DIR, "TRAINED-WEIGHTS", file)
    LSTMLipReaderModel.load_weights(LIPREADER_MODEL_FILE)
    return LSTMLipReaderModel, LSTMLipreaderEncoder


def make_LSTM_lipreader_model():
    useMask = True
    hiddenDim = 256
    depth = 2
    LSTMactiv = 'tanh'
    encodedDim = 64
    encodedActiv = 'relu'
    optimizer = 'adam'
    lr = 1e-3
    # Make model
    LSTMLipReaderModel, LSTMEncoder, fileNamePre \
        = LSTM_lipreader(useMask=useMask, hiddenDim=hiddenDim, depth=depth,
                         LSTMactiv=LSTMactiv, encodedDim=encodedDim,
                         encodedActiv=encodedActiv, optimizer=optimizer, lr=lr)
    return LSTMLipReaderModel, LSTMEncoder, fileNamePre

######################################################################
# Dependent functions
######################################################################
