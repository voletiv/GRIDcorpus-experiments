import cv2
import os
import numpy as np
import glob
import tqdm
import math
import matplotlib
# matplotlib.use('agg')     # Use this for remote terminals
import matplotlib.pyplot as plt
import random as rn

from matplotlib.patches import Rectangle
from imutils.face_utils import FaceAligner, shape_to_np
import dlib

import tensorflow as tf
# with tf.device('/cpu:0'):
from keras.models import Model, Sequential, load_model, model_from_yaml
from keras.layers import Input, Activation
# https://github.com/farizrahman4u/seq2seq
from seq2seq.models import SimpleSeq2Seq
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Reshape, Conv3D, BatchNormalization, Lambda, MaxPooling3D, Flatten, Dense, concatenate
from keras.layers import LSTM, RepeatVector

#############################################################
# PARAMS TO BE SET
#############################################################

rootDir = '/home/voletiv/Downloads/GRIDcorpus'
saveDir = rootDir
# rootDir = '/Neutron6/voleti.vikram/GRIDcorpus'
# saveDir = '/Neutron6/voleti.vikram/GRIDcorpusResults/1-h32-d1-Adam-2e-4-s0107-s0920-s2227'

trainAlign = False
valAlign = False
useMeanMouthImage = False
# Mean mouth image
meanMouthImageFile = os.path.join(rootDir, 'meanMouthImage_Vikram.npy')

#############################################################
# TRUE PARAMS
#############################################################

LRfeaturesDim = 64
batchSize = 128

framesPerVid = 75
wordsPerVideo = 6
framesPerWord = 14
nOfMouthPixels = 1600
mouthW = 40
mouthH = 40
nOfUniqueWords = 53     # including silent and short pause
# excluding 'sil' and 'sp', +1 for padding
wordsVocabSize = (nOfUniqueWords - 2) + 1

# Unique Words Idx
uniqueWordsFile = os.path.join(rootDir, 'uniqueWords.npy')
uniqueWords = np.load(uniqueWordsFile)
# num of unique words, including 'sil' and 'sp'
nOfUniqueWords = len(uniqueWords)
# Remove silent
uniqueWords = np.delete(
    uniqueWords, np.argwhere(uniqueWords == 'sil'))
# Remove short pauses
uniqueWords = np.delete(
    uniqueWords, np.argwhere(uniqueWords == 'sp'))
# Vocabulary size
# excluding 'sil' and 'sp', +1 for padding
wordsVocabSize = len(uniqueWords) + 1
# Word indices
wordIdx = {}
for i, word in enumerate(uniqueWords):
    wordIdx[word] = i

dummyMeanMouthImage = np.zeros((nOfMouthPixels,))

#############################################################
# MAKE WORDS AS FRAME OUTPUTS FOR MANY-TO-MANY SETTING
#############################################################


def genMouthImages(allDirs, batchSize, align, wordIdx, wordsVocabSize=52, wordsPerVideo=6, framesPervid=75, framesPerWord=14, nOfMouthPixels=1600,
                   useMeanMouthImage=False, meanMouthImage=dummyMeanMouthImage, shuffle=True, shuffleWords=False, keepPadResults=True):
    np.random.seed(29)
    dirs = np.array(allDirs)
    # Looping generator
    while 1:
        # Shuffling input list
        if shuffle is True:
            np.random.shuffle(dirs)
        # For each slice of batchSize number of files:
        for batch in range(0, len(dirs), batchSize):
            # print("batch " + str(batch))
            # Initializing output variables
            X = np.zeros((batchSize * wordsPerVideo,
                          framesPerWord, nOfMouthPixels))
            if keepPadResults:
                y = np.zeros((batchSize * wordsPerVideo, 2, wordsVocabSize))
            else:
                y = np.zeros((batchSize * wordsPerVideo, wordsVocabSize))
            # For each video in the batch
            batchDirs = dirs[batch:batch + batchSize]
            # print(batchDirs[:10])
            # If at the end, there are lesser number of video files than
            # batchSize
            if len(batchDirs) < batchSize:
                continue
            # Else
            for vid, vidDir in enumerate(batchDirs):
                # print("  vid " + + str(vid) + " " + str(vidDir))
                # Get the file names of the mouth images
                # If aligned mouth images are needed
                if align:
                    mouthFiles = sorted(glob.glob(
                        os.path.join(vidDir, '*Aligned*.jpg')))
                # If non-aligned mouth images are needed
                else:
                    mouthFiles = sorted(
                        glob.glob(os.path.join(vidDir, '*Mouth*.jpg')))
                # Get the align file with the words-time data
                alignFile = vidDir[:-1] + '.align'
                # Get the words-time data
                wordTimeData = open(alignFile).readlines()
                # Get the max time of the video
                maxClipDuration = float(wordTimeData[-1].split(' ')[1])
                # Remove Silent and Short Pauses
                for line in wordTimeData:
                    if 'sil' in line or 'sp' in line:
                        wordTimeData.remove(line)
                # For each word, excluding the "silent" words and "short
                # pauses"; len(wordTimeData) = 6
                for word in range(len(wordTimeData)):
                    # print("    word " + str(word))
                    # Extract the word and save this word in y using one-hot
                    # encoding
                    if keepPadResults:
                        y[vid * wordsPerVideo + word][0] = np_utils.to_categorical(
                            wordIdx[wordTimeData[word].split(' ')[-1][:-1]], wordsVocabSize)
                    else:
                        y[vid * wordsPerVideo + word] = np_utils.to_categorical(
                            wordIdx[wordTimeData[word].split(' ')[-1][:-1]], wordsVocabSize)
                    # Initialize the array of images for this word
                    wordImages = np.zeros((framesPerWord, nOfMouthPixels))
                    # Find the start and end frame for this word
                    wordStartFrame = math.floor(int(wordTimeData[word].split(' ')[
                                                0]) / maxClipDuration * framesPerVid)
                    wordEndFrame = math.floor(int(wordTimeData[word].split(' ')[
                                             1]) / maxClipDuration * framesPerVid)
                    # Note the file names
                    wordMouthFiles = mouthFiles[
                        wordStartFrame:wordEndFrame + 1]
                    # # if len(wordMouthFiles) > 14:
                    # excessMouthLens.append(len(wordMouthFiles))
                    # excessMouthLenNames.append(wordMouthFiles[0][:-11] + "_word" + str(word))
                    # For blanks
                    if keepPadResults and len(wordMouthFiles) < 14:
                        y[vid * wordsPerVideo + word][1][-1] = 1
                    # For each frame of this word
                    for f, wordMouthFrame in enumerate(wordMouthFiles[:framesPerWord]):
                        # Note the corresponding mouth image in greyscale,
                        # (padded with zeros before the frames, if len(wordMouthFiles) < framesPerWord)
                        wordImages[f + (framesPerWord-(min(len(wordMouthFiles), framesPerWord)))] \
                            = np.reshape(cv2.imread(wordMouthFrame, 0) / 255., (1600,))
                        # subtracted by the mean image
                        if useMeanMouthImage:
                            wordImages[f] -= meanMouthImage
                    # Save this in X
                    X[vid * wordsPerVideo + word] = wordImages
            # Shuffle the results
            if shuffleWords is True:
                fullIdx = list(range(len(X)))
                np.random.shuffle(fullIdx)
                X = X[fullIdx]
                y = y[fullIdx]
            # Yield the results
            yield (X, y)

#############################################################
# LOAD STUFF
#############################################################

# Read meanMouthImage
meanMouthImage = np.load(meanMouthImageFile)

# For each speaker
# Select which speaker to train & validate from
# speakersList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16,
# 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
speakersList = [1, 2, 3, 4, 5, 6, 7, 9]
# speakersList = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# TRAIN AND VAL
valSplit = 0.1
trainDirs = []
valDirs = []
np.random.seed(29)
# For each speaker
for speaker in sorted(tqdm.tqdm(speakersList)):
    speakerDir = os.path.join(rootDir, 's' + '{0:02d}'.format(speaker))
    # List of all videos for each speaker
    vidDirs = sorted(glob.glob(os.path.join(speakerDir, '*/')))
    totalNumOfImages = len(vidDirs)
    # To shuffle directories before splitting into train and validate
    fullListIdx = list(range(totalNumOfImages))
    np.random.shuffle(fullListIdx)
    # Append training directories
    for i in fullListIdx[:int((1 - valSplit) * totalNumOfImages)]:
        trainDirs.append(vidDirs[i])
    # Append val directories
    for i in fullListIdx[int((1 - valSplit) * totalNumOfImages):]:
        valDirs.append(vidDirs[i])

# Numbers
print("No. of training videos: " + str(len(trainDirs)))
print("No. of val videos: " + str(len(valDirs)))

# SPEAKER INDEPENDENT
siList = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
siDirs = []
for speaker in sorted(tqdm.tqdm(siList)):
    speakerDir = os.path.join(rootDir, 's' + '{0:02d}'.format(speaker))
    vidDirs = sorted(glob.glob(os.path.join(speakerDir, '*/')))
    for i in fullListIdx:
            siDirs.append(vidDirs[i])

siValSteps = int(len(siDirs) / batchSize)

# Numbers
print("No. of speaker-independent videos: " + str(len(siDirs)))

#############################################################
# Simple LSTM LipReader MODEL with Encoding
#############################################################

hiddenDim = 256
depth = 2
LSTMactiv = 'tanh'
encodedDim = 64
encodedActiv = 'sigmoid'

vidInput = Input(shape=(framesPerWord, nOfMouthPixels,))

# If depth > 1
if depth > 1:
    # First layer
    encoded = LSTM(hiddenDim, activation=LSTMactiv,
                   return_sequences=True)(vidInput)
    for d in range(depth - 2):
        encoded = LSTM(hiddenDim, activation=LSTMactiv,
                       return_sequences=True)(encoded)
    # Last layer
    encoded = LSTM(hiddenDim, activation=LSTMactiv)(encoded)
# If depth = 1
else:
    encoded = LSTM(hiddenDim, activation=LSTMactiv)(vidInput)

encoded = Dense(encodedDim, activation=encodedActiv)(encoded)

encoder = Model(inputs=vidInput, outputs=encoded)

myWord = Dense(wordsVocabSize, activation='softmax')(encoded)
myWord = Reshape((1, wordsVocabSize))(myWord)
myPad = Dense(wordsVocabSize, activation='softmax')(encoded)
myPad = Reshape((1, wordsVocabSize))(myPad)

myOutput = concatenate([myWord, myPad], axis=1)

LSTMLipReaderModel = Model(inputs=vidInput, outputs=myOutput)

lr = 1e-3
adam = Adam(lr=lr)
LSTMLipReaderModel.compile(optimizer=adam, loss='categorical_crossentropy',
                           metrics=['accuracy'])
LSTMLipReaderModel.summary()

filenamePre = 'LSTM-h' + \
    str(hiddenDim) + '-depth' + str(depth) + '-LSTMactiv' + LSTMactiv + \
    '-enc' + str(encodedDim) + '-encodedActiv' + encodedActiv + \
    '-Adam-%1.e' % lr + '-GRIDcorpus-s'
print(filenamePre)

LSTMLipReaderModel.load_weights(os.path.join(saveDir, "LSTM-h256-depth2-LSTMactivtanh-enc64-encodedActivsigmoid-Adam-1e-03-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub-epoch099-tl0.3307-ta0.8417-vl0.3782-va0.8304.hdf5"))

######################################################################
# C3D Critic MODEL function with all inputs and 1 output Hidden layer
######################################################################


def C3DCriticModel(
        layer1Filters=8,
        layer2Filters=16,
        layer3Filters=32,
        fc1Nodes=64,
        vidFeaturesDim=64,
        useWord=False,
        wordDim=1,
        useEncWord=False,
        encWordDim=64,
        useEncWordFc=False,
        encWordFcDim=10,
        useOneHotWord=False,
        oneHotWordDim=52,
        useOneHotWordFc=False,
        oneHotWordFcDim=10,
        usePredWordDis=False,
        predWordDisDim=52,
        outputHDim=64,
        lr=5e-4
    ):
    # Manual seeds
    os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
    np.random.seed(29)
    rn.seed(29)
    tf.set_random_seed(29)
    vidInput = Input(shape=(framesPerWord, nOfMouthPixels,))
    # 0 1 2 3  2 3 4 5  4 5 6 7  6 7 8 9  8 9 10 11  10 11 12 13
    i1 = Lambda(lambda x: x[:, 0:4, :])(vidInput)
    i2 = Lambda(lambda x: x[:, 2:6, :])(vidInput)
    i3 = Lambda(lambda x: x[:, 4:8, :])(vidInput)
    i4 = Lambda(lambda x: x[:, 6:10, :])(vidInput)
    i5 = Lambda(lambda x: x[:, 8:12, :])(vidInput)
    i6 = Lambda(lambda x: x[:, 10:14, :])(vidInput)
    layer1 = Sequential()
    layer1.add(Reshape((4, mouthW, mouthH, 1), input_shape=(4, nOfMouthPixels)))
    layer1.add(Conv3D(layer1Filters, 3, activation='relu'))
    layer1.add(BatchNormalization())
    layer1.add(MaxPooling3D(pool_size=(2, 2, 2)))
    x1 = layer1(i1)
    x2 = layer1(i2)
    x3 = layer1(i3)
    x4 = layer1(i4)
    x5 = layer1(i5)
    x6 = layer1(i6)
    y1 = concatenate([x1, x2], axis=1)
    y2 = concatenate([x3, x4], axis=1)
    y3 = concatenate([x5, x6], axis=1)
    layer2 = Sequential()
    layer2.add(Conv3D(layer2Filters, (2, 3, 3), activation='relu',
                      input_shape=(y1.get_shape()[1], y1.get_shape()[2], y1.get_shape()[3], layer1Filters)))
    layer2.add(BatchNormalization())
    layer2.add(MaxPooling3D(pool_size=(1, 2, 2)))
    z1 = layer2(y1)
    z2 = layer2(y2)
    z3 = layer2(y3)
    z = concatenate([z1, z2, z3], axis=1)
    layer3 = Sequential()
    layer3.add(Conv3D(layer3Filters, (3, 3, 3), activation='relu',
                      input_shape=(z.get_shape()[1], z.get_shape()[2], z.get_shape()[3], layer2Filters)))
    layer3.add(BatchNormalization())
    z = layer3(z)
    z = Flatten()(z)
    z = Dense(fc1Nodes, activation='relu')(z)
    vidFeatures = Dense(vidFeaturesDim, activation='relu')(z)
    # Other Inputs
    if useWord:
        wordFeatures = Input(shape=(wordDim,))
        print("...using wordFeatures")
    if useEncWord:
        encWordFeatures = Input(shape=(encWordDim,))
        print("...using encWordFeatures")
    if useEncWordFc:
        encWordFcFeatures = Dense(encWordFcDim, activation='relu')(encWordFeatures)
        print("...using encWordFcFeatures")
    if useOneHotWord:
        oneHotWordFeatures = Input(shape=(oneHotWordDim,))
        print("...using oneHotWordFeatures")
    if useOneHotWordFc:
        oneHotWordFcFeatures = Dense(oneHotWordFcDim, activation='relu')(oneHotWordFeatures)
        print("...using oneHotWordFcFeatures")
    if usePredWordDis:
        predWordDis = Input(shape=(predWordDisDim,))
        print("...using predWordDis")
    # Full feature
    if useWord:
        fullFeature = concatenate([vidFeatures, wordFeatures])
        print("...fullFeature wordFeatures")
    if useEncWord and not useEncWordFc and not useOneHotWord:
        fullFeature = concatenate([vidFeatures, encWordFeatures])
        print("...fullFeature encWordFeatures")
    if useEncWord and useEncWordFc and not useOneHotWord:
        fullFeature = concatenate([vidFeatures, encWordFcFeatures])
        print("...fullFeature encWordFeatures, encWordFcFeatures")
    if useOneHotWord and not useOneHotWordFc and not useEncWord:
        fullFeature = concatenate([vidFeatures, oneHotWordFeatures])
        print("...fullFeature oneHotWordFeatures")
    if useOneHotWord and useOneHotWordFc and not useEncWord:
        fullFeature = concatenate([vidFeatures, oneHotWordFcFeatures])
        print("...fullFeature oneHotWordFeatures, oneHotWordFcFeatures")
    if useEncWord and not useEncWordFc and useOneHotWord and not useOneHotWordFc:
        fullFeature = concatenate([vidFeatures, encWordFeatures, oneHotWordFeatures])
        print("...fullFeature encWordFeatures, oneHotWordFeatures")
    if useEncWord and useEncWordFc and useOneHotWord and useOneHotWordFc:
        fullFeature = concatenate([vidFeatures, encWordFcFeatures, oneHotWordFcFeatures])
        print("...fullFeature encWordFeatures, encWordFcFeatures, oneHotWordFeatures, oneHotWordFcFeatures")
    if usePredWordDis:
        fullFeature = concatenate([vidFeatures, predWordDis])
        print("...fullFeature predWordDis")
    # Output
    y = Dense(outputHDim, activation='relu')(fullFeature)
    print("...y")
    myOutput = Dense(1, activation='sigmoid')(y)
    print("...myOutput")
    # Model
    if useWord:
        criticModel = Model(inputs=[vidInput, wordFeatures], outputs=myOutput)
        print("...model wordFeatures")
    if useEncWord and not useOneHotWord:
        criticModel = Model(inputs=[vidInput, encWordFeatures], outputs=myOutput)
        print("...model encWordFeatures")
    if useOneHotWord and not useEncWord:
        criticModel = Model(inputs=[vidInput, oneHotWordFeatures], outputs=myOutput)
        print("...model oneHotWordFeatures")
    if useEncWord and useOneHotWord:
        criticModel = Model(inputs=[vidInput, encWordFeatures, oneHotWordFeatures], outputs=myOutput)
        print("...model encWordFeatures, oneHotWordFeatures")
    if usePredWordDis:
        criticModel = Model(inputs=[vidInput, predWordDis], outputs=myOutput)
        print("...model predWordDis")
    # lr = 5e-4
    adam = Adam(lr=lr)
    criticModel.compile(optimizer=adam, loss='binary_crossentropy',
                        metrics=['accuracy'])
    filenamePre ='C3DCritic-l1f' + str(layer1Filters) + \
    '-l2f' + str(layer2Filters) + \
    '-l3f' + str(layer3Filters) + \
    '-fc1n' + str(fc1Nodes) + \
    '-vid' + str(vidFeaturesDim)
    if useWord:
        filenamePre += '-word' + str(wordDim)
    if useEncWord:
        filenamePre += '-encWord' + str(encWordDim)
    if useOneHotWord:
        filenamePre += '-OHWord' + str(oneHotWordDim)
    if useOneHotWordFc:
        filenamePre += '-OHWordFc' + str(oneHotWordFcDim)
    if usePredWordDis:
        filenamePre += '-predWordDisDim' + str(predWordDisDim)
    filenamePre += '-out' + str(outputHDim) + \
        '-Adam-%1.e' % lr + \
        '-GRIDcorpus-s'
    print(filenamePre)
    return criticModel, filenamePre

######################################################################
# C3D Critic MODEL with WORD input and 1 output Hidden layer
######################################################################

# 35
criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=True, wordDim=1,
                                useEncWord=False, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=False, oneHotWordDim=52, useOneHotWordFc=False, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)

# OR

# Manual seeds
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)

# layer1Filters = 2
# layer2Filters = 2
# layer3Filters = 2
# fc1Nodes = 2
# vidFeaturesDim = 2
# encodedWordDim = 64
# outputHDim = 2

layer1Filters = 8
layer2Filters = 16
layer3Filters = 32
fc1Nodes = 64
vidFeaturesDim = 64
wordDim = 1
outputHDim = 64

vidInput = Input(shape=(framesPerWord, nOfMouthPixels,))
# 0 1 2 3  2 3 4 5  4 5 6 7  6 7 8 9  8 9 10 11  10 11 12 13
i1 = Lambda(lambda x: x[:, 0:4, :])(vidInput)
i2 = Lambda(lambda x: x[:, 2:6, :])(vidInput)
i3 = Lambda(lambda x: x[:, 4:8, :])(vidInput)
i4 = Lambda(lambda x: x[:, 6:10, :])(vidInput)
i5 = Lambda(lambda x: x[:, 8:12, :])(vidInput)
i6 = Lambda(lambda x: x[:, 10:14, :])(vidInput)

layer1 = Sequential()
layer1.add(Reshape((4, mouthW, mouthH, 1), input_shape=(4, nOfMouthPixels)))
layer1.add(Conv3D(layer1Filters, 3, activation='relu'))
layer1.add(BatchNormalization())
layer1.add(MaxPooling3D(pool_size=(2, 2, 2)))

x1 = layer1(i1)
x2 = layer1(i2)
x3 = layer1(i3)
x4 = layer1(i4)
x5 = layer1(i5)
x6 = layer1(i6)

y1 = concatenate([x1, x2], axis=1)
y2 = concatenate([x3, x4], axis=1)
y3 = concatenate([x5, x6], axis=1)

layer2 = Sequential()
layer2.add(Conv3D(layer2Filters, (2, 3, 3), activation='relu',
                  input_shape=(y1.get_shape()[1], y1.get_shape()[2], y1.get_shape()[3], layer1Filters)))
layer2.add(BatchNormalization())
layer2.add(MaxPooling3D(pool_size=(1, 2, 2)))

z1 = layer2(y1)
z2 = layer2(y2)
z3 = layer2(y3)

z = concatenate([z1, z2, z3], axis=1)

layer3 = Sequential()
layer3.add(Conv3D(layer3Filters, (3, 3, 3), activation='relu',
                  input_shape=(z.get_shape()[1], z.get_shape()[2], z.get_shape()[3], layer2Filters)))
layer3.add(BatchNormalization())

z = layer3(z)
z = Flatten()(z)
z = Dense(fc1Nodes, activation='relu')(z)
vidFeatures = Dense(vidFeaturesDim, activation='relu')(z)

wordFeatures = Input(shape=(wordDim,))

fullFeature = concatenate([vidFeatures, wordFeatures])

y = Dense(outputHDim, activation='relu')(fullFeature)

# yWord = Dense(52, activation='sigmoid')(y)
# yWord = Reshape((1, 52))(yWord)
# yPad = Dense(52, activation='sigmoid')(y)
# yPad = Reshape((1, 52))(yPad)
# myOutput = concatenate([yWord, yPad], axis=1)

myOutput = Dense(1, activation='sigmoid')(y)

criticModelWithWord = Model(inputs=[vidInput, wordFeatures], outputs=myOutput)

lr = 5e-4
adam = Adam(lr=lr)
criticModelWithWord.compile(optimizer=adam, loss='binary_crossentropy',
                    metrics=['accuracy'])

criticModelWithWord.summary()

# i = np.zeros((2, 14, 20, 20, 1))
# w = np.zeros((2, 1))
# model.predict([i, w])

filenamePre = 'C3DCritic-l1f' + str(layer1Filters) + \
    '-l2f' + str(layer2Filters) + \
    '-l3f' + str(layer3Filters) + \
    '-fc1n' + str(fc1Nodes) + \
    '-vid' + str(vidFeaturesDim) + \
    '-word' + str(wordDim) + \
    '-out' + str(outputHDim) + \
    '-Adam-%1.e' % lr + \
    '-GRIDcorpus-s'
print(filenamePre)

######################################################################
# C3D Critic MODEL with WordFeatures input and 1 output Hidden layer
######################################################################

# 36a
criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=False, wordDim=1,
                                useEncWord=True, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=False, oneHotWordDim=52, useOneHotWordFc=False, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)

# OR

# Manual seeds
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)

# layer1Filters = 2
# layer2Filters = 2
# layer3Filters = 2
# fc1Nodes = 2
# vidFeaturesDim = 2
# encodedWordDim = 64
# outputHDim = 2

layer1Filters = 8
layer2Filters = 16
layer3Filters = 32
fc1Nodes = 64
vidFeaturesDim = 64
encodedWordDim = 64
outputHDim = 64

vidInput = Input(shape=(framesPerWord, nOfMouthPixels,))
# 0 1 2 3  2 3 4 5  4 5 6 7  6 7 8 9  8 9 10 11  10 11 12 13
i1 = Lambda(lambda x: x[:, 0:4, :])(vidInput)
i2 = Lambda(lambda x: x[:, 2:6, :])(vidInput)
i3 = Lambda(lambda x: x[:, 4:8, :])(vidInput)
i4 = Lambda(lambda x: x[:, 6:10, :])(vidInput)
i5 = Lambda(lambda x: x[:, 8:12, :])(vidInput)
i6 = Lambda(lambda x: x[:, 10:14, :])(vidInput)

layer1 = Sequential()
layer1.add(Reshape((4, mouthW, mouthH, 1), input_shape=(4, nOfMouthPixels)))
layer1.add(Conv3D(layer1Filters, 3, activation='relu'))
layer1.add(BatchNormalization())
layer1.add(MaxPooling3D(pool_size=(2, 2, 2)))

x1 = layer1(i1)
x2 = layer1(i2)
x3 = layer1(i3)
x4 = layer1(i4)
x5 = layer1(i5)
x6 = layer1(i6)

y1 = concatenate([x1, x2], axis=1)
y2 = concatenate([x3, x4], axis=1)
y3 = concatenate([x5, x6], axis=1)

layer2 = Sequential()
layer2.add(Conv3D(layer2Filters, (2, 3, 3), activation='relu',
                  input_shape=(2, 19, 19, layer1Filters)))
layer2.add(BatchNormalization())
layer2.add(MaxPooling3D(pool_size=(1, 2, 2)))

z1 = layer2(y1)
z2 = layer2(y2)
z3 = layer2(y3)

z = concatenate([z1, z2, z3], axis=1)

layer3 = Sequential()
layer3.add(Conv3D(layer3Filters, (3, 3, 3), activation='relu',
                  input_shape=(3, 8, 8, layer2Filters)))
layer3.add(BatchNormalization())

z = layer3(z)
z = Flatten()(z)
z = Dense(fc1Nodes, activation='relu')(z)
vidFeatures = Dense(vidFeaturesDim, activation='relu')(z)

# wordFeatures = encoder(vidInput)
# wordFeatures = Lambda(lambda x: x[:, 0, :])(word)
wordFeatures = Input(shape=(encodedWordDim ,))

fullFeature = concatenate([vidFeatures, wordFeatures])

y = Dense(outputHDim, activation='relu')(fullFeature)

myOutput = Dense(1, activation='sigmoid')(y)

criticModelWithWordFeatures = Model(inputs=[vidInput, wordFeatures], outputs=myOutput)

lr = 1e-4
adam = Adam(lr=lr)
criticModelWithWordFeatures.compile(optimizer=adam, loss='binary_crossentropy',
                    metrics=['accuracy'])

criticModelWithWordFeatures.summary()

filenamePre = 'C3DCritic-l1f' + str(layer1Filters) + \
    '-l2f' + str(layer2Filters) + \
    '-l3f' + str(layer3Filters) + \
    '-fc1n' + str(fc1Nodes) + \
    '-vid' + str(vidFeaturesDim) + \
    '-enc' + str(encodedWordDim) + \
    '-oHn' + str(outputHDim) + \
    '-Adam-%1.e' % lr + \
    '-GRIDcorpus-s'
print(filenamePre)

# criticModelWithWordFeatures.load_weights(os.path.join(saveDir, "C3DCritic-l1f8-l2f16-l3f32-fc1n64-vid64-enc64-oHn64-Adam-1e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub-epoch007-tl0.3004-ta0.8631-vl0.4015-va0.8022.hdf5"))

######################################################################
# C3D Critic MODEL with ONE-HOT WORD input and 1 output Hidden layer
######################################################################

# 37a
criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=False, wordDim=1,
                                useEncWord=False, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=True, oneHotWordDim=52, useOneHotWordFc=False, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)

# OR

# Manual seeds
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)

# layer1Filters = 2
# layer2Filters = 2
# layer3Filters = 2
# fc1Nodes = 2
# vidFeaturesDim = 2
# encodedWordDim = 64
# outputHDim = 2

layer1Filters = 8
layer2Filters = 16
layer3Filters = 32
fc1Nodes = 64
vidFeaturesDim = 64
wordDim = wordsVocabSize
outputHDim = 64

vidInput = Input(shape=(framesPerWord, nOfMouthPixels,))
# 0 1 2 3  2 3 4 5  4 5 6 7  6 7 8 9  8 9 10 11  10 11 12 13
i1 = Lambda(lambda x: x[:, 0:4, :])(vidInput)
i2 = Lambda(lambda x: x[:, 2:6, :])(vidInput)
i3 = Lambda(lambda x: x[:, 4:8, :])(vidInput)
i4 = Lambda(lambda x: x[:, 6:10, :])(vidInput)
i5 = Lambda(lambda x: x[:, 8:12, :])(vidInput)
i6 = Lambda(lambda x: x[:, 10:14, :])(vidInput)

layer1 = Sequential()
layer1.add(Reshape((4, mouthW, mouthH, 1), input_shape=(4, nOfMouthPixels)))
layer1.add(Conv3D(layer1Filters, 3, activation='relu'))
layer1.add(BatchNormalization())
layer1.add(MaxPooling3D(pool_size=(2, 2, 2)))

x1 = layer1(i1)
x2 = layer1(i2)
x3 = layer1(i3)
x4 = layer1(i4)
x5 = layer1(i5)
x6 = layer1(i6)

y1 = concatenate([x1, x2], axis=1)
y2 = concatenate([x3, x4], axis=1)
y3 = concatenate([x5, x6], axis=1)

layer2 = Sequential()
layer2.add(Conv3D(layer2Filters, (2, 3, 3), activation='relu',
                  input_shape=(y1.get_shape()[1], y1.get_shape()[2], y1.get_shape()[3], layer1Filters)))
layer2.add(BatchNormalization())
layer2.add(MaxPooling3D(pool_size=(1, 2, 2)))

z1 = layer2(y1)
z2 = layer2(y2)
z3 = layer2(y3)

z = concatenate([z1, z2, z3], axis=1)

layer3 = Sequential()
layer3.add(Conv3D(layer3Filters, (3, 3, 3), activation='relu',
                  input_shape=(z.get_shape()[1], z.get_shape()[2], z.get_shape()[3], layer2Filters)))
layer3.add(BatchNormalization())

z = layer3(z)
z = Flatten()(z)
z = Dense(fc1Nodes, activation='relu')(z)
vidFeatures = Dense(vidFeaturesDim, activation='relu')(z)

oneHotWordFeatures = Input(shape=(wordDim,))

fullFeature = concatenate([vidFeatures, oneHotWordFeatures])

y = Dense(outputHDim, activation='relu')(fullFeature)

# yWord = Dense(52, activation='sigmoid')(y)
# yWord = Reshape((1, 52))(yWord)
# yPad = Dense(52, activation='sigmoid')(y)
# yPad = Reshape((1, 52))(yPad)
# myOutput = concatenate([yWord, yPad], axis=1)

myOutput = Dense(1, activation='sigmoid')(y)

criticModelWithOneHotWord = Model(inputs=[vidInput, oneHotWordFeatures], outputs=myOutput)

lr = 5e-4
adam = Adam(lr=lr)
criticModelWithOneHotWord.compile(optimizer=adam, loss='binary_crossentropy',
                    metrics=['accuracy'])

criticModelWithOneHotWord.summary()

# i = np.zeros((2, 14, 20, 20, 1))
# w = np.zeros((2, 1))
# model.predict([i, w])

# filenamePre = 'C3DCritic-l1f' + str(layer1Filters) + \
#     '-l2f' + str(layer2Filters) + \
#     '-l3f' + str(layer3Filters) + \
#     '-fc1n' + str(fc1Nodes) + \
#     '-vid' + str(vidFeaturesDim) + \
#     '-oneHotWord' + str(wordDim) + \
#     '-out' + str(outputHDim) + \
#     '-Adam-%1.e' % lr + \
#     '-GRIDcorpus-s'
# print(filenamePre)

filenamePre = 'C3DCritic-l1f' + str(layer1Filters) + \
    '-l2f' + str(layer2Filters) + \
    '-l3f' + str(layer3Filters) + \
    '-fc1n' + str(fc1Nodes) + \
    '-vid' + str(vidFeaturesDim) + \
    '-predWordDis' + str(wordDim) + \
    '-out' + str(outputHDim) + \
    '-Adam-%1.e' % lr + \
    '-GRIDcorpus-s'
print(filenamePre)

################################################################################
# C3D Critic MODEL with reduced ONE-HOT WORD input and 1 output Hidden layer
################################################################################

# 38a
criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=False, wordDim=1,
                                useEncWord=False, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=True, oneHotWordDim=52, useOneHotWordFc=True, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)

# OR

# Manual seeds
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)

# layer1Filters = 2
# layer2Filters = 2
# layer3Filters = 2
# fc1Nodes = 2
# vidFeaturesDim = 2
# encodedWordDim = 64
# outputHDim = 2

layer1Filters = 8
layer2Filters = 16
layer3Filters = 32
fc1Nodes = 64
vidFeaturesDim = 64
wordDim = wordsVocabSize
oneHotWordFc = 10
outputHDim = 64

vidInput = Input(shape=(framesPerWord, nOfMouthPixels,))
# 0 1 2 3  2 3 4 5  4 5 6 7  6 7 8 9  8 9 10 11  10 11 12 13
i1 = Lambda(lambda x: x[:, 0:4, :])(vidInput)
i2 = Lambda(lambda x: x[:, 2:6, :])(vidInput)
i3 = Lambda(lambda x: x[:, 4:8, :])(vidInput)
i4 = Lambda(lambda x: x[:, 6:10, :])(vidInput)
i5 = Lambda(lambda x: x[:, 8:12, :])(vidInput)
i6 = Lambda(lambda x: x[:, 10:14, :])(vidInput)

layer1 = Sequential()
layer1.add(Reshape((4, mouthW, mouthH, 1), input_shape=(4, nOfMouthPixels)))
layer1.add(Conv3D(layer1Filters, 3, activation='relu'))
layer1.add(BatchNormalization())
layer1.add(MaxPooling3D(pool_size=(2, 2, 2)))

x1 = layer1(i1)
x2 = layer1(i2)
x3 = layer1(i3)
x4 = layer1(i4)
x5 = layer1(i5)
x6 = layer1(i6)

y1 = concatenate([x1, x2], axis=1)
y2 = concatenate([x3, x4], axis=1)
y3 = concatenate([x5, x6], axis=1)

layer2 = Sequential()
layer2.add(Conv3D(layer2Filters, (2, 3, 3), activation='relu',
                  input_shape=(y1.get_shape()[1], y1.get_shape()[2], y1.get_shape()[3], layer1Filters)))
layer2.add(BatchNormalization())
layer2.add(MaxPooling3D(pool_size=(1, 2, 2)))

z1 = layer2(y1)
z2 = layer2(y2)
z3 = layer2(y3)

z = concatenate([z1, z2, z3], axis=1)

layer3 = Sequential()
layer3.add(Conv3D(layer3Filters, (3, 3, 3), activation='relu',
                  input_shape=(z.get_shape()[1], z.get_shape()[2], z.get_shape()[3], layer2Filters)))
layer3.add(BatchNormalization())

z = layer3(z)
z = Flatten()(z)
z = Dense(fc1Nodes, activation='relu')(z)
vidFeatures = Dense(vidFeaturesDim, activation='relu')(z)

oneHotWordFeatures = Input(shape=(wordDim,))
redWordFeatures = Dense(oneHotWordFc, activation='relu')(oneHotWordFeatures)

fullFeature = concatenate([vidFeatures, redWordFeatures])

y = Dense(outputHDim, activation='relu')(fullFeature)

# yWord = Dense(52, activation='sigmoid')(y)
# yWord = Reshape((1, 52))(yWord)
# yPad = Dense(52, activation='sigmoid')(y)
# yPad = Reshape((1, 52))(yPad)
# myOutput = concatenate([yWord, yPad], axis=1)

myOutput = Dense(1, activation='sigmoid')(y)

criticModelWithOneHotWord = Model(inputs=[vidInput, oneHotWordFeatures], outputs=myOutput)

lr = 5e-4
adam = Adam(lr=lr)
criticModelWithOneHotWord.compile(optimizer=adam, loss='binary_crossentropy',
                    metrics=['accuracy'])

criticModelWithOneHotWord.summary()

# i = np.zeros((2, 14, 20, 20, 1))
# w = np.zeros((2, 1))
# model.predict([i, w])

filenamePre = 'C3DCritic-l1f' + str(layer1Filters) + \
    '-l2f' + str(layer2Filters) + \
    '-l3f' + str(layer3Filters) + \
    '-fc1n' + str(fc1Nodes) + \
    '-vid' + str(vidFeaturesDim) + \
    '-oneHotWord' + str(wordDim) + \
    '-oneHotWordFc' + str(oneHotWordFc) + \
    '-out' + str(outputHDim) + \
    '-Adam-%1.e' % lr + \
    '-GRIDcorpus-s'
print(filenamePre)

####################################################################################################
# C3D Critic MODEL with Encoded WORD + ONE-HOT WORD input and 1 output Hidden layer
####################################################################################################

# 39a
criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=False, wordDim=1,
                                useEncWord=True, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=True, oneHotWordDim=52, useOneHotWordFc=False, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)

# OR

# Manual seeds
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)

# layer1Filters = 2
# layer2Filters = 2
# layer3Filters = 2
# fc1Nodes = 2
# vidFeaturesDim = 2
# encodedWordDim = 64
# outputHDim = 2

layer1Filters = 8
layer2Filters = 16
layer3Filters = 32
fc1Nodes = 64
vidFeaturesDim = 64
encodedWordDim = 64
oneHotWordDim = wordsVocabSize
outputHDim = 64

vidInput = Input(shape=(framesPerWord, nOfMouthPixels,))
# 0 1 2 3  2 3 4 5  4 5 6 7  6 7 8 9  8 9 10 11  10 11 12 13
i1 = Lambda(lambda x: x[:, 0:4, :])(vidInput)
i2 = Lambda(lambda x: x[:, 2:6, :])(vidInput)
i3 = Lambda(lambda x: x[:, 4:8, :])(vidInput)
i4 = Lambda(lambda x: x[:, 6:10, :])(vidInput)
i5 = Lambda(lambda x: x[:, 8:12, :])(vidInput)
i6 = Lambda(lambda x: x[:, 10:14, :])(vidInput)

layer1 = Sequential()
layer1.add(Reshape((4, mouthW, mouthH, 1), input_shape=(4, nOfMouthPixels)))
layer1.add(Conv3D(layer1Filters, 3, activation='relu'))
layer1.add(BatchNormalization())
layer1.add(MaxPooling3D(pool_size=(2, 2, 2)))

x1 = layer1(i1)
x2 = layer1(i2)
x3 = layer1(i3)
x4 = layer1(i4)
x5 = layer1(i5)
x6 = layer1(i6)

y1 = concatenate([x1, x2], axis=1)
y2 = concatenate([x3, x4], axis=1)
y3 = concatenate([x5, x6], axis=1)

layer2 = Sequential()
layer2.add(Conv3D(layer2Filters, (2, 3, 3), activation='relu',
                  input_shape=(y1.get_shape()[1], y1.get_shape()[2], y1.get_shape()[3], layer1Filters)))
layer2.add(BatchNormalization())
layer2.add(MaxPooling3D(pool_size=(1, 2, 2)))

z1 = layer2(y1)
z2 = layer2(y2)
z3 = layer2(y3)

z = concatenate([z1, z2, z3], axis=1)

layer3 = Sequential()
layer3.add(Conv3D(layer3Filters, (3, 3, 3), activation='relu',
                  input_shape=(z.get_shape()[1], z.get_shape()[2], z.get_shape()[3], layer2Filters)))
layer3.add(BatchNormalization())

v = layer3(z)
vF = Flatten()(v)
vFD = Dense(fc1Nodes, activation='relu', name="DenseVideo1")(vF)
vidFeatures = Dense(vidFeaturesDim, activation='relu', name="DenseVideoFeatures")(vFD)

encodedWordFeatures = Input(shape=(encodedWordDim,))

oneHotWordFeatures = Input(shape=(oneHotWordDim,))

fullFeature = concatenate([vidFeatures, encodedWordFeatures, oneHotWordFeatures])

y = Dense(outputHDim, activation='relu', name="DenseFullFeature")(fullFeature)

myOutput = Dense(1, activation='sigmoid', name="DenseOutput")(y)

criticModelWithEncAndOneHotWord = Model(inputs=[vidInput, encodedWordFeatures, oneHotWordFeatures], outputs=myOutput)

lr = 1e-2
adam = Adam(lr=lr)
criticModelWithEncAndOneHotWord.compile(optimizer=adam, loss='binary_crossentropy',
                    metrics=['accuracy'])

criticModelWithEncAndOneHotWord.summary()

# i = np.zeros((2, 14, 20, 20, 1))
# w = np.zeros((2, 1))
# model.predict([i, w])

filenamePre = 'C3DCritic-LRnoPadResults-l1f' + str(layer1Filters) + \
    '-l2f' + str(layer2Filters) + \
    '-l3f' + str(layer3Filters) + \
    '-fc1n' + str(fc1Nodes) + \
    '-vid' + str(vidFeaturesDim) + \
    '-enc' + str(encodedWordDim) + \
    '-oneHotWord' + str(oneHotWordDim) + \
    '-out' + str(outputHDim) + \
    '-Adam-%1.e' % lr + \
    '-GRIDcorpus-s'
print(filenamePre)

############################################################################################################
# C3D Critic MODEL with Encoded WORD & 1HidLayer + ONE-HOT WORD input & 1HidLayer and 1 output Hidden layer
############################################################################################################

# 40a
criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=False, wordDim=1,
                                useEncWord=True, encWordDim=64, useEncWordFc=True, encWordFcDim=10,
                                useOneHotWord=True, oneHotWordDim=52, useOneHotWordFc=True, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)

# OR

# Manual seeds
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)

# layer1Filters = 2
# layer2Filters = 2
# layer3Filters = 2
# fc1Nodes = 2
# vidFeaturesDim = 2
# encodedWordDim = 64
# outputHDim = 2

layer1Filters = 8
layer2Filters = 16
layer3Filters = 32
fc1Nodes = 64
vidFeaturesDim = 10
encWordFeatureDim = 64
encWordHiddenDim = 10
oneHotWordDim = wordsVocabSize
oneHotWordHiddenDim = 10
outputHDim = 64

vidInput = Input(shape=(framesPerWord, nOfMouthPixels,), name="InputVideo")
# 0 1 2 3  2 3 4 5  4 5 6 7  6 7 8 9  8 9 10 11  10 11 12 13
i1 = Lambda(lambda x: x[:, 0:4, :])(vidInput)
i2 = Lambda(lambda x: x[:, 2:6, :])(vidInput)
i3 = Lambda(lambda x: x[:, 4:8, :])(vidInput)
i4 = Lambda(lambda x: x[:, 6:10, :])(vidInput)
i5 = Lambda(lambda x: x[:, 8:12, :])(vidInput)
i6 = Lambda(lambda x: x[:, 10:14, :])(vidInput)

layer1 = Sequential()
layer1.add(Reshape((4, mouthW, mouthH, 1), input_shape=(4, nOfMouthPixels)))
layer1.add(Conv3D(layer1Filters, 3, activation='relu', name="Conv3DLayer1"))
layer1.add(BatchNormalization())
layer1.add(MaxPooling3D(pool_size=(2, 2, 2)))

x1 = layer1(i1)
x2 = layer1(i2)
x3 = layer1(i3)
x4 = layer1(i4)
x5 = layer1(i5)
x6 = layer1(i6)

y1 = concatenate([x1, x2], axis=1)
y2 = concatenate([x3, x4], axis=1)
y3 = concatenate([x5, x6], axis=1)

layer2 = Sequential()
layer2.add(Conv3D(layer2Filters, (2, 3, 3), activation='relu',
                  input_shape=(y1.shape.as_list()[1], y1.shape.as_list()[2], y1.shape.as_list()[3], layer1Filters), name="Conv3DLayer2"))
layer2.add(BatchNormalization())
layer2.add(MaxPooling3D(pool_size=(1, 2, 2)))

z1 = layer2(y1)
z2 = layer2(y2)
z3 = layer2(y3)

z = concatenate([z1, z2, z3], axis=1)

layer3 = Sequential()
layer3.add(Conv3D(layer3Filters, (3, 3, 3), activation='relu',
                  input_shape=(z.shape.as_list()[1], z.shape.as_list()[2], z.shape.as_list()[3], layer2Filters), name="Conv3DLayer3"))
layer3.add(BatchNormalization())

v = layer3(z)
vF = Flatten()(v)
vFD = Dense(fc1Nodes, activation='relu', name="DenseVideo1")(vF)
vidFeatures = Dense(vidFeaturesDim, activation='relu', name="DenseVideoFeatures")(vFD)

encodedWordFeatures = Input(shape=(encWordFeatureDim,), name="InputEncWord")
encodedWordFcFeatures = Dense(encWordHiddenDim, activation='relu', name="DenseEncWord")(encodedWordFeatures)

oneHotWordFeatures = Input(shape=(oneHotWordDim,), name="InputOneHotWord")
oneHotWordFcFeatures = Dense(oneHotWordHiddenDim, activation='relu', name="DenseOneHotWord")(oneHotWordFeatures)

fullFeature = concatenate([vidFeatures, encodedWordFcFeatures, oneHotWordFcFeatures], name="ConcFullFeature")

y = Dense(outputHDim, activation='relu', name="DenseFullFeature")(fullFeature)

myOutput = Dense(1, activation='sigmoid', name="DenseOutput")(y)

criticModelWithEncAndOneHotWord = Model(inputs=[vidInput, encodedWordFeatures, oneHotWordFeatures], outputs=myOutput)

lr = 1e-4
adam = Adam(lr=lr)
criticModelWithEncAndOneHotWord.compile(optimizer=adam, loss='binary_crossentropy',
                    metrics=['accuracy'])

criticModelWithEncAndOneHotWord.summary()

# i = np.zeros((2, 14, 20, 20, 1))
# w = np.zeros((2, 1))
# model.predict([i, w])

filenamePre = 'C3DCritic-l1f' + str(layer1Filters) + \
    '-l2f' + str(layer2Filters) + \
    '-l3f' + str(layer3Filters) + \
    '-fc1n' + str(fc1Nodes) + \
    '-vid' + str(vidFeaturesDim) + \
    '-enc' + str(encWordFeatureDim) + \
    '-encH' + str(encWordHiddenDim) + \
    '-oneHotWord' + str(oneHotWordDim) + \
    '-oneHotWordH' + str(oneHotWordHiddenDim) + \
    '-out' + str(outputHDim) + \
    '-Adam-%1.e' % lr + \
    '-GRIDcorpus-s'
print(filenamePre)

##############################################################################
# C3D Critic MODEL with Reduced WordFeatures input and 1 output Hidden layer
##############################################################################

criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=False, wordDim=1,
                                useEncWord=True, encWordDim=64, useEncWordFc=True, encWordFcDim=10,
                                useOneHotWord=False, oneHotWordDim=52, useOneHotWordFc=False, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)

# OR

# Manual seeds
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)

layer1Filters = 4
layer2Filters = 2
layer3Filters = 2
fc1Nodes = 2
vidFeaturesDim = 2
encodedWordDim = 64
encFc2Nodes = 2
outputHDim = 2

vidInput = Input(shape=(framesPerWord, nOfMouthPixels,))
# 0 1 2 3  2 3 4 5  4 5 6 7  6 7 8 9  8 9 10 11  10 11 12 13
i1 = Lambda(lambda x: x[:, 0:4, :])(vidInput)
i2 = Lambda(lambda x: x[:, 2:6, :])(vidInput)
i3 = Lambda(lambda x: x[:, 4:8, :])(vidInput)
i4 = Lambda(lambda x: x[:, 6:10, :])(vidInput)
i5 = Lambda(lambda x: x[:, 8:12, :])(vidInput)
i6 = Lambda(lambda x: x[:, 10:14, :])(vidInput)

layer1 = Sequential()
layer1.add(Reshape((4, mouthW, mouthH, 1), input_shape=(4, nOfMouthPixels)))
layer1.add(Conv3D(layer1Filters, 3, activation='relu'))
layer1.add(BatchNormalization())
layer1.add(MaxPooling3D(pool_size=(2, 2, 2)))

x1 = layer1(i1)
x2 = layer1(i2)
x3 = layer1(i3)
x4 = layer1(i4)
x5 = layer1(i5)
x6 = layer1(i6)

y1 = concatenate([x1, x2], axis=1)
y2 = concatenate([x3, x4], axis=1)
y3 = concatenate([x5, x6], axis=1)

layer2 = Sequential()
layer2.add(Conv3D(layer2Filters, (2, 3, 3), activation='relu',
                  input_shape=(2, 19, 19, layer1Filters)))
layer2.add(BatchNormalization())
layer2.add(MaxPooling3D(pool_size=(1, 2, 2)))

z1 = layer2(y1)
z2 = layer2(y2)
z3 = layer2(y3)

z = concatenate([z1, z2, z3], axis=1)

layer3 = Sequential()
layer3.add(Conv3D(layer3Filters, (3, 3, 3), activation='relu',
                  input_shape=(3, 8, 8, layer2Filters)))
layer3.add(BatchNormalization())

z = layer3(z)
z = Flatten()(z)
z = Dense(fc1Nodes, activation='relu')(z)
vidFeatures = Dense(vidFeaturesDim, activation='relu')(z)

# wordFeatures = encoder(vidInput)
# wordFeatures = Lambda(lambda x: x[:, 0, :])(word)
wordFeatures = Input(shape=(encodedWordDim ,))
redWordFeatures = Dense(encFc2Nodes, activation='relu')(wordFeatures)

fullFeature = concatenate([vidFeatures, redWordFeatures])

y = Dense(outputHDim, activation='relu')(fullFeature)

myOutput = Dense(1, activation='sigmoid')(y)

criticModelWithWordFeatures = Model(inputs=[vidInput, wordFeatures], outputs=myOutput)

lr = 1e-4
adam = Adam(lr=lr)
criticModelWithWordFeatures.compile(optimizer=adam, loss='binary_crossentropy',
                    metrics=['accuracy'])

criticModelWithWordFeatures.summary()

filenamePre = 'C3DCritic-l1f' + str(layer1Filters) + \
    '-l2f' + str(layer2Filters) + \
    '-l3f' + str(layer3Filters) + \
    '-fc1n' + str(fc1Nodes) + \
    '-vid' + str(vidFeaturesDim) + \
    '-enc' + str(encodedWordDim) + \
    '-encFc2n' + str(encFc2Nodes) + \
    '-oHn' + str(outputHDim) + \
    '-Adam-%1.e' % lr + \
    '-GRIDcorpus-s'
print(filenamePre)

#######################################################################
# C3D Critic MODEL with WordFeatures input and 2 output Hidden layers
#######################################################################

# Manual seeds
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)

layer1Filters = 4
layer2Filters = 4
layer3Filters = 4
fc1Nodes = 4
vidFeaturesDim = 4
encodedWordDim = 64
outputHDim1 = 4
outputHDim2 = 4

vidInput = Input(shape=(framesPerWord, nOfMouthPixels,))
# 0 1 2 3  2 3 4 5  4 5 6 7  6 7 8 9  8 9 10 11  10 11 12 13
i1 = Lambda(lambda x: x[:, 0:4, :])(vidInput)
i2 = Lambda(lambda x: x[:, 2:6, :])(vidInput)
i3 = Lambda(lambda x: x[:, 4:8, :])(vidInput)
i4 = Lambda(lambda x: x[:, 6:10, :])(vidInput)
i5 = Lambda(lambda x: x[:, 8:12, :])(vidInput)
i6 = Lambda(lambda x: x[:, 10:14, :])(vidInput)

layer1 = Sequential()
layer1.add(Reshape((4, mouthW, mouthH, 1), input_shape=(4, nOfMouthPixels)))
layer1.add(Conv3D(layer1Filters, 3, activation='relu'))
layer1.add(BatchNormalization())
layer1.add(MaxPooling3D(pool_size=(2, 2, 2)))

x1 = layer1(i1)
x2 = layer1(i2)
x3 = layer1(i3)
x4 = layer1(i4)
x5 = layer1(i5)
x6 = layer1(i6)

y1 = concatenate([x1, x2], axis=1)
y2 = concatenate([x3, x4], axis=1)
y3 = concatenate([x5, x6], axis=1)

layer2 = Sequential()
layer2.add(Conv3D(layer2Filters, (2, 3, 3), activation='relu',
                  input_shape=(2, 19, 19, layer1Filters)))
layer2.add(BatchNormalization())
layer2.add(MaxPooling3D(pool_size=(1, 2, 2)))

z1 = layer2(y1)
z2 = layer2(y2)
z3 = layer2(y3)

z = concatenate([z1, z2, z3], axis=1)

layer3 = Sequential()
layer3.add(Conv3D(layer3Filters, (3, 3, 3), activation='relu',
                  input_shape=(3, 3, 3, layer2Filters)))
layer3.add(BatchNormalization())

z = layer3(z)
z = Flatten()(z)
z = Dense(fc1Nodes, activation='relu')(z)
vidFeatures = Dense(vidFeaturesDim, activation='relu')(z)

# wordFeatures = encoder(vidInput)
# wordFeatures = Lambda(lambda x: x[:, 0, :])(word)
wordFeatures = Input(shape=(encodedWordDim ,))

fullFeature = concatenate([vidFeatures, wordFeatures])

y = Dense(outputHDim1, activation='relu')(fullFeature)
y = Dense(outputHDim2, activation='relu')(y)

myOutput = Dense(1, activation='sigmoid')(y)

criticModelWithWordFeatures = Model(inputs=[vidInput, wordFeatures], outputs=myOutput)

lr = 1e-4
adam = Adam(lr=lr)
criticModelWithWordFeatures.compile(optimizer=adam, loss='binary_crossentropy',
                    metrics=['accuracy'])

criticModelWithWordFeatures.summary()

# i = np.zeros((2, 14, 20, 20, 1))
# w = np.zeros((2, 1))
# model.predict([i, w])

filenamePre = 'C3DCritic-l1f' + str(layer1Filters) + \
    '-l2f' + str(layer2Filters) + \
    '-l3f' + str(layer3Filters) + \
    '-fc1n' + str(fc1Nodes) + \
    '-vid' + str(vidFeaturesDim) + \
    '-enc' + str(encodedWordDim) + \
    '-oH1n' + str(outputHDim1) + \
    '-oH2n' + str(outputHDim2) + \
    '-Adam-%1.e' % lr + \
    '-GRIDcorpus-s'
print(filenamePre)

##############################################################################
# C3D Critic MODEL with Reduced WordFeatures input and 2 output Hidden layers
##############################################################################

# Manual seeds
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)

layer1Filters = 4
layer2Filters = 4
layer3Filters = 4
fc1Nodes = 4
vidFeaturesDim = 4
encodedWordDim = 64
encFc2Nodes = 4
outputHDim1 = 4
outputHDim2 = 4

vidInput = Input(shape=(framesPerWord, nOfMouthPixels,))
# 0 1 2 3  2 3 4 5  4 5 6 7  6 7 8 9  8 9 10 11  10 11 12 13
i1 = Lambda(lambda x: x[:, 0:4, :])(vidInput)
i2 = Lambda(lambda x: x[:, 2:6, :])(vidInput)
i3 = Lambda(lambda x: x[:, 4:8, :])(vidInput)
i4 = Lambda(lambda x: x[:, 6:10, :])(vidInput)
i5 = Lambda(lambda x: x[:, 8:12, :])(vidInput)
i6 = Lambda(lambda x: x[:, 10:14, :])(vidInput)

layer1 = Sequential()
layer1.add(Reshape((4, mouthW, mouthH, 1), input_shape=(4, nOfMouthPixels)))
layer1.add(Conv3D(layer1Filters, 3, activation='relu'))
layer1.add(BatchNormalization())
layer1.add(MaxPooling3D(pool_size=(2, 2, 2)))

x1 = layer1(i1)
x2 = layer1(i2)
x3 = layer1(i3)
x4 = layer1(i4)
x5 = layer1(i5)
x6 = layer1(i6)

y1 = concatenate([x1, x2], axis=1)
y2 = concatenate([x3, x4], axis=1)
y3 = concatenate([x5, x6], axis=1)

layer2 = Sequential()
layer2.add(Conv3D(layer2Filters, (2, 3, 3), activation='relu',
                  input_shape=(2, 19, 19, layer1Filters)))
layer2.add(BatchNormalization())
layer2.add(MaxPooling3D(pool_size=(1, 2, 2)))

z1 = layer2(y1)
z2 = layer2(y2)
z3 = layer2(y3)

z = concatenate([z1, z2, z3], axis=1)

layer3 = Sequential()
layer3.add(Conv3D(layer3Filters, (3, 3, 3), activation='relu',
                  input_shape=(3, 3, 3, layer2Filters)))
layer3.add(BatchNormalization())

z = layer3(z)
z = Flatten()(z)
z = Dense(fc1Nodes, activation='relu')(z)
vidFeatures = Dense(vidFeaturesDim, activation='relu')(z)

# wordFeatures = encoder(vidInput)
# wordFeatures = Lambda(lambda x: x[:, 0, :])(word)
wordFeatures = Input(shape=(encodedWordDim ,))
redWordFeatures = Dense(encFc2Nodes, activation='relu')(wordFeatures)

fullFeature = concatenate([vidFeatures, redWordFeatures])

y = Dense(outputHDim1, activation='relu')(fullFeature)
y = Dense(outputHDim2, activation='relu')(y)

myOutput = Dense(1, activation='sigmoid')(y)

criticModelWithWordFeatures = Model(inputs=[vidInput, wordFeatures], outputs=myOutput)

lr = 1e-4
adam = Adam(lr=lr)
criticModelWithWordFeatures.compile(optimizer=adam, loss='binary_crossentropy',
                    metrics=['accuracy'])

criticModelWithWordFeatures.summary()

# i = np.zeros((2, 14, 20, 20, 1))
# w = np.zeros((2, 1))
# model.predict([i, w])

filenamePre = 'C3DCritic-l1f' + str(layer1Filters) + \
    '-l2f' + str(layer2Filters) + \
    '-l3f' + str(layer3Filters) + \
    '-fc1n' + str(fc1Nodes) + \
    '-vid' + str(vidFeaturesDim) + \
    '-enc' + str(encodedWordDim) + \
    '-encFc2n' + str(encFc2Nodes) + \
    '-oH1n' + str(outputHDim1) + \
    '-oH2n' + str(outputHDim2) + \
    '-Adam-%1.e' % lr + \
    '-GRIDcorpus-s'
print(filenamePre)

##############################################################################################################
# LSTM Critic MODEL with EncWordFeatures + fc + OneHotWord + fc input and 2 output Hidden layers
##############################################################################################################

# Manual seeds
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)

LSTMdim = 20
vidFeaturesDim = 20
encDim = 64
encFcDim = 20
OHWordDim = wordsVocabSize
OHWordFcDim = 20
outDim1 = 20
outDim2 = 20

# Video input
vidInput = Input(shape=(framesPerWord, nOfMouthPixels,))
encoded = LSTM(LSTMdim, return_sequences=False)(vidInput)
vidFeatures = Dense(vidFeaturesDim, activation='relu')(encoded)

wordFeatures = Input(shape=(encDim,))
wordFcFeatures = Dense(encFcDim, activation='relu')(wordFeatures)

oneHotWord = Input(shape=(OHWordDim,))
oneHotWordFcFeatures = Dense(OHWordFcDim, activation='relu')(oneHotWord)

fullFeature = concatenate([vidFeatures, wordFcFeatures, oneHotWordFcFeatures])

y = Dense(outDim1, activation='relu')(fullFeature)
y = Dense(outDim2, activation='relu')(y)

myOutput = Dense(1, activation='sigmoid')(y)

criticModel = Model(inputs=[vidInput, wordFeatures, oneHotWord], outputs=myOutput)

lr = 1e-7
adam = Adam(lr=lr)
criticModel.compile(optimizer=adam, loss='binary_crossentropy',
                    metrics=['accuracy'])

criticModel.summary()

filenamePre = 'CriticLSTM-LSTMdim' + str(LSTMdim) + \
    '-vid' + str(vidFeaturesDim) + \
    '-enc' + str(encDim) + \
    '-encFc' + str(encFcDim) + \
    '-OHWord' + str(OHWordDim) + \
    '-OHWordFc' + str(OHWordFcDim) + \
    '-out1Dim' + str(outDim1) + \
    '-out2Dim' + str(outDim2) + \
    '-Adam-%1.e' % lr + \
    '-GRIDcorpus-s'
print(filenamePre)

#############################################################
# COMPLETE FILENAMEPRE
#############################################################

def completeFilename(filenamePre):
    # SPEAKERS LIST IN FILENAME
    prevS = -1
    for s in speakersList:
        # print(s)
        if prevS == -1:
            filenamePre += '{0:02d}'.format(s)
        elif s - prevS > 1:
            filenamePre += '{0:02d}-s{1:02d}'.format(prevS, s)
        # Prev S
        prevS = s
        # print(filenamePre)
    # Last speaker
    filenamePre += '{0:02d}'.format(s)
    if trainAlign:
        filenamePre += '-tAlign'
    else:
        filenamePre += '-tMouth'
    if valAlign:
        filenamePre += '-vAlign'
    else:
        filenamePre += '-vMouth'
    if useMeanMouthImage:
        filenamePre += '-meanSub'
    else:
        filenamePre += '-NOmeanSub'
    print(filenamePre)
    return filenamePre

##################################################################
# TRAIN CRITIC WITH WORD ONLY WITH LIPREADER PREDICTIONS
##################################################################

tlS = []
vlS = []
taS = []
vaS = []
tlE = []
vlE = []
taE = []
vaE = []
tl = 0
ta = 0
vl = 0
va = 0

nEpochs = 11
batchSize = 128
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)

genTrainImages = genMouthImages(trainDirs, batchSize=batchSize, align=trainAlign, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)
genValImages = genMouthImages(valDirs, batchSize=batchSize, align=valAlign, wordIdx=wordIdx,
                              wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)

# For each epoch
for epoch in tqdm.tqdm(range(nEpochs)):
    # For each train batch
    for step in tqdm.tqdm(range(trainSteps)):
        vids, words = next(genTrainImages)
        words = np.argmax(words[:, 0, :], axis=1)
        predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        fullIdx = list(range(len(vids)))
        np.random.shuffle(fullIdx)
        inputs1 = vids[fullIdx]
        inputs2 = predWords[fullIdx] * 1. / wordsVocabSize
        outputs = correctPreds[fullIdx]
        h = criticModelWithWord.fit([inputs1, inputs2], outputs,
                            batch_size=batchSize, epochs=1, initial_epoch=0)
        tl = h.history['loss'][0]
        ta = h.history['acc'][0]
        tlS.append(tl)
        taS.append(ta)
    # For each val batch
    for step in tqdm.tqdm(range(valSteps)):
        vids, words = next(genValImages)
        words = np.argmax(words[:, 0, :], axis=1)
        predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = predWords * 1. / wordsVocabSize
        outputs = correctPreds
        vl, va = criticModelWithWord.evaluate(
            [inputs1, inputs2], outputs, batch_size=batchSize)
        vlS.append(vl)
        vaS.append(va)
    # Append values
    tl = np.mean(tlS)
    tlE.append(tl)
    tlS = []
    ta = np.mean(taS)
    taE.append(ta)
    taS = []
    vl = np.mean(vlS)
    vlE.append(vl)
    vlS = []
    va = np.mean(vaS)
    vaE.append(va)
    vaS = []
    print(
        "epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}".format(epoch, tl, ta, vl, va))
    criticModelWithWord.save_weights(os.path.join(
        saveDir, filenamePre + "-epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}.hdf5".format(epoch, tl, ta, vl, va)))
    plt.plot(tlE, label='trainingLoss')
    plt.plot(vlE, label='valLoss')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(os.path.join(saveDir, filenamePre + "-plotLosses.png"))
    plt.close()
    plt.plot(taE, label='trainingAcc')
    plt.plot(vaE, label='valAcc')
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.gca().yaxis.grid(True)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.savefig(os.path.join(saveDir, filenamePre + "-plotAcc.png"))
    plt.close()

#####################################################################
# TRAIN CRITIC WITH ENC WORD FEATURES ONLY WITH LIPREADER PREDICTIONS
#####################################################################

criticModel, filenamePre = C3DCriticModel(
                            layer1Filters=4,
                            layer2Filters=4,
                            layer3Filters=4,
                            fc1Nodes=10,
                            vidFeaturesDim=10,
                            useWord=False,
                            wordDim=1,
                            useEncWord=True,
                            encWordDim=64,
                            useEncWordFc=False,
                            encWordFcDim=10,
                            useOneHotWord=False,
                            oneHotWordDim=52,
                            useOneHotWordFc=False,
                            oneHotWordFcDim=10,
                            usePredWordDis=False,
                            predWordDisDim=52,
                            outputHDim=64,
                            lr=5e-5)

filenamePre = completeFilename(filenamePre)

tlS = []
vlS = []
silS = []
taS = []
vaS = []
siaS = []
tlE = []
vlE = []
silE = []
taE = []
vaE = []
siaE = []
tl = 0
vl = 0
sil = 0
ta = 0
va = 0
sia = 0

nEpochs = 11
batchSize = 128
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)

genTrainImages = genMouthImages(trainDirs, batchSize=batchSize, align=False, wordIdx=wordIdx, wordsVocabSize=wordsVocabSize,
                                useMeanMouthImage=False, shuffle=True, shuffleWords=True, keepPadResults=False)

genValImages = genMouthImages(valDirs, batchSize=batchSize, align=False, wordIdx=wordIdx, wordsVocabSize=wordsVocabSize,
                                useMeanMouthImage=False, shuffle=False, shuffleWords=False, keepPadResults=False)

t0 = time.time()
# For each epoch
for epoch in tqdm.tqdm(range(nEpochs)):
    # For each train batch
    print("TRAINING...")
    for step in tqdm.tqdm(range(trainSteps)):
        vids, words = next(genTrainImages)
        predWordFeatures = encoder.predict(vids)
        # words = np.argmax(words[:, 0, :], axis=1)
        words = np.argmax(words, axis=1)
        # predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
        predWords = np.argmax(LSTMLipReaderModel.predict(vids), axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = predWordFeatures
        outputs = correctPreds
        h = criticModel.fit([inputs1, inputs2], outputs,
                            batch_size=batchSize, epochs=1, initial_epoch=0)
        tl = h.history['loss'][0]
        ta = h.history['acc'][0]
        tlS.append(tl)
        taS.append(ta)
    # For each val batch
    print("VAL...")
    for step in tqdm.tqdm(range(valSteps)):
        vids, words = next(genValImages)
        predWordFeatures = encoder.predict(vids)
        words = np.argmax(words, axis=1)
        predWords = np.argmax(LSTMLipReaderModel.predict(vids), axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = predWordFeatures
        outputs = correctPreds
        vl, va = criticModel.evaluate([inputs1, inputs2], outputs, batch_size=batchSize)
        vlS.append(vl)
        vaS.append(va)
    # For each si batch
    print("SPEAKER INDEPENDENT...")
    genSiImages = genMouthImages(siDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
            wordsVocabSize=wordsVocabSize, useMeanMouthImage=False, shuffle=False, shuffleWords=False, keepPadResults=False)
    nOfSteps = 27
    for step in tqdm.tqdm(range(nOfSteps)):
        vids, words = next(genSiImages)
        predWordFeatures = encoder.predict(vids)
        words = np.argmax(words, axis=1)
        predWords = np.argmax(LSTMLipReaderModel.predict(vids), axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = predWordFeatures
        outputs = correctPreds
        sil, sia = criticModel.evaluate([inputs1, inputs2], outputs, batch_size=batchSize)
        silS.append(sil)
        siaS.append(sia)
    # Append values
    tl = np.mean(tlS)
    tlE.append(tl)
    tlS = []
    ta = np.mean(taS)
    taE.append(ta)
    taS = []
    vl = np.mean(vlS)
    vlE.append(vl)
    vlS = []
    va = np.mean(vaS)
    vaE.append(va)
    vaS = []
    sil = np.mean(silS)
    silE.append(sil)
    silS = []
    sia = np.mean(siaS)
    siaE.append(sia)
    siaS = []
    print(
        "epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-sil{5:.4f}-sia{6:.4f}".format(epoch, tl, ta, vl, va, sil, sia))
    criticModel.save_weights(os.path.join(
        saveDir, filenamePre + "-epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-sil{5:.4f}-sia{6:.4f}.hdf5".format(epoch, tl, ta, vl, va, sil, sia)))
    plt.plot(tlE, label='trainingLoss', color='r', linestyle='-.')
    plt.plot(vlE, label='valLoss', color='r', linestyle='-')
    plt.plot(silE, label='siLoss', color='r', linestyle='--')
    leg = plt.legend(loc='best', fontsize=11, fancybox=True)
    leg.get_frame().set_alpha(0.3)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(os.path.join(saveDir, filenamePre + "-plotLosses.png"))
    plt.close()
    plt.plot(taE, label='trainingAcc', color='r', linestyle='-.')
    plt.plot(vaE, label='valAcc', color='r', linestyle='-')
    plt.plot(siaE, label='siaAcc', color='r', linestyle='--')
    leg = plt.legend(loc='best', fontsize=11, fancybox=True)
    leg.get_frame().set_alpha(0.3)
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.gca().yaxis.grid(True)
    plt.savefig(os.path.join(saveDir, filenamePre + "-plotAcc.png"))
    plt.close()
    t1 = time.time()
    print("{0:.4f} seconds\n".format(t1 - t0))
    t0 = t1

##################################################################
# TRAIN CRITIC WITH ONE-HOT WORD ONLY WITH LIPREADER PREDICTIONS
##################################################################

tlS = []
vlS = []
taS = []
vaS = []
tlE = []
vlE = []
taE = []
vaE = []
tl = 0
ta = 0
vl = 0
va = 0

nEpochs = 11
batchSize = 128
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)

genTrainImages = genMouthImages(trainDirs, batchSize=batchSize, align=trainAlign, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)
genValImages = genMouthImages(valDirs, batchSize=batchSize, align=valAlign, wordIdx=wordIdx,
                              wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)

# For each epoch
for epoch in tqdm.tqdm(range(nEpochs)):
    # For each train batch
    for step in tqdm.tqdm(range(trainSteps)):
        vids, words = next(genTrainImages)
        words = np.argmax(words[:, 0, :], axis=1)
        predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        fullIdx = list(range(len(vids)))
        np.random.shuffle(fullIdx)
        inputs1 = vids[fullIdx]
        inputs2 = np_utils.to_categorical(predWords[fullIdx], wordsVocabSize)
        outputs = correctPreds[fullIdx]
        h = criticModelWithOneHotWord.fit([inputs1, inputs2], outputs,
                            batch_size=batchSize, epochs=1, initial_epoch=0)
        tl = h.history['loss'][0]
        ta = h.history['acc'][0]
        tlS.append(tl)
        taS.append(ta)
    # For each val batch
    for step in tqdm.tqdm(range(valSteps)):
        vids, words = next(genValImages)
        words = np.argmax(words[:, 0, :], axis=1)
        predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = np_utils.to_categorical(predWords, wordsVocabSize)
        outputs = correctPreds
        vl, va = criticModelWithOneHotWord.evaluate(
            [inputs1, inputs2], outputs, batch_size=batchSize)
        vlS.append(vl)
        vaS.append(va)
    # Append values
    tl = np.mean(tlS)
    tlE.append(tl)
    tlS = []
    ta = np.mean(taS)
    taE.append(ta)
    taS = []
    vl = np.mean(vlS)
    vlE.append(vl)
    vlS = []
    va = np.mean(vaS)
    vaE.append(va)
    vaS = []
    print(
        "epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}".format(epoch, tl, ta, vl, va))
    criticModelWithOneHotWord.save_weights(os.path.join(
        saveDir, filenamePre + "-epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}.hdf5".format(epoch, tl, ta, vl, va)))
    plt.plot(tlE, label='trainingLoss')
    plt.plot(vlE, label='valLoss')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(os.path.join(saveDir, filenamePre + "-plotLosses.png"))
    plt.close()
    plt.plot(taE, label='trainingAcc')
    plt.plot(vaE, label='valAcc')
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.gca().yaxis.grid(True)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.savefig(os.path.join(saveDir, filenamePre + "-plotAcc.png"))
    plt.close()

##################################################################################################
# TRAIN CRITIC WITH Encoded WORD + ONE-HOT WORD input ONLY WITH LIPREADER PREDICTIONS
##################################################################################################

criticModel, filenamePre = C3DCriticModel(
                            layer1Filters=4,
                            layer2Filters=4,
                            layer3Filters=4,
                            fc1Nodes=10,
                            vidFeaturesDim=10,
                            useWord=False,
                            wordDim=1,
                            useEncWord=True,
                            encWordDim=64,
                            useEncWordFc=False,
                            encWordFcDim=10,
                            useOneHotWord=True,
                            oneHotWordDim=52,
                            useOneHotWordFc=False,
                            oneHotWordFcDim=10,
                            usePredWordDis=False,
                            predWordDisDim=52,
                            outputHDim=64,
                            lr=5e-5)

filenamePre = completeFilename(filenamePre)

tlS = []
vlS = []
silS = []
taS = []
vaS = []
siaS = []
tlE = []
vlE = []
silE = []
taE = []
vaE = []
siaE = []
tl = 0
vl = 0
sil = 0
ta = 0
va = 0
sia = 0

nEpochs = 11
batchSize = 128
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)

genTrainImages = genMouthImages(trainDirs, batchSize=batchSize, align=False, wordIdx=wordIdx, wordsVocabSize=wordsVocabSize,
                                useMeanMouthImage=False, shuffle=True, shuffleWords=True, keepPadResults=False)

genValImages = genMouthImages(valDirs, batchSize=batchSize, align=False, wordIdx=wordIdx, wordsVocabSize=wordsVocabSize,
                                useMeanMouthImage=False, shuffle=False, shuffleWords=False, keepPadResults=False)

t0 = time.time()
# For each epoch
for epoch in tqdm.tqdm(range(nEpochs)):
    # For each train batch
    print("TRAINING...")
    for step in tqdm.tqdm(range(trainSteps)):
        vids, words = next(genTrainImages)
        predWordFeatures = encoder.predict(vids)
        # words = np.argmax(words[:, 0, :], axis=1)
        words = np.argmax(words, axis=1)
        # predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
        predWords = np.argmax(LSTMLipReaderModel.predict(vids), axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = predWordFeatures
        inputs3 = np_utils.to_categorical(predWords, wordsVocabSize)
        outputs = correctPreds
        h = criticModel.fit([inputs1, inputs2, inputs3], outputs,
                            batch_size=batchSize, epochs=1, initial_epoch=0)
        tl = h.history['loss'][0]
        ta = h.history['acc'][0]
        tlS.append(tl)
        taS.append(ta)
    # For each val batch
    print("VAL...")
    for step in tqdm.tqdm(range(valSteps)):
        vids, words = next(genValImages)
        predWordFeatures = encoder.predict(vids)
        words = np.argmax(words, axis=1)
        predWords = np.argmax(LSTMLipReaderModel.predict(vids), axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = predWordFeatures
        inputs3 = np_utils.to_categorical(predWords, wordsVocabSize)
        outputs = correctPreds
        vl, va = criticModel.evaluate([inputs1, inputs2, inputs3], outputs, batch_size=batchSize)
        vlS.append(vl)
        vaS.append(va)
    # For each si batch
    print("SPEAKER INDEPENDENT...")
    genSiImages = genMouthImages(siDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
            wordsVocabSize=wordsVocabSize, useMeanMouthImage=False, shuffle=False, shuffleWords=False, keepPadResults=False)
    nOfSteps = 27
    for step in tqdm.tqdm(range(nOfSteps)):
        vids, words = next(genSiImages)
        predWordFeatures = encoder.predict(vids)
        words = np.argmax(words, axis=1)
        predWords = np.argmax(LSTMLipReaderModel.predict(vids), axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = predWordFeatures
        inputs3 = np_utils.to_categorical(predWords, wordsVocabSize)
        outputs = correctPreds
        sil, sia = criticModel.evaluate([inputs1, inputs2, inputs3], outputs, batch_size=batchSize)
        silS.append(sil)
        siaS.append(sia)
    # Append values
    tl = np.mean(tlS)
    tlE.append(tl)
    tlS = []
    ta = np.mean(taS)
    taE.append(ta)
    taS = []
    vl = np.mean(vlS)
    vlE.append(vl)
    vlS = []
    va = np.mean(vaS)
    vaE.append(va)
    vaS = []
    sil = np.mean(silS)
    silE.append(sil)
    silS = []
    sia = np.mean(siaS)
    siaE.append(sia)
    siaS = []
    print(
        "epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-sil{5:.4f}-sia{6:.4f}".format(epoch, tl, ta, vl, va, sil, sia))
    criticModel.save_weights(os.path.join(
        saveDir, filenamePre + "-epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-sil{5:.4f}-sia{6:.4f}.hdf5".format(epoch, tl, ta, vl, va, sil, sia)))
    plt.plot(tlE, label='trainingLoss', color='r', linestyle='-.')
    plt.plot(vlE, label='valLoss', color='r', linestyle='-')
    plt.plot(silE, label='siLoss', color='r', linestyle='--')
    leg = plt.legend(loc='best', fontsize=11, fancybox=True)
    leg.get_frame().set_alpha(0.3)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(os.path.join(saveDir, filenamePre + "-plotLosses.png"))
    plt.close()
    plt.plot(taE, label='trainingAcc', color='r', linestyle='-.')
    plt.plot(vaE, label='valAcc', color='r', linestyle='-')
    plt.plot(siaE, label='siaAcc', color='r', linestyle='--')
    leg = plt.legend(loc='best', fontsize=11, fancybox=True)
    leg.get_frame().set_alpha(0.3)
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.gca().yaxis.grid(True)
    plt.savefig(os.path.join(saveDir, filenamePre + "-plotAcc.png"))
    plt.close()
    t1 = time.time()
    print("{0:.4f} seconds\n".format(t1 - t0))
    t0 = t1

##############################################################################
# TRAIN CRITIC WITH PRED-WORD-DISTRIBUTION ONLY WITH LIPREADER PREDICTIONS
##############################################################################

tlS = []
vlS = []
taS = []
vaS = []
tlE = []
vlE = []
taE = []
vaE = []
tl = 0
ta = 0
vl = 0
va = 0

nEpochs = 11
batchSize = 128
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)

genTrainImages = genMouthImages(trainDirs, batchSize=batchSize, align=trainAlign, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)
genValImages = genMouthImages(valDirs, batchSize=batchSize, align=valAlign, wordIdx=wordIdx,
                              wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)

# For each epoch
for epoch in tqdm.tqdm(range(nEpochs)):
    # For each train batch
    for step in tqdm.tqdm(range(trainSteps)):
        vids, words = next(genTrainImages)
        words = np.argmax(words[:, 0, :], axis=1)
        predWordDis = LSTMLipReaderModel.predict(vids)[:, 0, :]
        predWords = np.argmax(predWordDis, axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        fullIdx = list(range(len(vids)))
        np.random.shuffle(fullIdx)
        inputs1 = vids[fullIdx]
        inputs2 = predWordDis[fullIdx]
        outputs = correctPreds[fullIdx]
        h = criticModelWithOneHotWord.fit([inputs1, inputs2], outputs,
                            batch_size=batchSize, epochs=1, initial_epoch=0)
        tl = h.history['loss'][0]
        ta = h.history['acc'][0]
        tlS.append(tl)
        taS.append(ta)
    # For each val batch
    for step in tqdm.tqdm(range(valSteps)):
        vids, words = next(genValImages)
        words = np.argmax(words[:, 0, :], axis=1)
        predWordDis = LSTMLipReaderModel.predict(vids)[:, 0, :]
        predWords = np.argmax(predWordDis, axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = predWordDis
        outputs = correctPreds
        vl, va = criticModelWithOneHotWord.evaluate(
            [inputs1, inputs2], outputs, batch_size=batchSize)
        vlS.append(vl)
        vaS.append(va)
    # Append values
    tl = np.mean(tlS)
    tlE.append(tl)
    tlS = []
    ta = np.mean(taS)
    taE.append(ta)
    taS = []
    vl = np.mean(vlS)
    vlE.append(vl)
    vlS = []
    va = np.mean(vaS)
    vaE.append(va)
    vaS = []
    print(
        "epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}".format(epoch, tl, ta, vl, va))
    criticModelWithOneHotWord.save_weights(os.path.join(
        saveDir, filenamePre + "-epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}.hdf5".format(epoch, tl, ta, vl, va)))
    plt.plot(tlE, label='trainingLoss')
    plt.plot(vlE, label='valLoss')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(os.path.join(saveDir, filenamePre + "-plotLosses.png"))
    plt.close()
    plt.plot(taE, label='trainingAcc')
    plt.plot(vaE, label='valAcc')
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.gca().yaxis.grid(True)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.savefig(os.path.join(saveDir, filenamePre + "-plotAcc.png"))
    plt.close()

###############################################################################
# TRAIN CRITIC WITH WORD FEATURES WITH POSITIVE AND RANDOM NEGATIVE SAMPLES
###############################################################################

tlS = []
vlS = []
vlPureS = []
taS = []
vaS = []
vaPureS = []
tlE = []
vlE = []
vlPureE = []
taE = []
vaE = []
vaPureE = []
tl = 0
ta = 0
vl = 0
va = 0

nEpochs = 11
batchSize = 128
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)

genTrainImages = genMouthImages(trainDirs, batchSize=batchSize, align=trainAlign, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)
genValImages = genMouthImages(valDirs, batchSize=batchSize, align=valAlign, wordIdx=wordIdx,
                              wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)

# For each epoch
for epoch in tqdm.tqdm(range(nEpochs)):
    # For each train batch
    for step in tqdm.tqdm(range(trainSteps)):
        # Ground truth
        vids, words = next(genTrainImages)
        words = np.argmax(words[:, 0, :], axis=1)
        # Inputs1
        inputs1 = np.concatenate([vids, vids])
        # Inputs2
        predWordFeatures = encoder.predict(vids)
        wrongPredWordFeatures = np.vstack((predWordFeatures[-2:], predWordFeatures[:-2]))
        inputs2 = np.concatenate([predWordFeatures, wrongPredWordFeatures])
        # Outputs
        predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        wrongPredWords = np.hstack((predWords[-2:], predWords[:-2]))
        wrongPreds = np.array(words == wrongPredWords).astype(int)
        outputs = np.concatenate([correctPreds, wrongPreds])
        fullIdx = list(range(len(inputs1)))
        np.random.shuffle(fullIdx)
        inputs1 = inputs1[fullIdx]
        inputs2 = inputs2[fullIdx]
        outputs = outputs[fullIdx]
        h = criticModelWithWordFeatures.fit([inputs1, inputs2], outputs,
                            batch_size=batchSize, epochs=1, initial_epoch=0)
        tl = h.history['loss'][0]
        ta = h.history['acc'][0]
        tlS.append(tl)
        taS.append(ta)
    # For each val batch
    for step in tqdm.tqdm(range(valSteps)):
        vids, words = next(genValImages)
        words = np.argmax(words[:, 0, :], axis=1)
        # Inputs1
        inputs1 = np.concatenate([vids, vids])
        # Inputs2
        predWordFeatures = encoder.predict(vids)
        wrongPredWordFeatures = np.vstack((predWordFeatures[-2:], predWordFeatures[:-2]))
        inputs2 = np.concatenate([predWordFeatures, wrongPredWordFeatures])
        # Outputs
        predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        wrongPredWords = np.hstack((predWords[-2:], predWords[:-2]))
        wrongPreds = np.array(words == wrongPredWords).astype(int)
        outputs = np.concatenate([correctPreds, wrongPreds])
        vl, va = criticModelWithWordFeatures.evaluate(
            [inputs1, inputs2], outputs, batch_size=batchSize)
        vlS.append(vl)
        vaS.append(va)
        vl, va = criticModelWithWordFeatures.evaluate(
            [vids, predWordFeatures], correctPreds, batch_size=batchSize)
        vlPureS.append(vl)
        vaPureS.append(va)
    # Append values
    tl = np.mean(tlS)
    tlE.append(tl)
    tlS = []
    ta = np.mean(taS)
    taE.append(ta)
    taS = []
    vl = np.mean(vlS)
    vlE.append(vl)
    vlS = []
    va = np.mean(vaS)
    vaE.append(va)
    vaS = []
    vlP = np.mean(vlPureS)
    vlPureE.append(vlP)
    vlPureS = []
    vaP = np.mean(vaPureS)
    vaPureE.append(vaP)
    vaPureS = []
    print(
        "epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-vlPure{5:.4f}-vaPure{6:.4f}".format(epoch, tl, ta, vl, va, vlP, vaP))
    criticModelWithWordFeatures.save_weights(os.path.join(
        saveDir, filenamePre + "-epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-vlPure{5:.4f}-vaPure{6:.4f}.hdf5".format(epoch, tl, ta, vl, va, vlP, vaP)))
    plt.plot(tlE, label='trainingLoss')
    plt.plot(vlE, label='valLoss')
    plt.plot(vlPureE, label='valLossPure')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(os.path.join(saveDir, filenamePre + "-plotLosses.png"))
    plt.close()
    plt.plot(taE, label='trainingAcc')
    plt.plot(vaE, label='valAcc')
    plt.plot(vaPureE, label='valAccPure')
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.gca().yaxis.grid(True)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.savefig(os.path.join(saveDir, filenamePre + "-plotAcc.png"))
    plt.close()

#################################################################################################
# TRAIN CRITIC WITH ENC + ONE HOT WORD FEATURES WITH POSITIVE AND RANDOM NEGATIVE SAMPLES
#################################################################################################

tlS = []
vlS = []
vlPureS = []
taS = []
vaS = []
vaPureS = []
tlE = []
vlE = []
vlPureE = []
taE = []
vaE = []
vaPureE = []
tl = 0
ta = 0
vl = 0
va = 0

nEpochs = 11
batchSize = 128
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)

genTrainImages = genMouthImages(trainDirs, batchSize=batchSize, align=trainAlign, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)
genValImages = genMouthImages(valDirs, batchSize=batchSize, align=valAlign, wordIdx=wordIdx,
                              wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)

# For each epoch
for epoch in tqdm.tqdm(range(nEpochs)):
    # For each train batch
    for step in tqdm.tqdm(range(trainSteps)):
        vids, words = next(genTrainImages)
        words = np.argmax(words[:, 0, :], axis=1)
        # Inputs1
        inputs1 = np.concatenate([vids, vids])
        # Inputs2
        predWordFeatures = encoder.predict(vids)
        wrongPredWordFeatures = np.vstack((predWordFeatures[-2:], predWordFeatures[:-2]))
        inputs2 = np.concatenate([predWordFeatures, wrongPredWordFeatures])
        # Inputs3
        predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
        oneHotPredWords = np_utils.to_categorical(predWords, wordsVocabSize)
        wrongPredWords = np.hstack((predWords[-2:], predWords[:-2]))
        wrongOneHotPredWords = np.vstack((oneHotPredWords[-2:], oneHotPredWords[:-2]))
        inputs3 = np.concatenate([oneHotPredWords, wrongOneHotPredWords])
        # Output
        correctPreds = np.array(words == predWords).astype(int)
        wrongPreds = np.array(words == wrongPredWords).astype(int)
        outputs = np.concatenate([correctPreds, wrongPreds])
        fullIdx = list(range(len(inputs1)))
        np.random.shuffle(fullIdx)
        inputs1 = inputs1[fullIdx]
        inputs2 = inputs2[fullIdx]
        inputs3 = inputs3[fullIdx]
        outputs = outputs[fullIdx]
        h = criticModelWithEncAndOneHotWord.fit([inputs1, inputs2, inputs3], outputs,
                            batch_size=batchSize, epochs=1, initial_epoch=0)
        tl = h.history['loss'][0]
        ta = h.history['acc'][0]
        tlS.append(tl)
        taS.append(ta)
    # For each val batch
    for step in tqdm.tqdm(range(valSteps)):
        vids, words = next(genValImages)
        words = np.argmax(words[:, 0, :], axis=1)
        # Inputs1
        inputs1 = np.concatenate([vids, vids])
        # Inputs2
        predWordFeatures = encoder.predict(vids)
        wrongPredWordFeatures = np.vstack((predWordFeatures[-2:], predWordFeatures[:-2]))
        inputs2 = np.concatenate([predWordFeatures, wrongPredWordFeatures])
        # Inputs3
        predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
        oneHotPredWords = np_utils.to_categorical(predWords, wordsVocabSize)
        wrongPredWords = np.hstack((predWords[-2:], predWords[:-2]))
        wrongOneHotPredWords = np.vstack((oneHotPredWords[-2:], oneHotPredWords[:-2]))
        inputs3 = np.concatenate([oneHotPredWords, wrongOneHotPredWords])
        # Output
        correctPreds = np.array(words == predWords).astype(int)
        wrongPreds = np.array(words == wrongPredWords).astype(int)
        outputs = np.concatenate([correctPreds, wrongPreds])
        vl, va = criticModelWithEncAndOneHotWord.evaluate(
            [inputs1, inputs2, inputs3], outputs, batch_size=batchSize)
        vlS.append(vl)
        vaS.append(va)
        vl, va = criticModelWithEncAndOneHotWord.evaluate(
            [vids, predWordFeatures, oneHotPredWords], correctPreds, batch_size=batchSize)
        vlPureS.append(vl)
        vaPureS.append(va)
    # Append values
    tl = np.mean(tlS)
    tlE.append(tl)
    tlS = []
    ta = np.mean(taS)
    taE.append(ta)
    taS = []
    vl = np.mean(vlS)
    vlE.append(vl)
    vlS = []
    va = np.mean(vaS)
    vaE.append(va)
    vaS = []
    vlP = np.mean(vlPureS)
    vlPureE.append(vlP)
    vlPureS = []
    vaP = np.mean(vaPureS)
    vaPureE.append(vaP)
    vaPureS = []
    print(
        "epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-vlPure{5:.4f}-vaPure{6:.4f}".format(epoch, tl, ta, vl, va, vlP, vaP))
    criticModelWithEncAndOneHotWord.save_weights(os.path.join(
        saveDir, filenamePre + "-epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-vlPure{5:.4f}-vaPure{6:.4f}.hdf5".format(epoch, tl, ta, vl, va, vlP, vaP)))
    plt.plot(tlE, label='trainingLoss')
    plt.plot(vlE, label='valLoss')
    plt.plot(vlPureE, label='valLossPure')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(os.path.join(saveDir, filenamePre + "-plotLosses.png"))
    plt.close()
    plt.plot(taE, label='trainingAcc')
    plt.plot(vaE, label='valAcc')
    plt.plot(vaPureE, label='valAccPure')
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.gca().yaxis.grid(True)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.savefig(os.path.join(saveDir, filenamePre + "-plotAcc.png"))
    plt.close()


#######################################################
# SVM, input - LR 64 encoded + LR oneHotWord predict
#######################################################

nEpochs = 10
batchSize = 128
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)

genTrainImages = genMouthImages(trainDirs, batchSize=batchSize, align=trainAlign, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)
genValImages = genMouthImages(valDirs, batchSize=batchSize, align=valAlign, wordIdx=wordIdx,
                              wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)

svmTrainInput = np.empty((0, LRfeaturesDim + wordsVocabSize))
svmTrainOutput = np.empty((0, 1))
for step in range(trainSteps):
    vids, words = next(genTrainImages)
    words = np.argmax(words[:, 0, :], axis=1)
    videoFeatures = encoder.predict(vids)
    predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
    correctPreds = np.array(words == predWords).astype(int)
    oneHotPredWords = np_utils.to_categorical(predWords, wordsVocabSize)
    fullFeatures = np.hstack((videoFeatures, oneHotPredWords))
    svmInput = np.vstack((svmInput, fullFeatures))
    svmOutput = np.vstack((svmOutput, np.expand_dims(correctPreds, axis=1)))

svmValInput = np.empty((0, LRfeaturesDim + wordsVocabSize))
svmValOutput = np.empty((0, 1))
for step in range(valSteps):
    vids, words = next(genValImages)
    words = np.argmax(words[:, 0, :], axis=1)
    videoFeatures = encoder.predict(vids)
    predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
    correctPreds = np.array(words == predWords).astype(int)
    oneHotPredWords = np_utils.to_categorical(predWords, wordsVocabSize)
    fullFeatures = np.hstack((videoFeatures, oneHotPredWords))
    svmInput = np.vstack((svmInput, fullFeatures))
    svmOutput = np.vstack((svmOutput, np.expand_dims(correctPreds, axis=1)))


###########################################################################
# SVM, input - Critic Video encoding + LR 64 encoded + LR word predict
###########################################################################
# TODO
nEpochs = 20
batchSize = 128
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)

genTrainImages = genMouthImages(trainDirs, batchSize=batchSize, align=trainAlign, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)
genValImages = genMouthImages(valDirs, batchSize=batchSize, align=valAlign, wordIdx=wordIdx,
                              wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)

for step in range(trainSteps):
    vids, words = next(genTrainImages)
    words = np.argmax(words[:, 0, :], axis=1)
    videoFeatures = encoder.predict(vids)
    predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1) / wordsVocabSize
    correctPreds = np.array(words == predWords).astype(int)




