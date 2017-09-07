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

from keras.layers import Reshape, Conv3D, BatchNormalization, Lambda, MaxPooling3D

from keras.layers import Flatten, Dense, concatenate, LSTM, RepeatVector

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

# To generate training images on the fly
# From https://github.com/fchollet/keras/issues/1627
# - Get a list of all of the files, and pass this list into the generator
# - In the generator:
#   - Infinite loop
#   - Shuffle the list of files
#   - For each slice of the shuffled files, where len(slice) == batch_size
#     - Open files, read to a single array with first shape[0] == batch_size; yield data
#     - Have an edge case to handle the case where batch_size is not a multiple of the number of files,
# such that the generator will always yield batch_size number of examples


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


def loadMouthImages(dirs, batchSize, align, wordIdx, wordsVocabSize=52, wordsPerVideo=6, framesPerVid=75, framesPerWord=14, nOfMouthPixels=1600,
                    useMeanMouthImage=False, meanMouthImage=dummyMeanMouthImage):
    X = np.zeros((len(dirs) * wordsPerVideo,
                  framesPerWord, nOfMouthPixels))
    y = np.zeros((len(dirs) * wordsPerVideo, 2, wordsVocabSize))
    # For each dir
    for vid, vidDir in enumerate(tqdm.tqdm(dirs)):
        # print("  vid " + + str(vid) + " " + str(vidDir))
        # Get the file names of the mouth images
        # If aligned mouth images are needed
        if align:
            mouthFiles = sorted(glob.glob(
                os.path.join(vidDir, '*Aligned*.jpg')))
        # If non-aligned mouth images are needed
        else:
            mouthFiles = sorted(glob.glob(os.path.join(vidDir, '*Mouth*.jpg')))
        # Frames per vid
        framesPerVid = len(mouthFiles)
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
            y[vid * wordsPerVideo + word][0] = np_utils.to_categorical(
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
            if len(wordMouthFiles) < 14:
                y[vid * wordsPerVideo + word][1][-1] = 1
            # For each frame of this word
            for f, wordMouthFrame in enumerate(wordMouthFiles[:framesPerWord]):
                # Note the corresponding mouth image in greyscale,
                # (padded with zeros before the frames, if len(wordMouthFiles) < framesPerWord)
                wordImages[f + (framesPerWord-(min(len(wordMouthFiles), framesPerWord)))] \
                    = np.reshape(cv2.imread(wordMouthFrame, 0) / 255., (1600,))
            # Save this in X
            X[vid * wordsPerVideo + word] = wordImages
    # Return
    return X, y


# # Number of excess words
# len(excessMouthLens)
# # Percentage of excess words
# len(excessMouthLens) / len(dirs)
# # Histogram of excess words
# np.histogram(excessMouthLens, bins=(np.max(excessMouthLens) - np.min(excessMouthLens) + 1),
#              range=(np.min(excessMouthLens), np.max(excessMouthLens) + 1))

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

# # Load the trainDirs (instead of generating them)
# trainX, trainY = loadMouthImages(trainDirs, batchSize=batchSize, align=trainAlign,
#                                  wordIdx=wordIdx, wordsVocabSize=wordsVocabSize)
# valX, valY = loadMouthImages(valDirs, batchSize=batchSize, align=valAlign,
#                              wordIdx=wordIdx, wordsVocabSize=wordsVocabSize)


#############################################################
# Seq2Seq LipReader MODEL
#############################################################

# Manual seeds
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)

hiddenDim = 256
depth = 2

myInput = Input(shape=(framesPerWord, nOfMouthPixels,))
attn = SimpleSeq2Seq(input_dim=nOfMouthPixels, hidden_dim=hiddenDim,
                     output_length=2, output_dim=wordsVocabSize, depth=depth)(myInput)
actn = Activation("softmax")(attn)
Seq2SeqLipReaderModel = Model(inputs=myInput, outputs=actn)

lr = 5e-4
adam = Adam(lr=lr)
Seq2SeqLipReaderModel.compile(optimizer=adam, loss='categorical_crossentropy',
                           metrics=['accuracy'])
Seq2SeqLipReaderModel.summary()

filenamePre = 'SimpleSeq2Seq-h' + \
    str(hiddenDim) + '-depth' + str(depth) + \
    '-Adam-%1.e' % lr + '-GRIDcorpus-s'

# # Save Model Architecture
# model_yaml = Seq2SeqLipReaderModel.to_yaml()
# with open(os.path.join(saveDir, "modelArch-" + filenamePre + ".yaml"), "w") as yaml_file:
#     yaml_file.write(model_yaml)


#############################################################
# TRAIN LIPREADER
#############################################################

nEpochs = 100
batchSize = 128     # num of speaker vids
initEpoch = 0

# Checkpoint
filepath = os.path.join(saveDir,
                        filenamePre + "-epoch{epoch:03d}-tl{loss:.4f}-ta{acc:.4f}-vl{val_loss:.4f}-va{val_acc:.4f}.hdf5")
checkpoint = ModelCheckpoint(
    filepath, verbose=1, save_best_only=False, save_weights_only=True, period=1)


class CheckpointSIAndPlots(Callback):
    def __init___(self, plotColor):
        self.plotColor = plotColor
    # On train start
    def on_train_begin(self, logs={}):
        self.losses = []
        self.valLosses = []
        self.siLosses = []
        self.acc = []
        self.valAcc = []
        self.siAcc = []
        # Define epochIndex
        def epochIndex(x):
            x = x.split('/')[-1].split('-')
            return [i for i, word in enumerate(x) if 'epoch' in word][0]
        # Define epochNoInFile
        def epochNoInFile(x):
            epochIdx = epochIndex(x)
            return x.split('/')[-1].split('-')[epochIdx]
        # For all weight files
        for file in sorted(glob.glob(os.path.join(rootDir, "*" + filenamePre + "*-epoch*")), key=epochNoInFile):
            print(file)
            epochIdx = epochIndex(file)
            self.losses.append(
                float(file.split('/')[-1].split('.')[:-1].split('-')[epochIdx + 1][2:]))
            self.acc.append(
                float(file.split('/')[-1].split('.')[:-1].split('-')[epochIdx + 2][2:]))
            self.valLosses.append(
                float(file.split('/')[-1].split('.')[:-1].split('-')[epochIdx + 3][2:]))
            self.valAcc.append(
                float(file.split('/')[-1].split('.')[:-1].split('-')[epochIdx + 4][2:]))
            self.siLosses.append(
                float(file.split('/')[-1].split('.')[:-1].split('-')[epochIdx + 5][3:]))
            self.siAcc.append(
                float(file.split('/')[-1].split('.')[:-1].split('-')[epochIdx + 6][3:]))
    # At every epoch
    def on_epoch_end(self, epoch, logs={}):
        tl = logs.get('loss')
        ta = logs.get('acc')
        vl = logs.get('val_loss')
        va = logs.get('val_acc')
        # Speaker-Independent
        print("Calculating speaker-independent loss and acc...")
        [sil, sia] = LSTMLipReaderModel.evaluate_generator(genMouthImages(siDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
            wordsVocabSize=wordsVocabSize, useMeanMouthImage=False, shuffle=False, keepPadResults=False), siValSteps)
        # Checkpoint
        filepath = os.path.join(saveDir,
                                filenamePre + "-epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-sil{5:.4f}-sia{6:.4f}.hdf5".format(epoch, tl, ta, vl, va, sil, sia))
        checkpoint = ModelCheckpoint(
            filepath, verbose=1, save_best_only=False, save_weights_only=True, period=1)
        print("Saving plots for epoch " + str(epoch))
        self.losses.append(tl)
        self.valLosses.append(vl)
        self.siLosses.append(sil)
        self.acc.append(ta)
        self.valAcc.append(va)
        self.siAcc.append(sia)
        plt.plot(self.losses, label='trainingLoss', color=self.plotColor, linestyle='-.')
        plt.plot(self.valLosses, label='valLoss', color=self.plotColor, linestyle='-')
        plt.plot(self.siLosses, label='siLoss', color=self.plotColor, linestyle='--')
        leg = plt.legend(loc='best', fontsize=11, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig(os.path.join(saveDir, filenamePre + "-plotLosses.png"))
        plt.close()
        plt.plot(self.acc, label='trainingAcc', color=self.plotColor, linestyle='-.')
        plt.plot(self.valAcc, label='valAcc', color=self.plotColor, linestyle='-')
        plt.plot(self.siAcc, label='siAcc', color=self.plotColor, linestyle='--')
        leg = plt.legend(loc='best', fontsize=11, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
        plt.gca().yaxis.grid(True)
        plt.savefig(os.path.join(saveDir, filenamePre + "-plotAcc.png"))
        plt.close()

checkpointSIAndPlots = CheckpointSIAndPlots('g')

# Load previous lipReaderModel
mediaDir = '/media/voletiv/01D2BF774AC76280/GRIDcorpusResults/1-SimpleSeq2Seq-h256-depth2-Adam-5e-04-GRIDcorpus-s0107-s0909-meanSub-tAlign-vAlign'
# load YAML and create model
# yaml_file = open(os.path.join(saveDir, 'modelArch-SimpleSeq2Seq-h256-depth2-Adam-5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub.yaml'), 'r'); loaded_model_yaml = yaml_file.read(); yaml_file.close()
# LSTMLipReaderModel = model_from_yaml(loaded_model_yaml)
# load weights into new model
print("Loaded model from disk")
Seq2SeqLipReaderModel.load_weights(os.path.join(
    saveDir, "SimpleSeq2Seq-h256-depth2-Adam-5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub-epoch059-tl0.1136-ta0.9045-vl0.3732-va0.8513.hdf5"))
# LSTMLipReaderModel.load_weights(os.path.join(
#     saveDir, "SimpleSeq2Seq-h256-depth2-Adam-5e-04-GRIDcorpus-s0107-s0909-tAlign-vAlign-NOmeanSub-epoch059-tl0.1027-ta0.9081-vl0.4214-va0.8372.hdf5"))
# LSTMLipReaderModel.load_weights(os.path.join(
#     saveDir, "SimpleSeq2Seq-h256-depth2-Adam-5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-meanSub-epoch059-tl0.0224-ta0.9361-vl0.4127-va0.8694.hdf5"))
# LSTMLipReaderModel.load_weights(os.path.join(
#     saveDir, "SimpleSeq2Seq-h256-depth2-Adam-5e-04-GRIDcorpus-s0107-s0909-tAlign-vAlign-meanSub-epoch059-tl0.0256-ta0.9327-vl0.3754-va0.8746.hdf5"))
initEpoch = 60

# FIT (gen)
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)
LSTMLipReaderHistory = LSTMLipReaderModel.fit_generator(genMouthImages(trainDirs, batchSize=batchSize, align=False, wordIdx=wordIdx, wordsVocabSize=wordsVocabSize,
                                                                       useMeanMouthImage=False, shuffle=True, shuffleWords=False, keepPadResults=True),
                                                        steps_per_epoch=trainSteps, epochs=nEpochs, verbose=1, callbacks=[checkpoint, lossHistory],
                                                        validation_data=genMouthImages(valDirs, batchSize=batchSize, align=False, wordIdx=wordIdx, wordsVocabSize=wordsVocabSize,
                                                                                       useMeanMouthImage=False, shuffle=True, shuffleWords=False, keepPadResults=True),
                                                        validation_steps=valSteps, workers=1, initial_epoch=initEpoch)

# # FIT (load)
# history = model.fit(trainX, trainY, batch_size=batchSize, epochs=nEpochs, verbose=1, callbacks=[
# checkpoint, lossHistory], validation_data=(valX, valY),
# initial_epoch=initEpoch)
