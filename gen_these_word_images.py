# MAKE WORDS AS FRAME OUTPUTS FOR MANY-TO-MANY SETTING

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

import cv2
import glob
import math
import numpy as np
import os
import tqdm

from keras.utils import np_utils

# Import params
from params import *

def gen_these_word_images(allDirs, allWordNumbers, allWords=None, batchSize=32,
                        reverseImageSequence=True, wordsVocabSize=wordsVocabSize,
                        align=False, useMeanMouthImage=False, meanMouthImage=None,
                        shuffle=True, keepPadResults=False, verbose=False):
    np.random.seed(29)
    # Make copy so that allDirs' source is not affected by shuffle
    dirs = np.array(allDirs)
    wordNumbers = np.array(allWordNumbers)
    if allWords is not None:
        words = np.array(allWords)
    # Looping generator
    while 1:
        # Shuffling input list
        if shuffle is True:
            fullIdx = list(range(len(dirs)))
            np.random.shuffle(fullIdx)
            dirs = dirs[fullIdx]
            wordNumbers = wordNumbers[fullIdx]
            if allWords is not None:
                words = words[fullIdx]
        # For each slice of batchSize number of files:
        for batch in range(0, len(dirs), batchSize):
            # print("batch " + str(batch))
            # Initializing output variables
            X = np.zeros((batchSize, framesPerWord, nOfMouthPixels))
            if keepPadResults:
                y = np.zeros((batchSize, 2, wordsVocabSize))
            else:
                y = np.zeros((batchSize, wordsVocabSize))
            # For each video in the batch
            batchDirs = dirs[batch:batch + batchSize]
            batchWordNumbers = wordNumbers[batch:batch + batchSize]
            if allWords is not None:
                batchWords = words[batch:batch + batchSize]
            # print(batchDirs[:10])
            # If at the end, there are lesser number of video files than
            # batchSize
            if len(batchDirs) < batchSize:
                continue
            # Else
            for i, (vidDir, wordNum) in enumerate(zip(batchDirs, batchWordNumbers)):
                # print("  vid " + + str(vid) + " " + str(vidDir))
                # Get the file names of the mouth images
                # If aligned mouth images are needed
                if verbose:
                    print(vidDir, wordNum)
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
                try:
                    wordTimeData = open(alignFile).readlines()
                except:
                    print("Could not read align file!")
                    yield (X, y)
                # Get the max time of the video
                maxClipDuration = float(wordTimeData[-1].split(' ')[1])
                # Remove Silent and Short Pauses
                for line in wordTimeData:
                    if 'sil' in line or 'sp' in line:
                        wordTimeData.remove(line)
                # Extract the word and save this word in y using one-hot
                # encoding
                if keepPadResults:
                    if allWords is not None:
                        y[i][0] = np_utils.to_categorical(batchWords[i], wordsVocabSize)
                    else:
                        y[i][0] = np_utils.to_categorical(
                            wordIdx[wordTimeData[wordNum].split(' ')[-1][:-1]], wordsVocabSize)
                else:
                    if allWords is not None:
                        y[i] = np_utils.to_categorical(batchWords[i], wordsVocabSize)
                    else:
                        y[i] = np_utils.to_categorical(
                            wordIdx[wordTimeData[wordNum].split(' ')[-1][:-1]], wordsVocabSize)
                # Initialize the array of images for this word
                wordImages = np.zeros((framesPerWord, nOfMouthPixels))
                # Find the start and end frame for this word
                wordStartFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
                                            0]) / maxClipDuration * framesPerVid)
                wordEndFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
                                          1]) / maxClipDuration * framesPerVid)
                # Note the file names
                wordMouthFiles = mouthFiles[
                    wordStartFrame:wordEndFrame + 1]
                # # If len(wordMouthFiles) > 14:
                # excessMouthLens.append(len(wordMouthFiles))
                # excessMouthLenNames.append(wordMouthFiles[0][:-11] + "_word" + str(word))
                # For blanks
                if keepPadResults and len(wordMouthFiles) < 14:
                    y[i][1][-1] = 1
                # For each frame of this word
                for f, wordMouthFrame in enumerate(wordMouthFiles[:framesPerWord]):
                    # If to be subtracted by the mean image
                    if useMeanMouthImage:
                        if meanMouthImage == None:
                            print(
                                "Mean mouth image to be used but not provided!")
                            yield (X, y)
                    else:
                        meanMouthImage = np.zeros((nOfMouthPixels,))
                    # Note the corresponding mouth image in greyscale,
                    # padded with zeros before the frames, if
                    # len(wordMouthFiles) < framesPerWord),
                    def read_and_save_image(wordImages, f, wordMouthFrame, meanMouthImage, wordMouthFiles, vidDir, wordNum):
                        try:
                            if reverseImageSequence:
                                # in reverse order of frames. eg. If there are 7 frames:
                                # 0 0 0 0 0 0 0 7 6 5 4 3 2 1
                                wordImages[-f - 1] = np.reshape(cv2.imread(
                                    wordMouthFrame, 0) / 255., (nOfMouthPixels,)) - meanMouthImage
                            else:
                                # in same order. eg. If there are 7 frames:
                                # 0 0 0 0 0 0 1 2 3 4 5 6 7
                                wordImages[f + (framesPerWord - (min(len(wordMouthFiles), framesPerWord)))] \
                                    = np.reshape(cv2.imread(wordMouthFrame, 0) / 255., (nOfMouthPixels,)) - meanMouthImage
                            return wordImages
                        except TypeError:
                            print(vidDir, wordNum)
                            read_and_save_image(wordImages, f, wordMouthFrame, meanMouthImage, wordMouthFiles, vidDir, wordNum)
                    wordImages = read_and_save_image(wordImages, f, wordMouthFrame, meanMouthImage, wordMouthFiles, vidDir, wordNum)
                # Save this in X
                X[i] = wordImages
            # Yield the results
            yield (X, y)
