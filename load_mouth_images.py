import glob
import math
import numpy as np
import os
import tqdm

from keras.utils import np_utils

# Import params
from params import *


# Load Mouth Images
def load_mouth_images(allDirs=None, mode="train", imagesDir=None, align=False,
                        useMeanMouthImage=False, meanMouthImage=None, keepPadResults=False):
    # If directories are not specified
    if allDirs == None:
        # If the images directory (eg. /home/voletiv/Downloads/GRIDcorpus/) is not mentioned
        if imagesDir == None:
            imagesDir = rootDir
        # TRAIN or VAL mode
        if mode == "train" or mode == 'val':
            allDirs = sorted(glob.glob(os.path.join(imagesDir, "s0[0-7|9]/*")))
            # if mode == 'train':
            #     np.random.shuffle(allDirs)
        if mode == "si" :
            allDirs = sorted(glob.glob(os.path.join(imagesDir, "s1[0-9]/*")))
    X = np.zeros((len(allDirs) * wordsPerVideo,
                  framesPerWord, nOfMouthPixels))
    if keepPadResults:
        y = np.zeros((len(allDirs) * wordsPerVideo, 2, wordsVocabSize))
    else:
        y = np.zeros((len(allDirs) * wordsPerVideo, wordsVocabSize))
    # For each dir
    for vid, vidDir in enumerate(tqdm.tqdm(allDirs)):
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
        try:
            wordTimeData = open(alignFile).readlines()
        except:
            print("Could not read align file!")
            return X, y
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
                # If to be subtracted by the mean image
                if useMeanMouthImage:
                    if meanMouthImage == None:
                        print("Mean mouth image to be used but not provided!")
                        return X, y
                else:
                    meanMouthImage = np.zeros((nOfMouthPixels,))
                # Note the corresponding mouth image in greyscale,
                # padded with zeros before the frames, if len(wordMouthFiles) < framesPerWord),
                if reverseImageSequence:
                    # in reverse order of frames.
                    # eg. If there are 7 frames: 0 0 0 0 0 0 0 7 6 5 4 3 2 1
                    wordImages[-f-1] = np.reshape(cv2.imread(wordMouthFrame, 0) / 255., (1600,)) - meanMouthImage
                else:
                    # in same order
                    # eg. If there are 7 frames: 0 0 0 0 0 0 1 2 3 4 5 6 7
                    wordImages[f + (framesPerWord-(min(len(wordMouthFiles), framesPerWord)))] \
                        = np.reshape(cv2.imread(wordMouthFrame, 0) / 255., (1600,)) - meanMouthImage
            # Save this in X
            X[vid * wordsPerVideo + word] = wordImages
    # Return
    return X, y
