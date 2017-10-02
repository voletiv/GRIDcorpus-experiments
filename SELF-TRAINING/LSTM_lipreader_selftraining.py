import numpy as np
import random as rn
import os
import time

# import tensorflow as tf
# with tf.device('/cpu:0'):

from keras.callbacks import EarlyStopping

#################################################################
# IMPORT
#################################################################

from params import *
from gen_these_word_images import *
from C3D_critic_function import *
from LSTM_lipreader_function import *
from load_image_dirs_and_word_numbers import *
from LSTM_lipreader_selftraining_functions import *

#################################################################
# LSTM LipReader MODEL
#################################################################

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
    LSTMLipReaderModel, LSTMEncoder, fileNamePre = LSTM_lipreader(useMask=useMask,
        hiddenDim=hiddenDim, depth=depth, LSTMactiv=LSTMactiv, encodedDim=encodedDim,
        encodedActiv=encodedActiv, optimizer=optimizer, lr=lr)
    return LSTMLipReaderModel, LSTMEncoder, fileNamePre


#############################################################
# LOAD CRITIC
#############################################################

criticModel = None

# NEW
reverseImageSequence = True
wordsVocabSize = 51
criticModel, fileNamePre = C3D_critic(layer1Filters=4, layer2Filters=4, layer3Filters=4, fc1Nodes=4, vidFeaturesDim=16,
                                useWord=False, wordDim=1,
                                useEncWord=False, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=True, oneHotWordDim=wordsVocabSize, useOneHotWordFc=True, oneHotWordFcDim=16,
                                usePredWordDis=False, predWordDisDim=wordsVocabSize,
                                outputHDim=64, lr=1e-3)
criticModel.load_weights(os.path.join(saveDir, "TRAINED-WEIGHTS", "C3DCritic-l1f4-l2f4-l3f4-fc1n4-vid16-OHWord51-OHWordFc16-out64-Adam-1e-03-GRIDcorpus-s0107-09-epoch016-tl0.4776-ta0.7657-vl0.6650-va0.6450-sil0.6415-sia0.6669.hdf5"))


#############################################################
# LOAD IMAGES
#############################################################

trainDirs, trainWordNumbers, valDirs, valWordNumbers, siDirs, siWordNumbers = load_image_dirs_and_word_numbers(
    trainValSpeakersList=[1, 2, 3, 4, 5, 6, 7, 10],
    siList = [13, 14])

# fileNamePre += "-GRIDcorpus-s0107-10-si-s1314"

#################################################################
# 10%
#################################################################

labelledPercent = 10

# Starting training with only 10% of training data,
# and assuming the rest 90% to be unlabelled

# Split into labelled and unlabelled training data
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)
trainIdx = np.arange(len(trainDirs))
np.random.shuffle(trainIdx)
trainLabelledIdx = trainIdx[:int(labelledPercent/100*len(trainDirs))]
trainUnlabelledIdx = trainIdx[int(labelledPercent/100*len(trainDirs)):]
trainLabelledDirs = np.array(trainDirs)[trainLabelledIdx]
trainLabelledWordNumbers = np.array(trainWordNumbers)[trainLabelledIdx]
trainUnlabelledDirs = np.array(trainDirs)[trainUnlabelledIdx]
trainUnlabelledWordNumbers = np.array(trainWordNumbers)[trainUnlabelledIdx]

# Decide batchSize, nEpochs
batchSize = 512
nEpochs = 100

# Make Generating Functions
# genTrainImages = gen_these_word_images(trainDirs, trainWordNumbers, batchSize=batchSize, shuffle=False)
# trainSteps = len(trainDirs) // batchSize
genValImages = gen_these_word_images(valDirs, valWordNumbers, batchSize=batchSize, shuffle=False)
valSteps = len(valDirs) // batchSize
genSiImages = gen_these_word_images(siDirs, siWordNumbers, batchSize=batchSize, shuffle=False)
siSteps = len(siDirs) // batchSize

# Find out correct labels for labelled data

# First, use batchSize = 512
genTrainImagesLabelled = gen_these_word_images(trainLabelledDirs, trainLabelledWordNumbers, batchSize=batchSize, shuffle=False)
trainLabelledSteps = len(trainLabelledDirs) // batchSize
trainLabelledWords = np.empty((0))
for i in tqdm.tqdm(range(trainLabelledSteps)):
    _, words = next(genTrainImagesLabelled)
    for word in words:
        trainLabelledWords = np.append(trainLabelledWords, np.argmax(word))

# Then use batchSize = 1 for the remaining
genTrainImagesLabelledRemaining = gen_these_word_images(trainLabelledDirs[trainLabelledSteps * batchSize:], trainLabelledWordNumbers[trainLabelledSteps * batchSize:], batchSize=1, shuffle=False)
trainLabelledRemainingSteps = len(trainLabelledDirs) - trainLabelledSteps * batchSize
for i in tqdm.tqdm(range(trainLabelledRemainingSteps)):
    _, words = next(genTrainImagesLabelledRemaining)
    for word in words:
        trainLabelledWords = np.append(trainLabelledWords, np.argmax(word))

# All losses and accuracies thru self learning
# List of lists of all losses and accuracies thru iterations of self-learning
allApparentLabelledTrainLossesThruSelfLearning = []
allValLossesThruSelfLearning = []
allSiLossesThruSelfLearning = []
allApparentLabelledTrainAccuraciesThruSelfLearning = []
allValAccuraciesThruSelfLearning = []
allSiAccuraciesThruSelfLearning = []

# Losses and accuracies thru pc of labelled data
# List of final losses and accuracies thru iterations of self-learning
percentageOfLabelledData = []
apparentLabelledTrainLossesThruPcOfLabelledData = []
trueLabelledTrainLossesThruPcOfLabelledData = []
valLossesThruPcOfLabelledData = []
siLossesThruPcOfLabelledData = []
apparentLabelledTrainAccuraciesThruPcOfLabelledData = []
trueLabelledTrainAccuraciesThruPcOfLabelledData = []
valAccuraciesThruPcOfLabelledData = []
siAccuraciesThruPcOfLabelledData = []

# List of trainDirs, etc. per iteration
trainLabelledIdxItersList = []
trainLabelledWordsItersList = []
trainUnlabelledIdxItersList = []

# Append trainLabelledIdxItersList, trainUnlabelledIdxItersList
trainLabelledIdxItersList.append(trainLabelledIdx)
trainLabelledWordsItersList.append(trainLabelledWords)
trainUnlabelledIdxItersList.append(trainUnlabelledIdx)

# To fit
unlabelledLRPredMaxValueThresh = 0.8

unlabelledCriticPredsYesThresh = 0.9
nIters = 100
initIter = 0

# Remodel or Finetune
remodel = False

# FIT
for iterNumber in range(initIter, nIters):
    print("\nITER", iterNumber, "\n")
    print("Training with", len(trainLabelledIdxItersList[-1]), "words")
    # Remodel
    if remodel or iterNumber == 0:
        LSTMLipReaderModel, LSTMEncoder, fileNamePre = make_LSTM_lipreader_model()
        # Change fileNamePre
        fileNamePre += "-GRIDcorpus-s0107-10-si-s1314"
        fileNamePre += "-{0:02d}".format(labelledPercent)
        fileNamePre += "PercentSelfTraining-LRthresh{0:.2f}".format(unlabelledLRPredMaxValueThresh)
        if criticModel is not None:
            fileNamePre += "-criticThresh{0:.2f}".format(unlabelledCriticPredsYesThresh)
        fileNamePre += "-iter00"
    # Remodel / finetune
    fileNamePre = '-'.join(fileNamePre.split('-')[:-1]) + '-iter{0:02d}'.format(iterNumber)
    print(fileNamePre)
    # Callbacks
    checkSIAndMakePlots = CheckSIAndMakePlots(genSiImages=genSiImages, siSteps=siSteps, plotColor='g', fileNamePre=fileNamePre)
    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=True)
    # Fit
    LSTMLipReaderModel, checkSIAndMakePlots = fit_on_labelled_data(iterNumber,
        trainLabelledDirs, trainLabelledWordNumbers, trainLabelledWords,
        LSTMLipReaderModel, checkSIAndMakePlots, earlyStop, genValImages, valSteps, batchSize, nEpochs)
    # Save losses and accs
    save_losses_and_accuracies(
        LSTMLipReaderModel, checkSIAndMakePlots,
        allApparentLabelledTrainLossesThruSelfLearning, allValLossesThruSelfLearning, allSiLossesThruSelfLearning,
        allApparentLabelledTrainAccuraciesThruSelfLearning, allValAccuraciesThruSelfLearning, allSiAccuraciesThruSelfLearning,
        trueLabelledTrainLossesThruPcOfLabelledData, apparentLabelledTrainLossesThruPcOfLabelledData,
        percentageOfLabelledData, trainLabelledDirs, trainLabelledWordNumbers, trainDirs, batchSize,
        valLossesThruPcOfLabelledData, siLossesThruPcOfLabelledData,
        trueLabelledTrainAccuraciesThruPcOfLabelledData, apparentLabelledTrainAccuraciesThruPcOfLabelledData,
        valAccuraciesThruPcOfLabelledData, siAccuraciesThruPcOfLabelledData
    )
    # Plot loss and accuracy through all epochs of all iterations of self-learning
    plot_all_losses_and_accuracies_thru_self_learning(
        allApparentLabelledTrainLossesThruSelfLearning, allValLossesThruSelfLearning, allSiLossesThruSelfLearning,
        allApparentLabelledTrainAccuraciesThruSelfLearning, allValAccuraciesThruSelfLearning, allSiAccuraciesThruSelfLearning,
        fileNamePre
    )
    # Plot loss and accuracy through progress of percentage of labelled data
    plot_losses_and_accuracies_thru_percentage_of_labelled_data(
        percentageOfLabelledData,
        apparentLabelledTrainLossesThruPcOfLabelledData, trueLabelledTrainLossesThruPcOfLabelledData,
        valLossesThruPcOfLabelledData, siLossesThruPcOfLabelledData,
        apparentLabelledTrainAccuraciesThruPcOfLabelledData, trueLabelledTrainAccuraciesThruPcOfLabelledData,
        valAccuraciesThruPcOfLabelledData, siAccuraciesThruPcOfLabelledData,
        fileNamePre
    )
    # Change data
    trainLabelledIdx, trainUnlabelledIdx, trainLabelledDirs, trainLabelledWordNumbers, trainLabelledWords, trainUnlabelledDirs, trainUnlabelledWordNumbers \
        = add_unlabelled_data_to_labelled_data(labelledPercent, iterNumber,
            trainLabelledIdx, trainUnlabelledIdx,
            trainLabelledDirs, trainLabelledWordNumbers, trainLabelledWords,
            trainUnlabelledDirs, trainUnlabelledWordNumbers,
            LSTMLipReaderModel, batchSize, fileNamePre,
            percentageOfLabelledData, unlabelledLRPredMaxValueThresh,
            criticModel, unlabelledCriticPredsYesThresh)
    # Append trainLabelledIdxItersList, trainUnlabelledIdxItersList
    trainLabelledIdxItersList.append(trainLabelledIdx)
    trainLabelledWordsItersList.append(trainLabelledWords)
    trainUnlabelledIdxItersList.append(trainUnlabelledIdx)
    # Save
    np.savez(os.path.join(saveDir, '-'.join(fileNamePre.split('-')[:-1])), "-variables.npz",
        allApparentLabelledTrainLossesThruSelfLearning=allApparentLabelledTrainLossesThruSelfLearning,
        allValLossesThruSelfLearning=allValLossesThruSelfLearning,
        allSiLossesThruSelfLearning=allSiLossesThruSelfLearning,
        allApparentLabelledTrainAccuraciesThruSelfLearning=allApparentLabelledTrainAccuraciesThruSelfLearning,
        allValAccuraciesThruSelfLearning=allValAccuraciesThruSelfLearning,
        allSiAccuraciesThruSelfLearning=allSiAccuraciesThruSelfLearning,
        percentageOfLabelledData=percentageOfLabelledData,
        apparentLabelledTrainLossesThruPcOfLabelledData=apparentLabelledTrainLossesThruPcOfLabelledData,
        trueLabelledTrainLossesThruPcOfLabelledData=trueLabelledTrainLossesThruPcOfLabelledData,
        valLossesThruPcOfLabelledData=valLossesThruPcOfLabelledData,
        siLossesThruPcOfLabelledData=siLossesThruPcOfLabelledData,
        apparentLabelledTrainAccuraciesThruPcOfLabelledData=apparentLabelledTrainAccuraciesThruPcOfLabelledData,
        trueLabelledTrainAccuraciesThruPcOfLabelledData=trueLabelledTrainAccuraciesThruPcOfLabelledData,
        valAccuraciesThruPcOfLabelledData=valAccuraciesThruPcOfLabelledData,
        siAccuraciesThruPcOfLabelledData=siAccuraciesThruPcOfLabelledData,
        trainLabelledIdxItersList=trainLabelledIdxItersList,
        trainLabelledWordsItersList=trainLabelledWordsItersList,
        trainUnlabelledIdxItersList=trainUnlabelledIdxItersList
    )


# FOR fine-tuning from iter01
tl = []
ta = []
vl = []
va = []
sil = []
sia = []
mediaDir = '/media/voletiv/01D2BF774AC76280/GRIDcorpusResults/SELF-TRAINING/1-10pc-finetuning-LR0.9/'
for file in sorted(glob.glob(os.path.join(mediaDir, "*iter00*"))):
    if 'epoch' in file:
        tl.append(float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[-6][2:]))
        ta.append(float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[-5][2:]))
        vl.append(float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[-4][2:]))
        va.append(float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[-3][2:]))
        sil.append(float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[-2][3:]))
        sia.append(float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[-1][3:]))

# Save
allApparentLabelledTrainLossesThruSelfLearning.append(tl)
allValLossesThruSelfLearning.append(vl)
allSiLossesThruSelfLearning.append(sil)
allApparentLabelledTrainAccuraciesThruSelfLearning.append(ta)
allValAccuraciesThruSelfLearning.append(va)
allSiAccuraciesThruSelfLearning.append(sia)
percentageOfLabelledData.append(labelledPercent)
apparentLabelledTrainLossesThruPcOfLabelledData.append(tl[-1])
trueLabelledTrainLossesThruPcOfLabelledData.append(tl[-1])
valLossesThruPcOfLabelledData.append(vl[-1])
siLossesThruPcOfLabelledData.append(sil[-1])
apparentLabelledTrainAccuraciesThruPcOfLabelledData.append(ta[-1])
trueLabelledTrainAccuraciesThruPcOfLabelledData.append(ta[-1])
valAccuraciesThruPcOfLabelledData.append(va[-1])
siAccuraciesThruPcOfLabelledData.append(sia[-1])

# Load Lip Reader Model
LSTMLipReaderModel, LSTMEncoder, fileNamePre = make_LSTM_lipreader_model()
fileNamePre += "-GRIDcorpus-s0107-10-si-s1314"
fileNamePre += "-{0:02d}".format(labelledPercent)
fileNamePre += "PercentSelfTraining-LRthresh{0:.2f}".format(unlabelledLRPredMaxValueThresh)
if criticModel is not None:
    fileNamePre += "-criticThresh{0:.2f}".format(unlabelledCriticPredsYesThresh)

fileNamePre += "-iter00"
LSTMLipReaderModel.load_weights(os.path.join(mediaDir, 'LSTMLipReader-revSeq-Mask-LSTMh256-tanh-depth2-enc64-relu-adam-1e-03-tMouth-valMouth-NOmeanSub-GRIDcorpus-s0107-10-si-s1314-10PercentSelfTraining-LRthresh0.90-iter00-epoch079-tl1.1377-ta0.6460-vl1.5886-va0.5360-sil3.9002-sia0.2181.hdf5'))
