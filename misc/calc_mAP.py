import os
import numpy as np
import glob
import tqdm
import matplotlib
# matplotlib.use('agg')     # Use this for remote terminals
import matplotlib.pyplot as plt

from keras.utils import np_utils

#############################################################
# IMPORT
#############################################################

from params import *
from gen_mouth_images import *
from load_mouth_images import *
from LSTM_lipreader_function import *
from C3D_critic_function import *


#############################################################
# PARAMS TO BE SET
#############################################################

rootDir = '/home/voletiv/Datasets/GRIDcorpus'
saveDir = rootDir
# rootDir = '/Neutron6/voleti.vikram/GRIDcorpus'
# # saveDir = '/Neutron6/voleti.vikram/GRIDcorpusResults/1-h32-d1-Adam-2e-4-s0107-s0920-s2227'
# saveDir = rootDir


#############################################################
# LOAD IMAGES
#############################################################

# TRAIN AND VAL
trainValSpeakersList = [1, 2, 3, 4, 5, 6, 7]
valSplit = 0.1
trainDirs = []
valDirs = []
np.random.seed(29)
# For each speaker
for speaker in sorted(tqdm.tqdm(trainValSpeakersList)):
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
siList = [10, 11]
siDirs = []
for speaker in sorted(tqdm.tqdm(siList)):
    speakerDir = os.path.join(rootDir, 's' + '{0:02d}'.format(speaker))
    vidDirs = sorted(glob.glob(os.path.join(speakerDir, '*/')))
    for i in fullListIdx:
            siDirs.append(vidDirs[i])

# Numbers
print("No. of speaker-independent videos: " + str(len(siDirs)))


######################################################################
# BEST LIPREADER MODEL
######################################################################

# NEW; wordsVocabSize = 51; reverse=True
wordsVocabSize = 51
useMask = True
hiddenDim = 256
depth = 2
LSTMactiv = 'tanh'
encodedDim = 64
encodedActiv = 'relu'
lr = 1e-3
LSTMLipReaderModel, LSTMEncoder, fileNamePre = LSTM_lipreader(wordsVocabSize=wordsVocabSize,
    useMask=useMask, hiddenDim=hiddenDim, depth=depth, LSTMactiv=LSTMactiv,
    encodedDim=encodedDim, encodedActiv=encodedActiv, lr=lr)
LSTMLipReaderModel.load_weights(os.path.join(saveDir, "LSTMLipReader-revSeq-Mask-LSTMh256-LSTMactivtanh-depth2-enc64-encodedActivrelu-Adam-1e-03-GRIDcorpus-s0107-09-tMouth-valMouth-NOmeanSub-epoch079-tl0.2325-ta0.9232-vl0.5391-va0.8464-sil4.8667-sia0.2436.hdf5"))
reverseImageSequence = True

# LSTMLipReader-revSeq-Mask-LSTMh256-LSTMactivtanh-depth2-enc64-encodedActivrelu-Adam-1e-03-GRIDcorpus-s0107-09-tMouth-valMouth-NOmeanSub-epoch000-tl3.4336-ta0.0947-vl3.1424-va0.1332-sil3.3480-sia0.0970

######################################################################
# C3D CRITIC MODEL with one-hot Word input and 1 output Hidden layer
######################################################################

# OLD
reverseImageSequence = False
wordsVocabSize = 52
criticModel, fileNamePre = C3D_critic(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=False, wordDim=1,
                                useEncWord=True, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=True, oneHotWordDim=wordsVocabSize, useOneHotWordFc=False, oneHotWordFcDim=16,
                                usePredWordDis=False, predWordDisDim=wordsVocabSize,
                                outputHDim=64, lr=1e-3)
criticModel.load_weights(os.path.join(saveDir, "C3DCritic-LRnoPadResults-l1f8-l2f16-l3f32-fc1n64-vid64-enc64-oneHotWord52-out64-Adam-5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub-epoch002-tl0.2837-ta0.8783-vl0.4017-va0.8255-sil1.4835-sia0.3520.hdf5"))

# NEW
reverseImageSequence = True
wordsVocabSize = 51
criticModel, fileNamePre = C3D_critic(layer1Filters=4, layer2Filters=4, layer3Filters=4, fc1Nodes=4, vidFeaturesDim=16,
                                useWord=False, wordDim=1,
                                useEncWord=False, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=True, oneHotWordDim=wordsVocabSize, useOneHotWordFc=True, oneHotWordFcDim=16,
                                usePredWordDis=False, predWordDisDim=wordsVocabSize,
                                outputHDim=64, lr=1e-3)
criticModel.load_weights(os.path.join(saveDir, "C3DCritic-l1f4-l2f4-l3f4-fc1n4-vid16-OHWord51-OHWordFc16-out64-Adam-1e-03-GRIDcorpus-s0107-09-epoch016-tl0.4776-ta0.7657-vl0.6650-va0.6450-sil0.6415-sia0.6669.hdf5"))

tl = []
ta = []
vl = []
va = []
sil = []
sia = []
for file in sorted(files):
     if 'epoch' in file:
            tl.append(float(file.split('-')[16][2:]))
            ta.append(float(file.split('-')[17][2:]))
            vl.append(float(file.split('-')[18][2:]))
            va.append(float(file.split('-')[19][2:]))
            sil.append(float(file.split('-')[20][3:]))
            sia.append(float(file.split('-')[21][3:-5]))

plt.subplot(121)
plt.plot(tl, label='trainingAcc', color='g', linestyle='-.')
plt.plot(vl, label='valAcc', color='g', linestyle='-')
plt.plot(sil, label='siAcc', color='g', linestyle='--')
leg = plt.legend(loc='best', fontsize=11, fancybox=True)
leg.get_frame().set_alpha(0.3)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title("Loss")
plt.subplot(122)
plt.plot(ta, label='trainingAcc', color='g', linestyle='-.')
plt.plot(va, label='valAcc', color='g', linestyle='-')
plt.plot(sia, label='siAcc', color='g', linestyle='--')
leg = plt.legend(loc='best', fontsize=11, fancybox=True)
leg.get_frame().set_alpha(0.3)
plt.xlabel('epochs')
plt.ylabel('acc')
plt.yticks(np.arange(0, 1.05, 0.05))
plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.gca().yaxis.grid(True)
plt.title("Accuracy")
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.suptitle("LSTMLipReader-revSeq-Mask-LSTMh256-LSTMactivtanh-depth2-enc64-relu-Adam-1e-03")
plt.savefig(os.path.join(saveDir, "aa.png"))
plt.close()


######################################################################
# EVALUATION
######################################################################

batchSize = 128     # num of speaker vids
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)
siSteps = int(len(siDirs) / batchSize)

# Generating functions
genTrainImages = gen_mouth_images(trainDirs, batchSize=batchSize, reverseImageSequence=reverseImageSequence, wordsVocabSize=wordsVocabSize, shuffle=True, shuffleWords=True)
genValImages = gen_mouth_images(valDirs, batchSize=batchSize, reverseImageSequence=reverseImageSequence, wordsVocabSize=wordsVocabSize, shuffle=False, shuffleWords=False)
genSiImages = gen_mouth_images(siDirs, batchSize=batchSize, reverseImageSequence=reverseImageSequence, wordsVocabSize=wordsVocabSize, shuffle=False, shuffleWords=False)

# TRAIN RESULTS

LRTrainPredsFull = np.empty((0, wordsVocabSize))
LRTrainCorrectWordScores = np.empty((0))
LRTrainPredsBinary = np.empty((0), int)     # Whether the LR Prediction is correct or not
LRTrainTopNPredsBinary = np.empty((0), int)
LRTrainPredWordsAll = []
trainActualWords = []
criticTrainPredsAll = []
criticTrainPredsScore = np.empty((0), int)     # What the Critic predicted
for step in tqdm.tqdm((range(trainSteps))):
    # Ground truth
    vids, ohWords = next(genTrainImages)
    words = np.argmax(ohWords, axis=1)
    trainActualWords = np.append(trainActualWords, words)
    # Lip Reader predictions
    LRPredWordScores = LSTMLipReaderModel.predict(vids)
    LRTrainCorrectWordScores = np.append(LRTrainCorrectWordScores, [wordScores[words[i]] for i, wordScores in enumerate(LRPredWordScores)])
    LRTrainPredsFull = np.vstack((LRTrainPredsFull, LRPredWordScores))
    LRPredWords = np.argmax(LRPredWordScores, axis=1)
    LRTrainPredWordsAll = np.append(LRTrainPredWordsAll, LRPredWords)
    LRTrainPredsBinary = np.append(LRTrainPredsBinary, np.array(words == LRPredWords).astype(int))
    # # LIPREADER topN predictions
    # LRPredTopNWords = np.argsort(LRPredWordScores, axis=1)[:, -topN:]
    # for i in range(len(words)):
    #     if words[i] in LRPredTopNWords[i]:
    #         LRTrainTopNPredsBinary = np.append(LRTrainTopNPredsBinary, 1)
    #     else:
    #         LRTrainTopNPredsBinary = np.append(LRTrainTopNPredsBinary, 0)
    # Critic predictions
    criticPreds = criticModel.predict([vids, np_utils.to_categorical(LRPredWords, wordsVocabSize)])
    criticTrainPredsScore = np.append(criticTrainPredsScore, criticPreds)
    criticTrainPredsAll = np.append(criticTrainPredsAll, np.array(criticPreds>0.5, int))

# LR Softmax output
LRTrainCorrectWordScores = np.array([predWordScores[int(trainActualWords[i])] for i, predWordScores in enumerate(LRTrainPredsFull)])
LRTrainCorrectCountsPerThresh = []
manyThresh = np.arange(0, 1.001, 0.001)
for t in manyThresh:
    LRTrainCorrectCountsPerThresh.append(np.sum(LRTrainCorrectWordScores > t)/len(LRTrainCorrectWordScores))


# Evaluation with Thresholds
trainTotalPopulation = len(criticTrainPredsScore)
thresholds = np.arange(0, 1.001, 0.05)
trainTPPerThresh = np.zeros((len(thresholds)))
trainFPPerThresh = np.zeros((len(thresholds)))
trainFNPerThresh = np.zeros((len(thresholds)))
trainTNPerThresh = np.zeros((len(thresholds)))
for t, thresh in enumerate(thresholds):
    criticTrainPredsBinary = criticTrainPredsScore > thresh
    trainTPPerThresh[t] = np.sum(np.multiply((LRTrainPredsBinary == 1), (criticTrainPredsBinary == 1), dtype=int))
    trainFPPerThresh[t] = np.sum(np.multiply((LRTrainPredsBinary == 0), (criticTrainPredsBinary == 1), dtype=int))
    trainFNPerThresh[t] = np.sum(np.multiply((LRTrainPredsBinary == 1), (criticTrainPredsBinary == 0), dtype=int))
    trainTNPerThresh[t] = np.sum(np.multiply((LRTrainPredsBinary == 0), (criticTrainPredsBinary == 0), dtype=int))

# Precision = TP/(TP + FP)
# Not applicable for threshold = 1, because all are negative
trainPrecisionPerThresh = np.divide(trainTPPerThresh, trainTPPerThresh + trainFPPerThresh)

# Recall or Sensitivity = TP/(TP + FN)
trainSensitivityPerThresh = np.divide(trainTPPerThresh, trainTPPerThresh + trainFNPerThresh)

# Specificity = TN/(TN + FP)
trainSpecificityPerThresh = np.divide(trainTNPerThresh, trainTNPerThresh + trainFPPerThresh)

# Accuracy = (TP + TN)/Total
trainAccuracyPerThresh = (trainTPPerThresh + trainTNPerThresh) / trainTotalPopulation

print(trainAccuracyPerThresh[10])

# # Save
# np.savez(os.path.join(saveDir, "trainDataResults.npz"), LRTrainPredsBinary=LRTrainPredsBinary, criticTrainPredsScore=criticTrainPredsScore,
#     trainTPPerThresh=trainTPPerThresh, trainFPPerThresh=trainFPPerThresh, trainFNPerThresh=trainFNPerThresh, trainTNPerThresh=trainTNPerThresh,
#     trainPrecisionPerThresh=trainPrecisionPerThresh, trainSensitivityPerThresh=trainSensitivityPerThresh, trainSpecificityPerThresh=trainSpecificityPerThresh, trainAccuracyPerThresh=trainAccuracyPerThresh)

# VAL RESULTS

LRValPredsFull = np.empty((0, wordsVocabSize))
LRValCorrectWordScores = np.empty((0))
LRValPredsBinary = np.empty((0), int)     # Whether the LR Prediction is correct or not
LRValCorrectWordScores = np.empty((0))
LRValTopNPredsBinary = np.empty((0), int)
criticValPredsScore = np.empty((0), int)     # What the Critic predicted
LRValPredWordsAll = []
valActualWords = []
criticPredsAll = []
criticTrainPredsScore = np.empty((0), int)     # What the Critic predicted
for step in tqdm.tqdm((range(valSteps))):
    # Ground truth
    vids, ohWords = next(genValImages)
    words = np.argmax(ohWords, axis=1)
    valActualWords = np.append(valActualWords, words)
    # Lip Reader predictions
    LRPredWordScores = LSTMLipReaderModel.predict(vids)
    LRValPredsFull = np.vstack((LRValPredsFull, LRPredWordScores))
    LRPredWords = np.argmax(LRPredWordScores, axis=1)
    LRValPredWordsAll = np.append(LRValPredWordsAll, LRPredWords)
    LRValPredsBinary = np.append(LRValPredsBinary, np.array(words == LRPredWords).astype(int))
    # # LIPREADER topN predictions
    # LRPredTopNWords = np.argsort(LRPredWordScores, axis=1)[:, -topN:]
    # for i in range(len(words)):
    #     if words[i] in LRPredTopNWords[i]:
    #         LRValTopNPredsBinary = np.append(LRValTopNPredsBinary, 1)
    #     else:
    #         LRValTopNPredsBinary = np.append(LRValTopNPredsBinary, 0)
    # Critic predictions
    criticPreds = criticModel.predict([vids, np_utils.to_categorical(LRPredWords, wordsVocabSize)])
    criticValPredsScore = np.append(criticValPredsScore, criticPreds)

# LR Softmax output
LRValCorrectWordScores = np.array([predWordScores[int(valActualWords[i])] for i, predWordScores in enumerate(LRValPredsFull)])
LRValCorrectCountsPerThresh = []
manyThresh = np.arange(0, 1.001, 0.001)
for t in manyThresh:
    LRValCorrectCountsPerThresh.append(np.sum(LRValCorrectWordScores > t)/len(LRValCorrectWordScores))

# Evaluation with Thresholds
valTotalPopulation = len(criticValPredsScore)
thresholds = np.arange(0, 1.001, 0.05)
valTPPerThresh = np.zeros((len(thresholds)))
valFPPerThresh = np.zeros((len(thresholds)))
valFNPerThresh = np.zeros((len(thresholds)))
valTNPerThresh = np.zeros((len(thresholds)))
for t, thresh in enumerate(thresholds):
    criticValPredsBinary = criticValPredsScore > thresh
    valTPPerThresh[t] = np.sum(np.multiply((LRValPredsBinary == 1), (criticValPredsBinary == 1), dtype=int))
    valFPPerThresh[t] = np.sum(np.multiply((LRValPredsBinary == 0), (criticValPredsBinary == 1), dtype=int))
    valFNPerThresh[t] = np.sum(np.multiply((LRValPredsBinary == 1), (criticValPredsBinary == 0), dtype=int))
    valTNPerThresh[t] = np.sum(np.multiply((LRValPredsBinary == 0), (criticValPredsBinary == 0), dtype=int))

# Precision = TP/(TP + FP)
# Not applicable for threshold = 1, because all are negative
valPrecisionPerThresh = np.divide(valTPPerThresh, valTPPerThresh + valFPPerThresh)

# Recall or Sensitivity = TP/(TP + FN)
valSensitivityPerThresh = np.divide(valTPPerThresh, valTPPerThresh + valFNPerThresh)

# Specificity = TN/(TN + FP)
valSpecificityPerThresh = np.divide(valTNPerThresh, valTNPerThresh + valFPPerThresh)

# Accuracy = (TP + TN)/Total
valAccuracyPerThresh = (valTPPerThresh + valTNPerThresh) / valTotalPopulation

print(valAccuracyPerThresh[10])

# # Save
# np.savez(os.path.join(saveDir, "valDataResults.npz"), LRValPredsBinary=LRValPredsBinary, criticValPredsScore=criticValPredsScore,
#     valTPPerThresh=valTPPerThresh, valFPPerThresh=valFPPerThresh, valFNPerThresh=valFNPerThresh, valTNPerThresh=valTNPerThresh,
#     valPrecisionPerThresh=valPrecisionPerThresh, valSensitivityPerThresh=valSensitivityPerThresh, valSpecificityPerThresh=valSpecificityPerThresh, valAccuracyPerThresh=valAccuracyPerThresh)

# SPEAKER INDEPENDENT RESULTS

LRSiPredsFull = np.empty((0, wordsVocabSize))
LRSiPredsBinary = np.empty((0), int)     # Whether the LR Prediction is correct or not
LRSiTopNPredsBinary = np.empty((0), int)
LRSiPredWordsAll = []
siActualWords = []
criticPredsAll = []
criticSiPredsScore = np.empty((0), int)     # What the Critic predicted
for step in tqdm.tqdm((range(siSteps))):
    # Ground truth
    vids, ohWords = next(genSiImages)
    words = np.argmax(ohWords, axis=1)
    siActualWords = np.append(siActualWords, words)
    # Lip Reader predictions
    LRPredWordScores = LSTMLipReaderModel.predict(vids)
    LRSiPredsFull = np.vstack((LRSiPredsFull, LRPredWordScores))
    LRPredWords = np.argmax(LRPredWordScores, axis=1)
    LRSiPredsBinary = np.append(LRSiPredsBinary, np.array(words == LRPredWords).astype(int))
    # # LIPREADER topN predictions
    # LRPredTopNWords = np.argsort(LRPredWordScores, axis=1)[:, -topN:]
    # for i in range(len(words)):
    #     if words[i] in LRPredTopNWords[i]:
    #         LRSiTopNPredsBinary = np.append(LRSiTopNPredsBinary, 1)
    #     else:
    #         LRSiTopNPredsBinary = np.append(LRSiTopNPredsBinary, 0)
    # Critic predictions
    criticPreds = criticModel.predict([vids, np_utils.to_categorical(LRPredWords, wordsVocabSize)])
    criticSiPredsScore = np.append(criticSiPredsScore, criticPreds)

# LR Softmax output
LRSiCorrectWordScores = np.array([predWordScores[int(siActualWords[i])] for i, predWordScores in enumerate(LRSiPredsFull)])
LRSiCorrectCountsPerThresh = []
manyThresh = np.arange(0, 1.001, 0.001)
for t in manyThresh:
    LRSiCorrectCountsPerThresh.append(np.sum(LRSiCorrectWordScores > t)/len(LRSiCorrectWordScores))


# Evaluation with Thresholds
siTotalPopulation = len(criticSiPredsScore)
thresholds = np.arange(0, 1.001, 0.05)
siTPPerThresh = np.zeros((len(thresholds)))
siFPPerThresh = np.zeros((len(thresholds)))
siFNPerThresh = np.zeros((len(thresholds)))
siTNPerThresh = np.zeros((len(thresholds)))
for t, thresh in enumerate(thresholds):
    criticSiPredsBinary = criticSiPredsScore > thresh
    siTPPerThresh[t] = np.sum(np.multiply((LRSiPredsBinary == 1), (criticSiPredsBinary == 1), dtype=int))
    siFPPerThresh[t] = np.sum(np.multiply((LRSiPredsBinary == 0), (criticSiPredsBinary == 1), dtype=int))
    siFNPerThresh[t] = np.sum(np.multiply((LRSiPredsBinary == 1), (criticSiPredsBinary == 0), dtype=int))
    siTNPerThresh[t] = np.sum(np.multiply((LRSiPredsBinary == 0), (criticSiPredsBinary == 0), dtype=int))

# Precision = TP/(TP + FP)
# Not applicable for threshold = 1, because all are negative
siPrecisionPerThresh = np.divide(siTPPerThresh, siTPPerThresh + siFPPerThresh)

# Recall or Sensitivity = TP/(TP + FN)
siSensitivityPerThresh = np.divide(siTPPerThresh, siTPPerThresh + siFNPerThresh)

# Specificity = TN/(TN + FP)
siSpecificityPerThresh = np.divide(siTNPerThresh, siTNPerThresh + siFPPerThresh)

# Accuracy = (TP + TN)/Total
siAccuracyPerThresh = (siTPPerThresh + siTNPerThresh) / siTotalPopulation

print(siAccuracyPerThresh[10])

# # Save
# np.savez(os.path.join(saveDir, "siDataResults.npz"), LRSiPredsBinary=LRSiPredsBinary, criticSiPredsScore=criticSiPredsScore,
#     siTPPerThresh=siTPPerThresh, siFPPerThresh=siFPPerThresh, siFNPerThresh=siFNPerThresh, siTNPerThresh=siTNPerThresh,
#     siPrecisionPerThresh=siPrecisionPerThresh, siSensitivityPerThresh=siSensitivityPerThresh, siSpecificityPerThresh=siSpecificityPerThresh, siAccuracyPerThresh=siAccuracyPerThresh)

# PLOTS

plt.subplot(121, aspect='equal', adjustable='box-forced')
plt.title("ROC Curve")
# ROC CURVE: TPR vs FPR
# Baseline
plt.plot([0, 1], [0, 1], c='y', label="Baseline")
# Ideal
plt.plot([1, 0, 0], [1, 1, 0], c='g', label="Ideal")
# ROC Curve
plt.plot(1-trainSpecificityPerThresh, trainSensitivityPerThresh, '-.', c='r', label="Critic - Train")
plt.plot(1-trainSpecificityPerThresh[10], trainSensitivityPerThresh[10], '-x', c='r', markeredgewidth=2)
plt.plot(1-valSpecificityPerThresh, valSensitivityPerThresh, '-', c='r', label="Critic - Val")
plt.plot(1-valSpecificityPerThresh[10], valSensitivityPerThresh[10], '-x', c='r', markeredgewidth=2)
plt.plot(1-siSpecificityPerThresh, siSensitivityPerThresh, '--', c='r', label="Critic - SI")
plt.plot(1-siSpecificityPerThresh[10], siSensitivityPerThresh[10], '-x', c='r', markeredgewidth=2)
plt.ylabel("True Postive Rate")
plt.xlabel("False Postive Rate")
plt.legend(loc='lower right', fontsize=10, fancybox=True)
leg.get_frame().set_alpha(0.3)
plt.xlim([-.1, 1.1])
plt.ylim([-.1, 1.1])
# PRECISION-RECALL CURVE
plt.subplot(122, aspect='equal', adjustable='box-forced')
plt.title("Precision-Recall Curve")
basePrecTrain = np.sum(LRTrainPredsBinary)/len(LRTrainPredsBinary)
basePrecVal = np.sum(LRValPredsBinary)/len(LRValPredsBinary)
basePrecSi = np.sum(LRSiPredsBinary)/len(LRSiPredsBinary)
# Baseline
plt.plot([0, 1], [basePrecTrain, basePrecTrain], '-.', c='y', label="Baseline - Train")
plt.plot([0, 1], [basePrecVal, basePrecVal], '-', c='y', label="Baseline - Val")
plt.plot([0, 1], [basePrecSi, basePrecSi], '--', c='y', label="Baseline - SI")
# Ideal
plt.plot([1, 1, 0], [min(basePrecTrain, basePrecVal, basePrecSi), 1, 1], c='g', label="Ideal")
# PR Curve
precTrain = np.array(trainPrecisionPerThresh)
precVal = np.array(valPrecisionPerThresh)
precSi = np.array(siPrecisionPerThresh)
precTrain[-1] = precTrain[-2]
precVal[-1] = precVal[-2]
precSi[-1] = precSi[-2]
plt.plot(trainSensitivityPerThresh, precTrain, '-.', c='r', label="Critic - Train")
plt.plot(trainSensitivityPerThresh[10], precTrain[10], '-x', c='r', markeredgewidth=2)
plt.plot(valSensitivityPerThresh, precVal, '-', c='r', label="Critic - Val")
plt.plot(valSensitivityPerThresh[10], precVal[10], '-x', c='r', markeredgewidth=2)
plt.plot(siSensitivityPerThresh, precSi, '--', c='r', label="Critic - SI")
plt.plot(siSensitivityPerThresh[10], precSi[10], '-x', c='r', markeredgewidth=2)
plt.ylabel("Precision")
plt.xlabel("Recall")
leg = plt.legend(loc='lower right', fontsize=10, fancybox=True)
leg.get_frame().set_alpha(0.3)
plt.xlim([-.1, 1.1])
plt.ylim([-.4, 1.1])
plt.subplots_adjust(top=0.95)
plt.suptitle("C3DCritic-l1f4-l2f4-l3f4-fc1n4-vid16-OHWord51-OHWordFc16-out64-Adam-1e-03-epoch016\nLSTMLipReader-revSeq-words51-Mask-LSTMh256-LSTMtanh-depth2-enc64-relu-Adam-1e-03", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(saveDir, "C3DCritic-l1f4-l2f4-l3f4-fc1n4-vid16-OHWord51-OHWordFc16-out64-Adam-1e-03-GRIDcorpus-s0107-09-ROC-PR-epoch016.png"))
# plt.show()
plt.close()


# ACC
plt.plot(thresholds, trainAccuracyPerThresh, '-.', c='r', label="Critic - Train")
plt.plot(thresholds[10], trainAccuracyPerThresh[10], '-x', c='r', markeredgewidth=2)
plt.plot(thresholds, valAccuracyPerThresh, '-', c='r', label="Critic - Val")
plt.plot(thresholds[10], valAccuracyPerThresh[10], '-x', c='r', markeredgewidth=2)
plt.plot(thresholds, siAccuracyPerThresh, '--', c='r', label="Critic - SI")
plt.plot(thresholds[10], siAccuracyPerThresh[10], '-x', c='r', markeredgewidth=2)

plt.plot(manyThresh, LRTrainCorrectCountsPerThresh, '-.', c='g', label="LR - Train")
plt.plot(manyThresh[500], LRTrainCorrectCountsPerThresh[500], '-x', c='g', markeredgewidth=2)
plt.plot(manyThresh, LRValCorrectCountsPerThresh, '-', c='g', label="LR - Val")
plt.plot(manyThresh[500], LRValCorrectCountsPerThresh[500], '-x', c='g', markeredgewidth=2)
plt.plot(manyThresh, LRSiCorrectCountsPerThresh, '--', c='g', label="LR - SI")
plt.plot(manyThresh[500], LRSiCorrectCountsPerThresh[500], '-x', c='g', markeredgewidth=2)

plt.ylabel("Accuracy")
plt.xlabel("Threshold")
leg = plt.legend(loc='best', fontsize=10, fancybox=True)
leg.get_frame().set_alpha(0.3)
plt.xlim([-.1, 1.1])
plt.ylim([-.1, 1.1])
plt.subplots_adjust(top=0.85)
plt.title("C3DCritic-l1f4-l2f4-l3f4-fc1n4-vid16-OHWord51-OHWordFc16-out64-Adam-1e-03-epoch016\nLSTMLipReader-revSeq-words51-Mask-LSTMh256-LSTMtanh-depth2-enc64-relu-Adam-1e-03", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(saveDir, "C3DCritic-l1f4-l2f4-l3f4-fc1n4-vid16-OHWord51-OHWordFc16-out64-Adam-1e-03-GRIDcorpus-s0107-09-ROC-PR-acc-epoch016.png"))
# plt.show()
plt.close()


######################################################################
# LR * CRITIC
######################################################################

# Generating functions
genTrainImages = gen_mouth_images(trainDirs, batchSize=batchSize, reverseImageSequence=reverseImageSequence, wordsVocabSize=wordsVocabSize, shuffle=True, shuffleWords=True)
genValImages = gen_mouth_images(valDirs, batchSize=batchSize, reverseImageSequence=reverseImageSequence, wordsVocabSize=wordsVocabSize, shuffle=False, shuffleWords=False)
genSiImages = gen_mouth_images(siDirs, batchSize=batchSize, reverseImageSequence=reverseImageSequence, wordsVocabSize=wordsVocabSize, shuffle=False, shuffleWords=False)

batchSize = 128     # num of speaker vids
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)
siSteps = int(len(siDirs) / batchSize)

topN = 5

# TRAIN

LRTrainPredsBinary = np.empty((0), int)     # Whether the LR Prediction is correct or not
LRTrainCorrectWordScores = np.empty((0))
criticTrainPredsBinary = np.empty((0), int)
combinedTrainPredsBinary = np.empty((0), int)
for step in tqdm.tqdm((range(trainSteps))):
    # Ground truth
    vids, ohWords = next(genTrainImages)
    words = np.argmax(ohWords, axis=1)
    # Lip Reader predictions
    LRPredWordScores = LSTMLipReaderModel.predict(vids)
    LRTrainCorrectWordScores = np.append(LRTrainCorrectWordScores, [wordScores[words[i]] for i, wordScores in enumerate(LRPredWordScores)])
    LRPredWords = np.argmax(LRPredWordScores, axis=1)
    LRTrainPredsBinary = np.append(LRTrainPredsBinary, np.array(words == LRPredWords).astype(int))
    # LIPREADER topN predictions
    LRTopNPredWords = np.argsort(LRPredWordScores, axis=1)[:, -topN:]
    LRTopNPredWordScores = np.array([predWordScores[LRTopNPredWords[i]] for i, predWordScores in enumerate(LRPredWordScores)])
    # Critic predictions
    criticVids = np.tile(vids, (1, topN, 1)).reshape((vids.shape[0]*topN, vids.shape[1], -1))
    LRTopNPredWordsFlat = np.reshape(LRTopNPredWords, (-1, 1))
    criticTopNPredWordScores = criticModel.predict([criticVids, np_utils.to_categorical(LRTopNPredWordsFlat, wordsVocabSize)])
    criticTopNPredWordScores = np.reshape(criticTopNPredWordScores, LRTopNPredWordScores.shape)
    criticPredWordIdx = np.argmax(criticTopNPredWordScores, axis=1)
    criticPredWords = LRTopNPredWords[np.arange(len(LRTopNPredWords)), criticPredWordIdx]
    criticTrainPredsBinary = np.append(criticTrainPredsBinary, np.array(words == criticPredWords)).astype(int)
    # LR + Critic
    combinedPredWordScores = np.multiply(LRTopNPredWordScores, criticTopNPredWordScores)
    combinedPredWordIdx = np.argmax(combinedPredWordScores, axis=1)
    combinedPredWords = LRTopNPredWords[np.arange(len(LRTopNPredWords)), combinedPredWordIdx]
    combinedTrainPredsBinary = np.append(combinedTrainPredsBinary, np.array(words == combinedPredWords)).astype(int)

np.sum(LRTrainPredsBinary)/len(LRTrainPredsBinary)
np.sum(criticTrainPredsBinary)/len(criticTrainPredsBinary)
np.sum(combinedTrainPredsBinary)/len(combinedTrainPredsBinary)

# Good LR
# >>> np.sum(LRTrainPredsBinary)/len(LRTrainPredsBinary)
# 0.91510416666666672
# >>> np.sum(combinedTrainPredsBinary)/len(combinedTrainPredsBinary)
# 0.9149857954545455

# Average LR
# >>> np.sum(LRTrainPredsBinary)/len(LRTrainPredsBinary)
# 0.82464488636363631
# >>> np.sum(criticTrainPredsBinary)/len(criticTrainPredsBinary)
# 0.46666666666666667
# >>> np.sum(combinedTrainPredsBinary)/len(combinedTrainPredsBinary)
# 0.81294981060606064

# Bad LR
# np.sum(LRTrainPredsBinary)/len(LRTrainPredsBinary)
# 0.13373579545454545
# >>> np.sum(criticTrainPredsBinary)/len(criticTrainPredsBinary)
# 0.22762784090909091
# >>> np.sum(combinedTrainPredsBinary)/len(combinedTrainPredsBinary)
# 0.24651988636363636


# VAL

LRValPredsBinary = np.empty((0), int)     # Whether the LR Prediction is correct or not
criticValPredsBinary = np.empty((0), int)
combinedValPredsBinary = np.empty((0), int)
for step in tqdm.tqdm((range(valSteps))):
    # Ground truth
    vids, ohWords = next(genValImages)
    words = np.argmax(ohWords, axis=1)
    # Lip Reader predictions
    LRPredWordScores = LSTMLipReaderModel.predict(vids)
    LRPredWords = np.argmax(LRPredWordScores, axis=1)
    LRValPredsBinary = np.append(LRValPredsBinary, np.array(words == LRPredWords).astype(int))
    # LIPREADER topN predictions
    LRTopNPredWords = np.argsort(LRPredWordScores, axis=1)[:, -topN:]
    LRTopNPredWordScores = np.array([predWordScores[LRTopNPredWords[i]] for i, predWordScores in enumerate(LRPredWordScores)])
    # Critic predictions
    criticVids = np.tile(vids, (1, topN, 1)).reshape((vids.shape[0]*topN, vids.shape[1], -1))
    LRTopNPredWordsFlat = np.reshape(LRTopNPredWords, (-1, 1))
    criticTopNPredWordScores = criticModel.predict([criticVids, np_utils.to_categorical(LRTopNPredWordsFlat, wordsVocabSize)])
    criticTopNPredWordScores = np.reshape(criticTopNPredWordScores, LRTopNPredWordScores.shape)
    criticPredWordIdx = np.argmax(criticTopNPredWordScores, axis=1)
    criticPredWords = LRTopNPredWords[np.arange(len(LRTopNPredWords)), criticPredWordIdx]
    criticValPredsBinary = np.append(criticValPredsBinary, np.array(words == criticPredWords)).astype(int)
    # LR * Critic
    combinedPredWordScores = np.multiply(LRTopNPredWordScores, criticTopNPredWordScores)
    combinedPredWordIdx = np.argmax(combinedPredWordScores, axis=1)
    combinedPredWords = LRTopNPredWords[np.arange(len(LRTopNPredWords)), combinedPredWordIdx]
    combinedValPredsBinary = np.append(combinedValPredsBinary, np.array(words == combinedPredWords)).astype(int)

np.sum(LRValPredsBinary)/len(LRValPredsBinary)
np.sum(criticValPredsBinary)/len(criticValPredsBinary)
np.sum(combinedValPredsBinary)/len(combinedValPredsBinary)

# Good LR
# >>> np.sum(LRValPredsBinary)/len(LRValPredsBinary)
# 0.91818576388888884
# >>> np.sum(criticValPredsBinary)/len(criticValPredsBinary)
# 0.50651041666666663
# >>> np.sum(combinedValPredsBinary)/len(combinedValPredsBinary)
# 0.91427951388888884

# Average LR
# >>> np.sum(LRValPredsBinary)/len(LRValPredsBinary)
# 0.82921006944444442
# >>> np.sum(criticValPredsBinary)/len(criticValPredsBinary)
# 0.4650607638888889
# >>> np.sum(combinedValPredsBinary)/len(combinedValPredsBinary)
# 0.81401909722222221

# Bad LR
# >>> np.sum(LRValPredsBinary)/len(LRValPredsBinary)
# 0.1369357638888889
# >>> np.sum(criticValPredsBinary)/len(criticValPredsBinary)
# 0.2224392361111111
# >>> np.sum(combinedValPredsBinary)/len(combinedValPredsBinary)
# 0.24305555555555555


# SI

LRSiPredsBinary = np.empty((0), int)     # Whether the LR Prediction is correct or not
LRSiCorrectWordScores = np.empty((0))
criticSiPredsBinary = np.empty((0), int)
combinedSiPredsBinary = np.empty((0), int)
for step in tqdm.tqdm((range(siSteps))):
    # Ground truth
    vids, ohWords = next(genSiImages)
    words = np.argmax(ohWords, axis=1)
    # Lip Reader predictions
    LRPredWordScores = LSTMLipReaderModel.predict(vids)
    LRSiCorrectWordScores = np.append(LRSiCorrectWordScores, [wordScores[words[i]] for i, wordScores in enumerate(LRPredWordScores)])
    LRPredWords = np.argmax(LRPredWordScores, axis=1)
    LRSiPredsBinary = np.append(LRSiPredsBinary, np.array(words == LRPredWords).astype(int))
    # LIPREADER topN predictions
    LRTopNPredWords = np.argsort(LRPredWordScores, axis=1)[:, -topN:]
    LRTopNPredWordScores = np.array([predWordScores[LRTopNPredWords[i]] for i, predWordScores in enumerate(LRPredWordScores)])
    # Critic predictions
    criticVids = np.tile(vids, (1, topN, 1)).reshape((vids.shape[0]*topN, vids.shape[1], -1))
    LRTopNPredWordsFlat = np.reshape(LRTopNPredWords, (-1, 1))
    criticTopNPredWordScores = criticModel.predict([criticVids, np_utils.to_categorical(LRTopNPredWordsFlat, wordsVocabSize)])
    criticTopNPredWordScores = np.reshape(criticTopNPredWordScores, LRTopNPredWordScores.shape)
    criticPredWordIdx = np.argmax(criticTopNPredWordScores, axis=1)
    criticPredWords = LRTopNPredWords[np.arange(len(LRTopNPredWords)), criticPredWordIdx]
    criticSiPredsBinary = np.append(criticSiPredsBinary, np.array(words == criticPredWords)).astype(int)
    # LR + Critic
    combinedPredWordScores = np.multiply(LRTopNPredWordScores, criticTopNPredWordScores)
    combinedPredWordIdx = np.argmax(combinedPredWordScores, axis=1)
    combinedPredWords = LRTopNPredWords[np.arange(len(LRTopNPredWords)), combinedPredWordIdx]
    combinedSiPredsBinary = np.append(combinedSiPredsBinary, np.array(words == combinedPredWords)).astype(int)

np.sum(LRSiPredsBinary)/len(LRSiPredsBinary)
np.sum(criticSiPredsBinary)/len(criticSiPredsBinary)
np.sum(combinedSiPredsBinary)/len(combinedSiPredsBinary)

# Good LR
# >>> np.sum(LRSiPredsBinary)/len(LRSiPredsBinary)
# 0.24331597222222223
# >>> np.sum(criticSiPredsBinary)/len(criticSiPredsBinary)
# 0.20546875000000001
# >>> np.sum(combinedSiPredsBinary)/len(combinedSiPredsBinary)
# 0.25494791666666666

# Average LR
# >>> np.sum(LRSiPredsBinary)/len(LRSiPredsBinary)
# 0.2259548611111111
# >>> np.sum(criticSiPredsBinary)/len(criticSiPredsBinary)
# 0.1974826388888889
# >>> np.sum(combinedSiPredsBinary)/len(combinedSiPredsBinary)
# 0.23914930555555555

# Bad LR
# >>> np.sum(LRSiPredsBinary)/len(LRSiPredsBinary)
# 0.096961805555555558
# >>> np.sum(criticSiPredsBinary)/len(criticSiPredsBinary)
# 0.13333333333333333
# >>> np.sum(combinedSiPredsBinary)/len(combinedSiPredsBinary)
# 0.1517361111111111


# LR

# Train

for step in tqdm.tqdm((range(trainSteps))):
    # Ground truth
    vids, ohWords = next(genTrainImages)
    words = np.argmax(ohWords, axis=1)
    # Lip Reader predictions
    LRPredWordScores = LSTMLipReaderModel.predict(vids)
    correctTrainWordScores = np.array([wordScores[words[i]] for i, wordScores in enumerate(LRPredWordScores)])

LRCorrectTrainCountsPerThresh = []
manyThresh = np.arange(0, 1.001, 0.001)
for t in manyThresh:
    LRCorrectTrainCountsPerThresh.append(np.sum(LRCorrectTrainWordScores > t)/len(correctTrainWordScores))

# Val

for step in tqdm.tqdm((range(valSteps))):
    # Ground truth
    vids, ohWords = next(genValImages)
    words = np.argmax(ohWords, axis=1)
    # Lip Reader predictions
    LRPredWordScores = LSTMLipReaderModel.predict(vids)
    correctValWordScores = np.array([wordScores[words[i]] for i, wordScores in enumerate(LRPredWordScores)])

LRCorrectValCountsPerThresh = []
manyThresh = np.arange(0, 1.001, 0.001)
for t in manyThresh:
    LRCorrectValCountsPerThresh.append(np.sum(correctValWordScores > t)/len(correctValWordScores))


# Si

for step in tqdm.tqdm((range(siSteps))):
    # Ground truth
    vids, ohWords = next(genSiImages)
    words = np.argmax(ohWords, axis=1)
    # Lip Reader predictions
    LRPredWordScores = LSTMLipReaderModel.predict(vids)
    correctSiWordScores = np.array([wordScores[words[i]] for i, wordScores in enumerate(LRPredWordScores)])

LRCorrectSiCountsPerThresh = []
manyThresh = np.arange(0, 1.001, 0.001)
for t in manyThresh:
    LRCorrectSiCountsPerThresh.append(np.sum(correctSiWordScores > t)/len(correctSiWordScores))


plt.plot(thresholds, trainAccuracyPerThresh, '-.', c='r', label="Critic - Train")
plt.plot(thresholds[10], trainAccuracyPerThresh[10], '-x', c='r', markeredgewidth=2)
plt.plot(thresholds, valAccuracyPerThresh, '-', c='r', label="Critic - Val")
plt.plot(thresholds[10], valAccuracyPerThresh[10], '-x', c='r', markeredgewidth=2)
plt.plot(thresholds, siAccuracyPerThresh, '--', c='r', label="Critic - SI")
plt.plot(thresholds[10], siAccuracyPerThresh[10], '-x', c='r', markeredgewidth=2)
plt.plot(manyThresh, LRCorrectTrainCountsPerThresh, '-.', c='g', label="LR - Train")
plt.plot(manyThresh[500], LRCorrectTrainCountsPerThresh[500], '-x', c='g', markeredgewidth=2)
plt.plot(manyThresh, LRCorrectValCountsPerThresh, '-', c='g', label="LR - Val")
plt.plot(manyThresh[500], LRCorrectValCountsPerThresh[500], '-x', c='g', markeredgewidth=2)
plt.plot(manyThresh, LRCorrectSiCountsPerThresh, '--', c='g', label="LR - SI")
plt.plot(manyThresh[500], LRCorrectSiCountsPerThresh[500], '-x', c='g', markeredgewidth=2)
plt.legend()
plt.show()



# SI - ALL WORDS

LRSiPredsBinary = np.empty((0), int)     # Whether the LR Prediction is correct or not
criticSiAllPredsBinary = np.empty((0), int)
combinedSiPredsBinary = np.empty((0), int)
for step in tqdm.tqdm((range(siSteps))):
    # Ground truth
    vids, ohWords = next(genSiImages)
    words = np.argmax(ohWords, axis=1)
    # Lip Reader predictions
    # LRPredWordScores = LSTMLipReaderModel.predict(vids)
    # LRPredWords = np.argmax(LRPredWordScores, axis=1)
    # LRSiPredsBinary = np.append(LRSiPredsBinary, np.array(words == LRPredWords).astype(int))
    # LIPREADER topN predictions
    LRTopNPredWords = np.argsort(LRPredWordScores, axis=1)[:, -topN:]
    LRTopNPredWordScores = np.array([predWordScores[LRTopNPredWords[i]] for i, predWordScores in enumerate(LRPredWordScores)])
    # Critic predictions
    criticVids = np.tile(vids, (1, wordsVocabSize, 1)).reshape((vids.shape[0]*wordsVocabSize, vids.shape[1], -1))
    allWordsFlat = np.array(np.tile(np.arange(wordsVocabSize), (len(vids), 1))).reshape((vids.shape[0]*wordsVocabSize))
    criticAllPredWordScores = criticModel.predict([criticVids, np_utils.to_categorical(allWordsFlat, wordsVocabSize)])
    criticAllPredWordScores = np.reshape(criticAllPredWordScores, (len(vids), -1))
    criticPredWords = np.argmax(criticAllPredWordScores, axis=1)
    criticSiAllPredsBinary = np.append(criticSiAllPredsBinary, np.array(words == criticPredWords)).astype(int)
    # LR + Critic
    # combinedPredWordScores = np.multiply(LRTopNPredWordScores, criticTopNPredWordScores)
    # combinedPredWordIdx = np.argmax(combinedPredWordScores, axis=1)
    # combinedPredWords = LRTopNPredWords[np.arange(len(LRTopNPredWords)), combinedPredWordIdx]
    # combinedSiPredsBinary = np.append(combinedSiPredsBinary, np.array(words == combinedPredWords)).astype(int)

np.sum(LRSiPredsBinary)/len(LRSiPredsBinary)
np.sum(criticSiPredsBinary)/len(criticSiPredsBinary)
np.sum(combinedSiPredsBinary)/len(combinedSiPredsBinary)









# Evaluation with Thresholds
trainTotalPopulation = len(criticTrainPredsScore)
thresholds = np.arange(0, 1.001, 0.05)
trainTPPerThresh = np.zeros((len(thresholds)))
trainFPPerThresh = np.zeros((len(thresholds)))
trainFNPerThresh = np.zeros((len(thresholds)))
trainTNPerThresh = np.zeros((len(thresholds)))
for t, thresh in enumerate(thresholds):
    criticTrainPredsBinary = criticTrainPredsScore > thresh
    trainTPPerThresh[t] = np.sum(np.multiply((LRTrainPredsBinary == 1), (criticTrainPredsBinary == 1), dtype=int))
    trainFPPerThresh[t] = np.sum(np.multiply((LRTrainPredsBinary == 0), (criticTrainPredsBinary == 1), dtype=int))
    trainFNPerThresh[t] = np.sum(np.multiply((LRTrainPredsBinary == 1), (criticTrainPredsBinary == 0), dtype=int))
    trainTNPerThresh[t] = np.sum(np.multiply((LRTrainPredsBinary == 0), (criticTrainPredsBinary == 0), dtype=int))

# Precision = TP/(TP + FP)
# Not applicable for threshold = 1, because all are negative
trainPrecisionPerThresh = np.divide(trainTPPerThresh, trainTPPerThresh + trainFPPerThresh)

# Recall or Sensitivity = TP/(TP + FN)
trainSensitivityPerThresh = np.divide(trainTPPerThresh, trainTPPerThresh + trainFNPerThresh)

# Specificity = TN/(TN + FP)
trainSpecificityPerThresh = np.divide(trainTNPerThresh, trainTNPerThresh + trainFPPerThresh)

# Accuracy = (TP + TN)/Total
trainAccuracyPerThresh = (trainTPPerThresh + trainTNPerThresh) / trainTotalPopulation

print(trainAccuracyPerThresh[10])







######################################################################
# WORD-WISE
######################################################################


# RESULTS
lrResults = np.concatenate(np.concatenate((LRTrainPreds, LRSiPreds)), LRSiPreds)
criticResultsPerThresh = np.vstack((criticTrainPredsPerThresh, criticSiPredsPerThresh, criticSiPredsPerThresh))
criticResults = criticResultsPerThresh[:, 10]

# Ground Truth
gtTrain = LRTrainPreds
gtVal = LRValPreds
gtSi = LRSiPreds

# Without taking critic into account,
trainAccWOCritic = np.sum(LRTrainPreds)/len(LRTrainPreds)
valAccWOCritic = np.sum(LRValPreds)/len(LRValPreds)
siAccWOCritic = np.sum(LRSiPreds)/len(LRSiPreds)
print("W/O considering critic, trainAccuracy", trainAccWOCritic, ", valAccuracy", valAccWOCritic, ", speakerIndependentAccuracy", siAccWOCritic)

# Rejecting those the critic said are wrong
criticSaidRightTrain = criticTrainPredsPerThresh[:, 10]==1
gtCriticSaidRightTrain = gtTrain[criticSaidRightTrain]
criticSaidRightVal = criticValPredsPerThresh[:, 10]==1
gtCriticSaidRightVal = gtVal[criticSaidRightVal]
criticSaidRightSi = criticSiPredsPerThresh[:, 10]==1
gtCriticSaidRightSi = gtSi[criticSaidRightSi]
trainAccCriticSaidRight = np.sum(gtCriticSaidRightTrain)/len(gtCriticSaidRightTrain)
valAccCriticSaidRight = np.sum(gtCriticSaidRightVal)/len(gtCriticSaidRightVal)
siAccCriticSaidRight = np.sum(gtCriticSaidRightSi)/len(gtCriticSaidRightSi)
print("Only considering those critic said right, trainAccuracy", trainAccCriticSaidRight, ", valAccuracy", valAccCriticSaidRight, ", speakerIndependentAccuracy", siAccCriticSaidRight)

# Those critic said are wrong
criticSaidWrongTrain = criticTrainPredsPerThresh[:, 10]==0
gtCriticSaidWrongTrain = gtTrain[criticSaidWrongTrain]
criticSaidWrongVal = criticValPredsPerThresh[:, 10]==0
gtCriticSaidWrongVal = gtVal[criticSaidWrongVal]
criticSaidWrongSi = criticSiPredsPerThresh[:, 10]==0
gtCriticSaidWrongSi = gtSi[criticSaidWrongSi]
trainAccCriticSaidWrong = 1 - np.sum(gtCriticSaidWrongTrain)/len(gtCriticSaidWrongTrain)
valAccCriticSaidWrong = 1 - np.sum(gtCriticSaidWrongVal)/len(gtCriticSaidWrongVal)
siAccCriticSaidWrong = 1 - np.sum(gtCriticSaidWrongSi)/len(gtCriticSaidWrongSi)
print("Among those critic said wrong, % actually wrong: train", trainAccCriticSaidWrong, ", val", valAccCriticSaidWrong, ", speakerIndependent", siAccCriticSaidWrong)

# COMPARISON
# np.multiply(np.array(LRTrainPreds==1, int), np.array(criticTrainPredsPerThresh[:, 10]==1, int))
trainAcc = np.sum(np.array(criticTrainPredsPerThresh[:, 10] == gtTrain))/len(LRTrainPreds)
valAcc = np.sum(np.array(criticValPredsPerThresh[:, 10] == gtVal))/len(LRValPreds)
siAcc = np.sum(np.array(criticSiPredsPerThresh[:, 10] == gtSi))/len(LRSiPreds)

# WORD-WISE

# Generating functions
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)
genTrainImages = genMouthImages(trainDirs, batchSize=batchSize, align=trainAlign, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)
genValImages = genMouthImages(valDirs, batchSize=batchSize, align=valAlign, wordIdx=wordIdx,
                              wordsVocabSize=wordsVocabSize, useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage)

# Outputs of this model
thresholds = np.arange(0.01, 1.001, 0.05)
LRTrainPreds = {}    # Whether the LR Prediction is correct or not
criticTrainPreds = {}   # What the Critic predicted
LRValPreds = {}
criticValPreds = {}
for thresh in thresholds:
    LRTrainPreds[thresh] = {}
    criticTrainPreds[thresh] = {}
    LRValPreds[thresh] = {}
    criticValPreds[thresh] = {}
    for word in range(wordsVocabSize):
        LRTrainPreds[thresh][word] = np.empty((0, 1), int)
        criticTrainPreds[thresh][word] = np.empty((0, 1), int)
        LRValPreds[thresh][word] = np.empty((0, 1), int)
        criticValPreds[thresh][word] = np.empty((0, 1), int)

for step in tqdm.tqdm((range(trainSteps))):
    # Ground truth
    vids, words = next(genTrainImages)
    words = np.argmax(words[:, 0, :], axis=1)
    # Lip Reader predictions
    predWordFeatures = encoder.predict(vids)
    predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
    correctPreds = np.array(words == predWords).astype(int)
    criticPreds = criticModel.predict([vids, predWordFeatures, np_utils.to_categorical(predWords, wordsVocabSize)])
    for thresh in thresholds:
        currCriticPreds = (criticPreds < thresh).astype(int)
        for i, word in enumerate(words):
            LRTrainPreds[thresh][word] = np.append(LRTrainPreds[thresh][word], correctPreds[i])
            criticTrainPreds[thresh][word] = np.append(criticTrainPreds[thresh][word], currCriticPreds[i])

trainTP = np.zeros((len(thresholds), wordsVocabSize,))
trainFP = np.zeros((len(thresholds),wordsVocabSize,))
trainFN = np.zeros((len(thresholds),wordsVocabSize,))
trainTN = np.zeros((len(thresholds),wordsVocabSize,))
trainPrecisionPerWord = np.zeros((len(thresholds), wordsVocabSize-1))
avgTrainPrecision = np.zeros((len(thresholds)))
totalTrainPrecision = np.zeros((len(thresholds)))
trainRecallPerWord = np.zeros((len(thresholds), wordsVocabSize-1))
avgTrainRecall = np.zeros((len(thresholds)))
totalTrainRecall = np.zeros((len(thresholds)))
for t, thresh in enumerate(thresholds):
    for w, word in enumerate(range(wordsVocabSize)):
        trainTP[t][w] = np.sum(np.multiply((criticTrainPreds[thresh][word] == 1), (LRTrainPreds[thresh][word] == 1), dtype=int))
        trainFP[t][w] = np.sum(np.multiply((criticTrainPreds[thresh][word] == 1), (LRTrainPreds[thresh][word] == 0), dtype=int))
        trainFN[t][w] = np.sum(np.multiply((criticTrainPreds[thresh][word] == 0), (LRTrainPreds[thresh][word] == 1), dtype=int))
        trainTN[t][w] = np.sum(np.multiply((criticTrainPreds[thresh][word] == 0), (LRTrainPreds[thresh][word] == 0), dtype=int))
    # P, R
    trainPrecisionPerWord[t] = np.divide(trainTP[t], trainTP[t] + trainFP[t])[:-1]
    avgTrainPrecision[t] = np.mean(trainPrecisionPerWord[t][np.isfinite(trainPrecisionPerWord[t])])
    totalTrainPrecision[t] = trainTP[t].sum()/(trainTP[t].sum() + trainFP[t].sum())
    trainRecallPerWord[t] = np.divide(trainTP[t], trainTP[t] + trainFN[t])[:-1]
    avgTrainRecall[t] = np.mean(trainRecallPerWord[t][np.isfinite(trainRecallPerWord[t])])
    totalTrainRecall[t] = trainTP[t].sum()/(trainTP[t].sum() + trainFN[t].sum())

trainMAP = np.mean(avgTrainPrecision)

print("trainPrecisionPerWord = " + str(np.around(trainPrecisionPerWord[:-1], 4)))
print("meanTrainPrecision = " + str(meanTrainPrecision))
print("totalTrainPrecision = " + str(totalTrainPrecision))
print("trainRecallPerWord = " + str(np.around(trainRecallPerWord[:-1], 4)))
print("meanTrainRecall = " + str(meanTrainRecall))
print("totalTrainRecall = " + str(totalTrainRecall))

for step in tqdm.tqdm((range(valSteps))):
    # Ground truth
    vids, words = next(genValImages)
    words = np.argmax(words[:, 0, :], axis=1)
    # Lip Reader predictions
    predWordFeatures = encoder.predict(vids)
    predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
    correctPreds = np.array(words == predWords).astype(int)
    criticPreds = np.round(criticModelWithWordFeatures.predict([vids, predWordFeatures]))
    for i, word in enumerate(words):
        LRValPreds[word] = np.append(LRValPreds[word], correctPreds[i])
        criticValPreds[word] = np.append(criticValPreds[word], criticPreds[i])

valTP = np.zeros((wordsVocabSize,))
valFP = np.zeros((wordsVocabSize,))
valFN = np.zeros((wordsVocabSize,))
valTN = np.zeros((wordsVocabSize,))
for word in range(wordsVocabSize):
    valTP[word] = np.sum(np.multiply((criticValPreds[word] == 1), (LRValPreds[word] == 1), dtype=int))
    valFP[word] = np.sum(np.multiply((criticValPreds[word] == 1), (LRValPreds[word] == 0), dtype=int))
    valFN[word] = np.sum(np.multiply((criticValPreds[word] == 0), (LRValPreds[word] == 1), dtype=int))
    valTN[word] = np.sum(np.multiply((criticValPreds[word] == 0), (LRValPreds[word] == 0), dtype=int))

valPrecisionPerWord = np.divide(valTP, valTP + valFP)[:-1]
meanValPrecision = np.mean(np.isfinite(valPrecisionPerWord))
totalValPrecision = valTP.sum()/(valTP.sum() + valFP.sum())
valRecallPerWord = np.divide(valTP, valTP + valFN)[:-1]
meanValRecall = np.mean(np.isfinite(valRecallPerWord))
totalValRecall = valTP.sum()/(valTP.sum() + valFN.sum())

print("valPrecisionPerWord = " + str(np.around(valPrecisionPerWord[:-1], 4)))
print("meanValPrecision = " + str(meanValPrecision))
print("totalValPrecision = " + str(totalValPrecision))
print("valRecallPerWord = " + str(np.around(valRecallPerWord[:-1], 4)))
print("meanValRecall = " + str(meanValRecall))
print("totalValRecall = " + str(totalValRecall))

