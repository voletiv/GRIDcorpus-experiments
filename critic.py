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
from LSTM_critic_function import *


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
trainValSpeakersList = [1, 2, 3, 4, 5, 6, 7, 9]
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


# # Load the trainDirs (instead of generating them)
# trainX, trainY = load_mouth_images(trainDirs, batchSize=batchSize, align=trainAlign,
#                                  wordIdx=wordIdx, wordsVocabSize=wordsVocabSize)
# valX, valY = load_mouth_images(valDirs, batchSize=batchSize, align=valAlign,
#                              wordIdx=wordIdx, wordsVocabSize=wordsVocabSize)


######################################################################
# BEST LIPREADER MODEL
######################################################################

# OLD; wordsVocabSize = 52; reverse=False
wordsVocabSize = 52
hiddenDim = 256
depth = 2
LSTMactiv = 'tanh'
encodedDim = 64
encodedActiv = 'sigmoid'
lr = 1e-3
LSTMLipReaderModel, LSTMEncoder, _ = LSTMLipReader(wordsVocabSize,
    useMask=False, hiddenDim=hiddenDim, depth=depth, LSTMactiv=LSTMactiv,
    encodedDim=encodedDim, encodedActiv=encodedActiv, lr=lr)
LSTMLipReaderModel.load_weights(os.path.join(saveDir, "LSTM-noPadResults-h256-depth2-LSTMactivtanh-enc64-encodedActivsigmoid-Adam-1e-03-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub-epoch072-tl0.4812-ta0.8492-vl0.6383-va0.8095-sil2.9754-sia0.3441.hdf5"))
reverseImageSequence = False

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


######################################################################
# LSTM CRITIC MODEL with one-hot Word input and 1 output Hidden layer
######################################################################

reverseImageSequence = True
wordsVocabSize = 51
criticModel, fileNamePre = LSTM_critic(hiddenDim=100, LSTMactiv='tanh', depth=1,
        useLSTMfc=True, LSTMfcDim=16,
        oneHotWordDim=wordsVocabSize, useOneHotWordFc=True, oneHotWordFcDim=16,
        outputHDim=64, lr=1e-3)


######################################################################
# TRAIN CRITIC MODEL
######################################################################

nEpochs = 100
initEpoch = 0

batchSize = 128     # num of speaker vids
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)
siValSteps = int(len(siDirs) / batchSize)

print("trainSteps =", trainSteps, "; valSteps =", valSteps, "; siValSteps =", siValSteps)

genTrainImages = gen_mouth_images(trainDirs, batchSize=batchSize, reverseImageSequence=reverseImageSequence, wordsVocabSize=wordsVocabSize, shuffle=True, shuffleWords=True)
genValImages = gen_mouth_images(valDirs, batchSize=batchSize, reverseImageSequence=reverseImageSequence, wordsVocabSize=wordsVocabSize, shuffle=False, shuffleWords=False)
genSiImages = gen_mouth_images(siDirs, batchSize=batchSize, reverseImageSequence=reverseImageSequence, wordsVocabSize=wordsVocabSize, shuffle=False, shuffleWords=False)

topN = 3

tl = 0.
ta = 0.
vl = 0.
va = 0.
sil = 0.
sia = 0.
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

# For each epoch
for epoch in tqdm.tqdm(range(nEpochs)):
    # TRAIN
    # For each train batch
    for step in tqdm.tqdm(range(trainSteps)):
        # GT
        vids, ohActualWords = next(genTrainImages)
        actualWords = np.where(ohActualWords == 1)[1]
        # INPUTS1: topN+1 number of vids
        inputs1 = np.empty((0, framesPerWord, nOfMouthPixels))
        for i in range(topN+1):
            inputs1 = np.vstack((inputs1, vids))
        # INPUTS2: topN predicted and 1 actual
        LRpreds = LSTMLipReaderModel.predict(vids)
        predWordsTopN = np.argsort(LRpreds, axis=1)[:, -topN:]
        # Reshaping s.t. [top1...top2...top3...]
        inputWords2 = np.reshape(predWordsTopN.T[::-1], (batchSize * wordsPerVideo * topN,))
        inputWords2 = np.append(inputWords2, actualWords)
        inputs2 = np_utils.to_categorical(inputWords2, wordsVocabSize)
        # OUTPUTS
        actualWordsRepeat = actualWords
        for i in range(topN):
            actualWordsRepeat = np.append(actualWordsRepeat, actualWords)
        # Output
        outputs = np.array(actualWordsRepeat == inputWords2).astype(int)
        # SHUFFLE
        fullIdx = list(range(len(outputs)))
        np.random.shuffle(fullIdx)
        inputs1 = inputs1[fullIdx]
        inputs2 = inputs2[fullIdx]
        outputs = outputs[fullIdx]
        # FIT
        h = criticModel.fit([inputs1, inputs2], outputs, batch_size=batchSize, epochs=1)
        tl = h.history['loss'][0]
        ta = h.history['acc'][0]
        tlS.append(tl)
        taS.append(ta)
    # VAL
    # For each val batch
    for step in tqdm.tqdm(range(valSteps)):
        # GT
        vids, ohActualWords = next(genValImages)
        actualWords = np.where(ohActualWords == 1)[1]
        # INPUTS1: vids
        inputs1 = vids
        # INPUTS2: predicted words
        LRpreds = LSTMLipReaderModel.predict(vids)
        predWords = np.argmax(LRpreds, axis=1)
        inputs2 = np_utils.to_categorical(predWords, wordsVocabSize)
        # OUTPUTS
        outputs = np.array(actualWords == predWords).astype(int)
        vl, va = criticModel.evaluate([inputs1, inputs2], outputs, batch_size=batchSize)
        vlS.append(vl)
        vaS.append(va)
    # SI
    # For each speaker-independent batch
    for step in tqdm.tqdm(range(siValSteps)):
        # GT
        vids, ohActualWords = next(genSiImages)
        actualWords = np.where(ohActualWords == 1)[1]
        # INPUTS1: vids
        inputs1 = vids
        # INPUTS2: predicted words
        LRpreds = LSTMLipReaderModel.predict(vids)
        predWords = np.argmax(LRpreds, axis=1)
        inputs2 = np_utils.to_categorical(predWords, wordsVocabSize)
        # OUTPUTS
        outputs = np.array(actualWords == predWords).astype(int)
        sil, sia = criticModel.evaluate([inputs1, inputs2], outputs, batch_size=batchSize)
        silS.append(sil)
        siaS.append(sia)
    # Append values
    tlE.append(np.mean(tlS))
    tlS = []
    taE.append(np.mean(taS))
    taS = []
    vlE.append(np.mean(vlS))
    vlS = []
    vaE.append(np.mean(vaS))
    vaS = []
    silE.append(np.mean(silS))
    silS = []
    siaE.append(np.mean(siaS))
    siaS = []
    print(
        "epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-sil{5:.4f}-sia{6:.4f}".format(epoch, tlE[-1], taE[-1], vlE[-1], vaE[-1], silE[-1], siaE[-1]))
    criticModel.save_weights(os.path.join(
        saveDir, fileNamePre + "-epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-sil{5:.4f}-sia{6:.4f}.hdf5".format(epoch, tlE[-1], taE[-1], vlE[-1], vaE[-1], silE[-1], siaE[-1])))
    plotColor = 'g'
    plt.subplot(121)
    plt.plot(tlE, label='trainingLoss', color=plotColor, linestyle='-.')
    plt.plot(vlE, label='valLoss', color=plotColor, linestyle='-')
    plt.plot(silE, label='siLoss', color=plotColor, linestyle='--')
    leg = plt.legend(loc='best', fontsize=11, fancybox=True)
    leg.get_frame().set_alpha(0.3)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Loss")
    plt.subplot(122)
    plt.plot(taE, label='trainingAcc', color=plotColor, linestyle='-.')
    plt.plot(vaE, label='valAcc', color=plotColor, linestyle='-')
    plt.plot(siaE, label='siAcc', color=plotColor, linestyle='--')
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
    plt.suptitle(fileNamePre)
    plt.savefig(os.path.join(saveDir, fileNamePre + "-Plots.png"))
    plt.close()

