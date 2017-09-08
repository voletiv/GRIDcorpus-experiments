import glob
import matplotlib.pyplot as plt
import numpy as np
import os

# import tensorflow as tf
# with tf.device('/cpu:0'):

from keras.callbacks import Callback, EarlyStopping

#################################################################
# IMPORT
#################################################################

from params import *
from gen_these_word_images import *
from LSTM_lipreader_function import *
from load_image_dirs_and_word_numbers import *

#################################################################
# LSTM LipReader MODEL
#################################################################

useMask = True
hiddenDim = 256
depth = 2
LSTMactiv = 'tanh'
encodedDim = 64
encodedActiv = 'relu'
optimizer = 'rmsprop'
lr = 1e-3

# Make model
LSTMLipReaderModel, LSTMEncoder, fileNamePre = LSTM_lipreader(useMask=useMask,
    hiddenDim=hiddenDim, depth=depth, LSTMactiv=LSTMactiv, encodedDim=encodedDim,
    encodedActiv=encodedActiv, optimizer=optimizer, lr=lr)

#############################################################
# LOAD IMAGES
#############################################################

trainDirs, trainWordNumbers, valDirs, valWordNumbers, siDirs, siWordNumbers = load_image_dirs_and_word_numbers(
                                                        trainValSpeakersList=[1, 2, 3, 4, 5, 6, 7, 10],
                                                        siList = [13, 14])


#############################################################
# CALLBACK TO SAVE MODE CHECKPOINT AND PLOT
#############################################################

plotColor = 'g'

class CheckSIAndMakePlots(Callback):
    # On train start
    def on_train_begin(self, logs={}):
        self.plotColor = plotColor
        self.trainLosses = []
        self.valLosses = []
        self.siLosses = []
        self.trainAccuracies = []
        self.valAccuracies = []
        self.siAccuracies = []
        # Define epochIndex
        def epochIndex(x):
            x = x.split('/')[-1].split('-')
            return [i for i, word in enumerate(x) if 'epoch' in word][0]
        # Define epochNoInFile
        def epochNoInFile(x):
            epochIdx = epochIndex(x)
            return x.split('/')[-1].split('-')[epochIdx]
        # For all weight files
        for file in sorted(glob.glob(os.path.join(saveDir, "*" + fileNamePre + "*-epoch*")), key=epochNoInFile):
            print(file)
            epochIdx = epochIndex(file)
            self.trainLosses.append(
                float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[epochIdx + 1][2:]))
            self.trainAccuracies.append(
                float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[epochIdx + 2][2:]))
            self.valLosses.append(
                float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[epochIdx + 3][2:]))
            self.valAccuracies.append(
                float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[epochIdx + 4][2:]))
            self.siLosses.append(
                float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[epochIdx + 5][3:]))
            self.siAccuracies.append(
                float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[epochIdx + 6][3:]))
    # At every epoch
    def on_epoch_end(self, epoch, logs={}):
        # Get current training and validation loss and acc
        tl = logs.get('loss')
        ta = logs.get('acc')
        vl = logs.get('val_loss')
        va = logs.get('val_acc')
        # Speaker-Independent
        print("Calculating speaker-independent loss and acc...")
        sil, sia = calc_sil_and_sia()
        # Save model
        save_model_checkpoint(epoch, tl, ta, vl, va, sil, sia)
        # Append losses and accs
        self.trainLosses.append(tl)
        self.valLosses.append(vl)
        self.siLosses.append(sil)
        self.trainAccuracies.append(ta)
        self.valAccuracies.append(va)
        self.siAccuracies.append(sia)
        # Plot graphs
        plot_and_save_losses_and_accuracies(epoch)
    # Calculate speaker-independent loss and accuracy
    def calc_sil_and_sia(self):
        [sil, sia] = LSTMLipReaderModel.evaluate_generator(genSiImages, siSteps)
        return sil, sia
    # Save model checkpoint
    def save_model_checkpoint(self, epoch, tl, ta, vl, va, sil, sia):
        modelFilePath = os.path.join(saveDir,
            fileNamePre + "-epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-sil{5:.4f}-sia{6:.4f}.hdf5".format(epoch, tl, ta, vl, va, sil, sia))
        print("Saving model", modelFilePath)
        LSTMLipReaderModel.save_weights(modelFilePath)
    # Plot and save losses and accuracies
    def plot_and_save_losses_and_accuracies(self, epoch):
        print("Saving plots for epoch " + str(epoch))
        plt.subplot(121)
        plt.plot(self.trainLosses, label='trainLoss', color=self.plotColor, linestyle='--')
        plt.plot(self.valLosses, label='valLoss', color=self.plotColor, linestyle='-')
        plt.plot(self.siLosses, label='siLoss', color=self.plotColor, linestyle='-.')
        leg = plt.legend(loc='best', fontsize=11, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title("Loss")
        plt.subplot(122)
        plt.plot(self.trainAccuracies, label='trainAcc', color=self.plotColor, linestyle='--')
        plt.plot(self.valAccuracies, label='valAcc', color=self.plotColor, linestyle='-')
        plt.plot(self.siAccuracies, label='siAcc', color=self.plotColor, linestyle='-.')
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
        plt.suptitle(fileNamePre[:int(len(fileNamePre)/2)] + '\n' + fileNamePre[int(len(fileNamePre)/2):], fontsize=10)
        plt.savefig(os.path.join(saveDir, fileNamePre + "-Plots.png"))
        plt.close()


#################################################################
# FIT ON LABELED DATA
#################################################################


def fit_on_labelled_data(iterNumber, LSTMLipReaderModel, trainDirsLabelled,
        trainWordNumbersLabelled, trainWordsLabelled, batchSize, nEpochs, fileNamePre):
    # Make generator: Labelled generator has labels as input
    genTrainImagesLabelled = gen_these_word_images(trainDirsLabelled, trainWordNumbersLabelled,
                                                    allWords=trainWordsLabelled, batchSize=batchSize, shuffle=True)
    trainLabelledSteps = len(trainDirsLabelled) // batchSize
    # Callbacks
    checkSIAndMakePlots = CheckSIAndMakePlots()
    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=True)
    # FIT (gen)
    LSTMLipReaderModelHistory = LSTMLipReaderModel.fit_generator(genTrainImagesLabelled, steps_per_epoch=trainLabelledSteps, epochs=nEpochs, verbose=True,
                                                            callbacks=[checkSIAndMakePlots, earlyStop], validation_data=genValImages, validation_steps=valSteps)
    return LSTMLipReaderModel, checkSIAndMakePlots


#############################################################
# ADD UNLABELLED DATA TO LABELLED DATA BASED ON PRED MAX VAL
#############################################################


def add_unlabelled_data_to_labelled_data(iterNumber, trainDirsLabelled, trainWordNumbersLabelled, trainWordsLabelled,
                        trainDirsUnlabelled, trainWordNumbersUnlabelled,
                        LSTMLipReaderModel, batchSize, fileNamePre,
                        unlabelledPredMaxValueThresh = 0.99):
    print("Adding unlabelled data to labelled data...")
    genTrainImagesUnlabelled = gen_these_word_images(trainDirsUnlabelled, trainWordNumbersUnlabelled, batchSize=batchSize, shuffle=False)
    trainUnlabelledSteps = len(trainDirsUnlabelled) // batchSize
    # Find confidence values of predictions
    unlabelledActualWords = []
    unlabelledPreds = []
    unlabelledPredWords = []
    for step in tqdm.tqdm(range(trainUnlabelledSteps)):
        vids, words = next(genTrainImagesUnlabelled)
        actualWords = np.argmax(words, axis=1)
        preds = LSTMLipReaderModel.predict(vids)
        predWords = np.argmax(preds, axis=1)
        for i in range(len(preds)):
            unlabelledActualWords.append(actualWords[i])
            unlabelledPreds.append(preds[i])
            unlabelledPredWords.append(predWords[i])
    # Max confidence value
    unlabelledPredMaxValues = np.max(np.array(unlabelledPreds), axis=1)
    # Sort all according to unlabelledPredMaxValues
    sortedUnlabelledPredMaxValues, sortedUnlabelledPredWords, sortedUnlabelledActualWords, sortedTrainDirsUnlabelled, sortedTrainWordNumbersUnlabelled = (
        np.array(t) for t in zip(*sorted(zip(unlabelledPredMaxValues, unlabelledPredWords, unlabelledActualWords, trainDirsUnlabelled, trainWordNumbersUnlabelled), reverse=True)))
    # Plot Accuracy
    sortedUnlabelledAccuracyOnMaxValues = np.cumsum(np.equal(sortedUnlabelledPredWords, sortedUnlabelledActualWords)) / (1 + np.arange(len(sortedUnlabelledActualWords)))
    plt.plot(sortedUnlabelledAccuracyOnMaxValues[:int(labelledPercent/100*len(unlabelledAccuracyOnMaxValues))], label='accuracy')
    plt.plot(sortedUnlabelledPredMaxValues[:int(labelledPercent/100*len(unlabelledAccuracyOnMaxValues))], label='max value')
    # plt.scatter(np.arange(int(labelledPercent/100*len(unlabelledAccuracyOnMaxValues))), sortedTrainWordNumbersUnlabelled[:int(labelledPercent/100*len(unlabelledAccuracyOnMaxValues))]/wordsPerVideo, label='wordNumber')
    plt.legend(loc='best')
    plt.xlabel("Number of instances considered, sorted by predicted max value")
    plt.ylabel("Accuracy")
    plt.title("Unlabelled Accuracy with Max Values trained with 10%-" + iterNumber, fontsize=12)
    plt.yticks(np.arange(0.95, 1.01, 0.01))
    plt.gca().yaxis.grid(True)
    # plt.show()
    plt.savefig(os.path.join(saveDir, fileNamePre + "-unlabelled-accuracy-max-values.png"))
    plt.close()
    # Choose those in the unlabelled set that exceed max value thresh
    maxValueFilter = sortedUnlabelledPredMaxValues > unlabelledPredMaxValueThresh
    newTrainDirsLabelled = sortedTrainDirsUnlabelled[maxValueFilter]
    newTrainWordNumbersLabelled = sortedTrainWordNumbersUnlabelled[maxValueFilter]
    newTrainWords = sortedUnlabelledPredWords[maxValueFilter]
    # Add them to the labelled directories
    for directory, wordNum, predWord in zip(newTrainDirsLabelled, newTrainWordNumbersLabelled, newTrainWords):
        trainDirsLabelled = np.append(trainDirsLabelled, directory)
        trainWordNumbersLabelled = np.append(trainWordNumbersLabelled, wordNum)
        trainWordsLabelled = np.append(trainWordsLabelled, predWord)
    # Remove them from unlabelled directories
    trainDirsUnlabelled = sortedTrainDirsUnlabelled[len(newTrainDirsLabelled):]
    trainWordNumbersUnlabelled = sortedTrainWordNumbersUnlabelled[len(newTrainWordNumbersLabelled):]
    return trainDirsLabelled, trainWordNumbersLabelled, trainWordsLabelled, trainDirsUnlabelled, trainWordNumbersUnlabelled

#################################################################
# SAVE LOSSES AND ACC
#################################################################


def save_losses_and_accuracies(checkSIAndMakePlots,
        allApparentLabelledTrainLossesThruSelfLearning, allValLossesThruSelfLearning, allSiLossesThruSelfLearning,
        allApparentLabelledTrainAccuraciesThruSelfLearning, allValAccuraciesThruSelfLearning, allSiAccuraciesThruSelfLearning,
        trueLabelledTrainLossesThruPcOfLabelledData, apparentLabelledTrainLossesThruPcOfLabelledData,
        percentageOfLabelledData, trainDirsLabelled, trainWordNumbersLabelled, batchSize,
        valLossesThruPcOfLabelledData, siLossesThruPcOfLabelledData,
        trueLabelledTrainAccuraciesThruPcOfLabelledData, apparentLabelledTrainAccuraciesThruPcOfLabelledData,
        valAccuraciesThruPcOfLabelledData, siAccuraciesThruPcOfLabelledData):
    # Append all losses and acc thru self-learning
    # i.e. through all epochs of all iterations
    allApparentLabelledTrainLossesThruSelfLearning.append(checkSIAndMakePlots.trainLosses)
    allValLossesThruSelfLearning.append(checkSIAndMakePlots.valLosses)
    allSiLossesThruSelfLearning.append(checkSIAndMakePlots.siLosses)
    allApparentLabelledTrainAccuraciesThruSelfLearning.append(checkSIAndMakePlots.trainAccuracies)
    allValAccuraciesThruSelfLearning.append(checkSIAndMakePlots.valAccuracies)
    allSiAccuraciesThruSelfLearning.append(checkSIAndMakePlots.siAccuracies)
    # To calc true tl and ta
    genTrainImagesLabelled = gen_these_word_images(trainDirsLabelled, trainWordNumbersLabelled, batchSize=batchSize, shuffle=False)
    trainUnlabelledSteps = len(trainDirsUnlabelled) // batchSize
    [trueTl, trueTa] = LSTMLipReaderModel.evaluate_generator(genSiImages, siSteps)
    # Append final losses and acc thru every iteration of self-learning
    percentageOfLabelledData.append(len(trainDirsLabelled)/len(trainDirs))
    trueLabelledTrainLossesThruPcOfLabelledData.append(trueTl)
    apparentLabelledTrainLossesThruPcOfLabelledData.append(checkSIAndMakePlots.trainLosses[-1])
    valLossesThruPcOfLabelledData.append(checkSIAndMakePlots.valLosses[-1])
    siLossesThruPcOfLabelledData.append(checkSIAndMakePlots.siLosses[-1])
    trueLabelledTrainAccuraciesThruPcOfLabelledData.append(trueTa)
    apparentLabelledTrainAccuraciesThruPcOfLabelledData.append(checkSIAndMakePlots.trainAccuracies[-1])
    valAccuraciesThruPcOfLabelledData.append(checkSIAndMakePlots.valAccuracies[-1])
    siAccuraciesThruPcOfLabelledData.append(checkSIAndMakePlots.siAccuracies[-1])

#################################################################
# PLOT ALL LOSSES AND ACC THRU ALL ITERS AND ALL EPOCHS OF SELF-LEARNING
#################################################################


def plot_all_losses_and_accuracies_thru_self_learning(
        allApparentLabelledTrainLossesThruSelfLearning, allValLossesThruSelfLearning, allSiLossesThruSelfLearning,
        allApparentLabelledTrainAccuraciesThruSelfLearning, allValAccuraciesThruSelfLearning, allSiAccuraciesThruSelfLearning,
        fileNamePre, plotColor='g'
        ):
    # Plot name
    plotName = '-'.join(fileNamePre.split('-')[:-1])
    # Y values
    tl = []
    vl = []
    sil = []
    ta = []
    va = []
    sia = []
    # X axis ticks
    xAxisTicks = []
    verticalLineX = []
    count = -1
    # All values
    for iterNum in range(len(allApparentLabelledTrainLossesThruSelfLearning)):
        for epochNum in range(len(allApparentLabelledTrainLossesThruSelfLearning[iterNum])):
            count += 1
            xAxisTicks.append(epochNum)
            tl.append(allApparentLabelledTrainLossesThruSelfLearning[iterNum][epochNum])
            vl.append(allValLossesThruSelfLearning[iterNum][epochNum])
            sil.append(allSiLossesThruSelfLearning[iterNum][epochNum])
            ta.append(allApparentLabelledTrainAccuraciesThruSelfLearning[iterNum][epochNum])
            va.append(allValAccuraciesThruSelfLearning[iterNum][epochNum])
            sia.append(allSiAccuraciesThruSelfLearning[iterNum][epochNum])
        # Add a vertical line after every iteration
        verticalLineX.append(count)
    # Plot
    plt.subplot(211)
    plt.plot(tl, label='apparent train loss', color=plotColor, linestyle='--')
    plt.plot(vl, label='val loss', color=plotColor, linestyle='-')
    plt.plot(sil, label='SI loss', color=plotColor, linestyle='-.')
    plt.xticks(np.arange(len(xAxisTicks)), xAxisTicks)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    for x in verticalLineX:
        plt.plot([x, x], plt.gca().get_ylim(), color='k')
    plt.xlabel('epochs within iterations of self-learning')
    plt.ylabel('loss')
    leg = plt.legend(loc='best', fontsize=11, fancybox=True)
    leg.get_frame().set_alpha(0.3)
    plt.title("Loss")
    plt.tight_layout()
    plt.subplot(212)
    plt.plot(ta, label='apparent train acc', color=plotColor, linestyle='--')
    plt.plot(va, label='val acc', color=plotColor, linestyle='-')
    plt.plot(sia, label='SI acc', color=plotColor, linestyle='-.')
    plt.xticks(np.arange(len(xAxisTicks)), xAxisTicks)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.gca().yaxis.grid(True)
    for x in verticalLineX:
        plt.plot([x, x], plt.gca().get_ylim(), color='k')
    plt.xlabel('epochs within iterations of self-learning')
    plt.ylabel('accuracy')
    leg = plt.legend(loc='best', fontsize=11, fancybox=True)
    leg.get_frame().set_alpha(0.3)
    plt.title("Accuracy")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle(plotName[:int(len(plotName)/2)] + '\n' + plotName[int(len(plotName)/2):], fontsize=10)
    plt.savefig(os.path.join(saveDir, plotName + "-All-Loss-Acc.png"))
    plt.close()


#################################################################
# PLOT LOSSES AND ACC THRU ITERS (PERCENTAGE OF LABELLED DATA)
#################################################################


def plot_losses_and_accuracies_thru_percentage_of_labelled_data(
        percentageOfLabelledData,
        apparentLabelledTrainLossesThruPcOfLabelledData, trueLabelledTrainLossesThruPcOfLabelledData,
        valLossesThruPcOfLabelledData, siLossesThruPcOfLabelledData,
        apparentLabelledTrainAccuraciesThruPcOfLabelledData, trueLabelledTrainAccuraciesThruPcOfLabelledData,
        valAccuraciesThruPcOfLabelledData, siAccuraciesThruPcOfLabelledData,
        fileNamePre, plotColor='g'
        ):
    plotName = '-'.join(fileNamePre.split('-')[:-1])
    # Plot
    plt.subplot(121)
    plt.xlim([0, 100])
    plt.plot(percentageOfLabelledData, apparentLabelledTrainLossesThruPcOfLabelledData, label='apparent train loss', color=plotColor, linestyle=':', marker='.')
    plt.plot(percentageOfLabelledData, trueLabelledTrainLossesThruPcOfLabelledData, label='true train loss', color=plotColor, linestyle=':', marker='o')
    plt.plot(percentageOfLabelledData, valLossesThruPcOfLabelledData, label='val loss', color=plotColor, linestyle=':', marker='D')
    plt.plot(percentageOfLabelledData, siLossesThruPcOfLabelledData, label='SI loss', color=plotColor, linestyle=':', marker='+')
    plt.xlabel('% of labelled data')
    plt.ylabel('loss')
    leg = plt.legend(loc='best', fontsize=11, fancybox=True)
    leg.get_frame().set_alpha(0.3)
    plt.title("Loss")
    plt.tight_layout()
    plt.subplot(122)
    plt.xlim([0, 100])
    plt.plot(percentageOfLabelledData, apparentLabelledTrainAccuraciesThruPcOfLabelledData, label='apparent train acc', color=plotColor, linestyle=':', marker='.')
    plt.plot(percentageOfLabelledData, trueLabelledTrainAccuraciesThruPcOfLabelledData, label='true train acc', color=plotColor, linestyle=':', marker='o')
    plt.plot(percentageOfLabelledData, valAccuraciesThruPcOfLabelledData, label='val acc', color=plotColor, linestyle=':', marker='D')
    plt.plot(percentageOfLabelledData, siAccuraciesThruPcOfLabelledData, label='SI acc', color=plotColor, linestyle=':', marker='+')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.gca().yaxis.grid(True)
    plt.xlabel('% of labelled data')
    plt.ylabel('accuracy')
    leg = plt.legend(loc='best', fontsize=11, fancybox=True)
    leg.get_frame().set_alpha(0.3)
    plt.title("Accuracy")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle(plotName[:int(len(plotName)/2)] + '\n' + plotName[int(len(plotName)/2):], fontsize=10)
    plt.savefig(os.path.join(saveDir, plotName + "-pcOfLabelledData-Loss-Acc.png"))
    plt.close()



#################################################################
# 10%
#################################################################

labelledPercent = 10

# Make model
useMask = True
hiddenDim = 256
depth = 2
LSTMactiv = 'tanh'
encodedDim = 64
encodedActiv = 'relu'
optimizer = 'adam'
lr = 1e-3
LSTMLipReaderModel, LSTMEncoder, fileNamePre = LSTM_lipreader(useMask=useMask,
    hiddenDim=hiddenDim, depth=depth, LSTMactiv=LSTMactiv, encodedDim=encodedDim,
    encodedActiv=encodedActiv, optimizer=optimizer, lr=lr)

fileNamePre += "-GRIDcorpus-s0107-10-si-s1314"

# Starting training with only 10% of training data,
# and assuming the rest 90% to be unlabelled
fileNamePre += "-10PercentSelfLearning"

# Split into labelled and unlabelled training data
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)
fullIdx = np.arange(len(trainDirs))
np.random.shuffle(fullIdx)
trainDirs = np.array(trainDirs)[fullIdx]
trainWordNumbers = np.array(trainWordNumbers)[fullIdx]
trainDirsLabelled = trainDirs[:int(labelledPercent/100*len(trainDirs))]
trainWordNumbersLabelled = trainWordNumbers[:int(labelledPercent/100*len(trainDirs))]
trainDirsUnlabelled = trainDirs[int(labelledPercent/100*len(trainDirs)):]
trainWordNumbersUnlabelled = trainWordNumbers[int(labelledPercent/100*len(trainDirs)):]

# Decide batchSize, nEpochs
batchSize = 512
nEpochs = 100

# Make Generating Functions
genTrainImages = gen_these_word_images(trainDirs, trainWordNumbers, batchSize=batchSize, shuffle=False)
trainSteps = len(trainDirs) // batchSize
genValImages = gen_these_word_images(valDirs, valWordNumbers, batchSize=batchSize, shuffle=False)
valSteps = len(valDirs) // batchSize
genSiImages = gen_these_word_images(siDirs, siWordNumbers, batchSize=batchSize, shuffle=False)
siSteps = len(siDirs) // batchSize

# Find out correct labels for labelled data
genTrainImagesLabelled = gen_these_word_images(trainDirsLabelled, trainWordNumbersLabelled, batchSize=1, shuffle=False)
trainLabelledSteps = len(trainDirsLabelled)
trainWordsLabelled = np.empty((0))
for i in tqdm.tqdm(range(trainLabelledSteps)):
    _, words = next(genTrainImagesLabelled)
    for word in words:
        trainWordsLabelled = np.append(trainWordsLabelled, np.argmax(word))

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

# To fit
unlabelledPredMaxValueThresh = 0.99
nIters = 100
fileNamePre += '-0'
for iterNumber in range(nIters):
    # Change fileNamePre
    fileNamePre = fileNamePre[:-2] + '-' + str(iterNumber)
    # Fit
    LSTMLipReaderModel, checkSIAndMakePlots = fit_on_labelled_data(iterNumber,
        LSTMLipReaderModel, trainDirsLabelled, trainWordNumbersLabelled, trainWordsLabelled, batchSize, nEpochs, fileNamePre)
    # Save losses and accs
    save_losses_and_accuracies(checkSIAndMakePlots,
        allApparentLabelledTrainLossesThruSelfLearning, allValLossesThruSelfLearning, allSiLossesThruSelfLearning,
        allApparentLabelledTrainAccuraciesThruSelfLearning, allValAccuraciesThruSelfLearning, allSiAccuraciesThruSelfLearning,
        percentageOfLabelledData, trainDirsLabelled, trainWordNumbersLabelled, batchSize,
        trueLabelledTrainLossesThruPcOfLabelledData, apparentLabelledTrainLossesThruPcOfLabelledData,
        valLossesThruPcOfLabelledData, siLossesThruPcOfLabelledData,
        trueLabelledTrainAccuraciesThruPcOfLabelledData, apparentLabelledTrainAccuraciesThruPcOfLabelledData,
        valAccuraciesThruPcOfLabelledData, siAccuraciesThruPcOfLabelledData)
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
    trainDirsLabelled, trainWordNumbersLabelled, trainWordsLabelled, trainDirsUnlabelled, trainWordNumbersUnlabelled \
        = add_unlabelled_data_to_labelled_data(iterNumber, trainDirsLabelled, trainWordNumbersLabelled, trainWordsLabelled,
                trainDirsUnlabelled, trainWordNumbersUnlabelled, LSTMLipReaderModel,
                batchSize, fileNamePre, unlabelledPredMaxValueThresh)



#################################################################
# 20%
#################################################################








#################################################################
# IGNORE
#################################################################

# # Find confidence values of predictions
# unlabelledActualWords = []
# unlabelledPreds = []
# unlabelledPredWords = []
# for step in range(trainUnlabelledSteps):
#     vids, words = next(genTrainImagesUnlabelled)
#     actualWords = np.argmax(words, axis=1)
#     preds = LSTMLipReaderModel.predict(vids)
#     predWords = np.argmax(preds, axis=1)
#     for i in range(len(preds)):
#         unlabelledActualWords.append(actualWords[i])
#         unlabelledPreds.append(preds[i])
#         unlabelledPredWords.append(predWords[i])

# unlabelledPredMaxValues = np.max(np.array(unlabelledPreds), axis=1)

# # unlabelledPreds = LSTMLipReaderModel.predict_generator(genTrainImagesUnlabelled, trainUnlabelledSteps)
# # unlabelledPredMaxValues = np.max(unlabelledPreds, axis=1)
# # unlabelledPredWords = np.argmax(unlabelledPreds, axis=1)
# # unlabelledActualWords = []
# # for i in range(trainUnlabelledSteps):
# #     _, words = next(genTrainImagesUnlabelled)
# #     for w in words:
# #         unlabelledActualWords.append(np.argmax(w))

# # Sort all according to unlabelledPredMaxValues
# sortedUnlabelledPredMaxValues, sortedUnlabelledPredWords, sortedUnlabelledActualWords, sortedTrainDirsUnlabelled, sortedTrainWordNumbersUnlabelled = (
#     np.array(t) for t in zip(*sorted(zip(unlabelledPredMaxValues, unlabelledPredWords, unlabelledActualWords, trainDirsUnlabelled, trainWordNumbersUnlabelled), reverse=True)))

# # Plot Accuracy
# unlabelledAccuracyOnMaxValues = np.cumsum(np.equal(sortedUnlabelledPredWords, sortedUnlabelledActualWords)) / (1 + np.arange(len(sortedUnlabelledActualWords)))
# plt.plot(unlabelledAccuracyOnMaxValues[:int(labelledPercent/100*len(unlabelledAccuracyOnMaxValues))], label='accuracy')
# plt.plot(sortedUnlabelledPredMaxValues[:int(labelledPercent/100*len(unlabelledAccuracyOnMaxValues))], label='max value')
# # plt.scatter(np.arange(int(labelledPercent/100*len(unlabelledAccuracyOnMaxValues))), sortedTrainWordNumbersUnlabelled[:int(labelledPercent/100*len(unlabelledAccuracyOnMaxValues))]/wordsPerVideo, label='wordNumber')
# plt.legend(loc='best')
# plt.xlabel("Number of instances considered, sorted by predicted max value")
# plt.ylabel("Accuracy")
# plt.title("Unlabelled Accuracy with Max Values trained with 10%-1", fontsize=12)
# plt.yticks(np.arange(0.95, 1.005, 0.005))
# plt.gca().yaxis.grid(True)
# plt.show()

# # For max % 2max
# # unlabelledPred2MaxValues = np.reshape(np.sort(unlabelledPreds, axis=1)[:, -2:-1], (len(unlabelledPred2MaxValues),))
# # unlabelledPredMaxBy2MaxValues = np.divide(unlabelledPredMaxValues, unlabelledPred2MaxValues)
# # # Sort all according to unlabelledPredMaxValues
# # sortedUnlabelledPredMaxBy2MaxValues, sortedUnlabelledPredWords, sortedUnlabelledActualWords = (
# #     list(t) for t in zip(*sorted(zip(unlabelledPredMaxBy2MaxValues, unlabelledPredWords, unlabelledActualWords), reverse=True)))
# # unlabelledAccuracyOnMaxBy2MaxValues = np.cumsum(np.equal(sortedUnlabelledPredWords, sortedUnlabelledActualWords)) / (1 + np.arange(len(sortedUnlabelledActualWords)))
# # plt.plot(unlabelledAccuracyOnMaxBy2MaxValues[:int(labelledPercent/100*len(unlabelledAccuracyOnMaxBy2MaxValues))], label='accuracy')
# # plt.plot(sortedUnlabelledPredMaxBy2MaxValues[:int(labelledPercent/100*len(unlabelledAccuracyOnMaxValues))] / sortedUnlabelledPredMaxBy2MaxValues[0] / 8 + .8 , label='max/2max')
# # plt.legend(loc='best')
# # plt.xlabel("Number of instances, sorted by predicted max value divided by 2nd max value")
# # plt.ylabel("Accuracy")
# # plt.title("Unlabelled Accuracy with Max divided by 2nd Max Values trained with 10%-1", fontsize=12)
# # plt.show()

# # Choose those in the unlabelled set that exceed max value thresh
# unlabelledPredMaxValueThresh = 0.99
# maxValueFilter = sortedUnlabelledPredMaxValues > unlabelledPredMaxValueThresh
# newTrainDirsLabelled = sortedTrainDirsUnlabelled[maxValueFilter]
# newTrainWordNumbersLabelled = sortedTrainWordNumbersUnlabelled[maxValueFilter]
# newTrainWords = sortedUnlabelledPredWords[maxValueFilter]
# # Add them to the training directories
# for directory, wordNum, predWord in zip(newTrainDirsLabelled, newTrainWordNumbersLabelled, newTrainWords):
#     trainDirsLabelled.append(directory)
#     trainWordNumbersLabelled.append(wordNum)
#     trainWordsLabelled.append(np_utils.to_categorical(predWord, wordsVocabSize))

# # Remove them from unlabelled directories
# trainDirsUnlabelled = sortedTrainDirsUnlabelled[len(newTrainDirsLabelled):]
# trainWordNumbersUnlabelled = sortedTrainWordNumbersUnlabelled[len(newTrainWordNumbersLabelled):]


