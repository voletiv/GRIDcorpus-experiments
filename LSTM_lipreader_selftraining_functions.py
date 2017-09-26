import glob
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
import numpy as np
import os
import time

# import tensorflow as tf
# with tf.device('/cpu:0'):

from keras.callbacks import Callback
from mpl_toolkits.axes_grid1 import host_subplot

#################################################################
# IMPORT
#################################################################

from params import *
from gen_these_word_images import *
from C3D_critic_function import *
from LSTM_lipreader_function import *
from load_image_dirs_and_word_numbers import *

#############################################################
# CALLBACK TO SAVE MODE CHECKPOINT AND PLOT
#############################################################

class CheckSIAndMakePlots(Callback):
    def __init__(self, genSiImages=None, siSteps=None, plotColor='g', fileNamePre='aaaa'):
        self.genSiImages = genSiImages
        self.siSteps = siSteps
        self.plotColor = plotColor
        self.fileNamePre = fileNamePre
    # On train start
    def on_train_begin(self, logs={}):
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
        for file in sorted(glob.glob(os.path.join(saveDir, "*" + self.fileNamePre + "*-epoch*")), key=epochNoInFile):
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
        sil, sia = self.calc_sil_and_sia()
        # Save model
        self.save_model_checkpoint(epoch, tl, ta, vl, va, sil, sia)
        # Append losses and accs
        self.trainLosses.append(tl)
        self.valLosses.append(vl)
        self.siLosses.append(sil)
        self.trainAccuracies.append(ta)
        self.valAccuracies.append(va)
        self.siAccuracies.append(sia)
        # Plot graphs
        self.plot_and_save_losses_and_accuracies(epoch)
    # Calculate speaker-independent loss and accuracy
    def calc_sil_and_sia(self):
        if self.genSiImages is not None:
            [sil, sia] = self.model.evaluate_generator(self.genSiImages, self.siSteps)
        else:
            sil, sia = -1, -1
        return sil, sia
    # Save model checkpoint
    def save_model_checkpoint(self, epoch, tl, ta, vl, va, sil, sia):
        modelFilePath = os.path.join(saveDir,
            self.fileNamePre + "-epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-sil{5:.4f}-sia{6:.4f}.hdf5".format(epoch, tl, ta, vl, va, sil, sia))
        print("Saving model", modelFilePath)
        self.model.save_weights(modelFilePath)
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
        plt.suptitle(self.fileNamePre[:int(len(self.fileNamePre)/2)] + '\n' + self.fileNamePre[int(len(self.fileNamePre)/2):], fontsize=10)
        plt.savefig(os.path.join(saveDir, self.fileNamePre + "-Plots.png"))
        time.sleep(1)
        plt.close()


#################################################################
# FIT ON LABELED DATA
#################################################################


def fit_on_labelled_data(iterNumber, trainLabelledDirs, trainLabelledWordNumbers, trainLabelledWords,
        LSTMLipReaderModel, checkSIAndMakePlots, earlyStop, genValImages, valSteps, batchSize, nEpochs):
    # Make generator: Labelled generator has labels as input
    genTrainImagesLabelled = gen_these_word_images(trainLabelledDirs, trainLabelledWordNumbers,
                                                    allWords=trainLabelledWords, batchSize=batchSize, shuffle=True)
    trainLabelledSteps = len(trainLabelledDirs) // batchSize
    # FIT (gen)
    LSTMLipReaderModelHistory = LSTMLipReaderModel.fit_generator(genTrainImagesLabelled, steps_per_epoch=trainLabelledSteps, epochs=nEpochs, verbose=True,
                                                            callbacks=[checkSIAndMakePlots, earlyStop], validation_data=genValImages, validation_steps=valSteps)
    return LSTMLipReaderModel, checkSIAndMakePlots


#################################################################
# SAVE LOSSES AND ACC
#################################################################


def save_losses_and_accuracies(LSTMLipReaderModel, checkSIAndMakePlots,
        allApparentLabelledTrainLossesThruSelfLearning, allValLossesThruSelfLearning, allSiLossesThruSelfLearning,
        allApparentLabelledTrainAccuraciesThruSelfLearning, allValAccuraciesThruSelfLearning, allSiAccuraciesThruSelfLearning,
        trueLabelledTrainLossesThruPcOfLabelledData, apparentLabelledTrainLossesThruPcOfLabelledData,
        percentageOfLabelledData, trainLabelledDirs, trainLabelledWordNumbers, trainDirs, batchSize,
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
    # Append final losses and acc thru every iteration of self-learning
    percentageOfLabelledData.append(len(trainLabelledDirs)/len(trainDirs)*100)
    apparentLabelledTrainLossesThruPcOfLabelledData.append(checkSIAndMakePlots.trainLosses[-1])
    valLossesThruPcOfLabelledData.append(checkSIAndMakePlots.valLosses[-1])
    siLossesThruPcOfLabelledData.append(checkSIAndMakePlots.siLosses[-1])
    apparentLabelledTrainAccuraciesThruPcOfLabelledData.append(checkSIAndMakePlots.trainAccuracies[-1])
    valAccuraciesThruPcOfLabelledData.append(checkSIAndMakePlots.valAccuracies[-1])
    siAccuraciesThruPcOfLabelledData.append(checkSIAndMakePlots.siAccuracies[-1])
    # To calc true tl and ta
    print("Calculating true accuracy...")
    genTrainImagesLabelled = gen_these_word_images(trainLabelledDirs, trainLabelledWordNumbers, batchSize=batchSize, shuffle=False)
    trainLabelledSteps = len(trainLabelledDirs) // batchSize
    [trueTl, trueTa] = LSTMLipReaderModel.evaluate_generator(genTrainImagesLabelled, trainLabelledSteps)
    trueLabelledTrainLossesThruPcOfLabelledData.append(trueTl)
    trueLabelledTrainAccuraciesThruPcOfLabelledData.append(trueTa)

#################################################################
# PLOT ALL LOSSES AND ACC THRU ALL ITERS OF ALL EPOCHS OF SELF-LEARNING
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
            xAxisTicks.append(" ")
            tl.append(allApparentLabelledTrainLossesThruSelfLearning[iterNum][epochNum])
            vl.append(allValLossesThruSelfLearning[iterNum][epochNum])
            sil.append(allSiLossesThruSelfLearning[iterNum][epochNum])
            ta.append(allApparentLabelledTrainAccuraciesThruSelfLearning[iterNum][epochNum])
            va.append(allValAccuraciesThruSelfLearning[iterNum][epochNum])
            sia.append(allSiAccuraciesThruSelfLearning[iterNum][epochNum])
        # Add a vertical line after every iteration
        verticalLineX.append(count)
        # Write total numbr of epochs in this iter, in xTicks
        xAxisTicks[-int(len(allApparentLabelledTrainLossesThruSelfLearning[iterNum])/2)] = len(allApparentLabelledTrainLossesThruSelfLearning[iterNum])
    # Plot
    plt.subplot(211)
    plt.plot(tl, label='apparent train loss', color=plotColor, linestyle='--')
    plt.plot(vl, label='val loss', color=plotColor, linestyle='-')
    plt.plot(sil, label='SI loss', color=plotColor, linestyle='-.')
    plt.xticks(np.arange(len(xAxisTicks)), xAxisTicks, fontsize=8)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    for x in verticalLineX:
        plt.plot([x, x], plt.gca().get_ylim(), color='k')
    plt.xlabel('epochs within iterations of self-learning')
    plt.ylabel('loss')
    leg = plt.legend(loc='best', fontsize=10, fancybox=True)
    leg.get_frame().set_alpha(0.3)
    yMin = min(min(tl), min(vl), min(sil))
    yMax = max(max(tl), max(vl), max(sil))
    plt.ylim([yMin, yMax])
    plt.title("Loss")
    plt.tight_layout()
    plt.subplot(212)
    plt.plot(ta, label='apparent train acc', color=plotColor, linestyle='--')
    plt.plot(va, label='val acc', color=plotColor, linestyle='-')
    plt.plot(sia, label='SI acc', color=plotColor, linestyle='-.')
    plt.xticks(np.arange(len(xAxisTicks)), xAxisTicks, fontsize=8)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.gca().yaxis.grid(True)
    for x in verticalLineX:
        plt.plot([x, x], plt.gca().get_ylim(), color='k')
    plt.xlabel('epochs within iterations of self-learning')
    plt.ylabel('accuracy')
    leg = plt.legend(loc='best', fontsize=10, fancybox=True)
    leg.get_frame().set_alpha(0.3)
    plt.ylim([0, 1])
    plt.title("Accuracy")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle(plotName[:int(len(plotName)/2)] + '\n' + plotName[int(len(plotName)/2):], fontsize=10)
    plt.savefig(os.path.join(saveDir, plotName + "-All-Loss-Acc.png"))
    time.sleep(1)
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
    # Plot losses
    plt.subplot(211)
    plt.xlim([0, 100])
    plt.plot(percentageOfLabelledData, apparentLabelledTrainLossesThruPcOfLabelledData, label='apparent train loss', color=plotColor, marker='.')
    plt.plot(percentageOfLabelledData, trueLabelledTrainLossesThruPcOfLabelledData, label='true train loss', color=plotColor, marker='o')
    plt.plot(percentageOfLabelledData, valLossesThruPcOfLabelledData, label='val loss', color=plotColor, marker='D')
    plt.plot(percentageOfLabelledData, siLossesThruPcOfLabelledData, label='SI loss', color=plotColor, marker='+')
    xTicks = [0, 100]
    # Add xTicks at % of data
    for x in percentageOfLabelledData:
        plt.plot([x, x], plt.gca().get_ylim(), color='k', linestyle=':', alpha=0.5)
        xTicks.append(x)
    plt.xticks(xTicks, fontsize=8, rotation=90)
    plt.xlabel('% of labelled data')
    plt.ylabel('loss')
    leg = plt.legend(loc='best', fontsize=10, fancybox=True)
    leg.get_frame().set_alpha(0.3)
    yMin = min(min(apparentLabelledTrainLossesThruPcOfLabelledData), min(trueLabelledTrainLossesThruPcOfLabelledData), min(valLossesThruPcOfLabelledData), min(siLossesThruPcOfLabelledData))
    yMax = max(max(apparentLabelledTrainLossesThruPcOfLabelledData), max(trueLabelledTrainLossesThruPcOfLabelledData), max(valLossesThruPcOfLabelledData), max(siLossesThruPcOfLabelledData))
    plt.ylim([yMin, yMax])
    plt.title("Loss")
    plt.tight_layout()
    # Plot accuracies
    plt.subplot(212)
    plt.xlim([0, 100])
    plt.plot(percentageOfLabelledData, apparentLabelledTrainAccuraciesThruPcOfLabelledData, label='apparent train acc', color=plotColor, marker='.')
    plt.plot(percentageOfLabelledData, trueLabelledTrainAccuraciesThruPcOfLabelledData, label='true train acc', color=plotColor, marker='o')
    plt.plot(percentageOfLabelledData, valAccuraciesThruPcOfLabelledData, label='val acc', color=plotColor, marker='D')
    plt.plot(percentageOfLabelledData, siAccuraciesThruPcOfLabelledData, label='SI acc', color=plotColor, marker='+')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.gca().yaxis.grid(True, alpha=0.5)
    xTicks = [0, 100]
    for x in percentageOfLabelledData:
        plt.plot([x, x], plt.gca().get_ylim(), color='k', linestyle=':', alpha=0.5)
        xTicks.append(x)
    plt.xticks(xTicks, fontsize=8, rotation=90)
    plt.xlabel('% of labelled data')
    plt.ylabel('accuracy')
    leg = plt.legend(loc='best', fontsize=10, fancybox=True)
    leg.get_frame().set_alpha(0.3)
    plt.ylim([0, 1])
    plt.title("Accuracy")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle(plotName[:int(len(plotName)/2)] + '\n' + plotName[int(len(plotName)/2):], fontsize=10)
    plt.savefig(os.path.join(saveDir, plotName + "-pcOfLabelledData-Loss-Acc.png"))
    time.sleep(1)
    plt.close()


#############################################################
# ADD UNLABELLED DATA TO LABELLED DATA BASED ON PRED MAX VAL
#############################################################


def add_unlabelled_data_to_labelled_data(labelledPercent, iterNumber,
                        trainLabelledIdx, trainUnlabelledIdx,
                        trainLabelledDirs, trainLabelledWordNumbers, trainLabelledWords,
                        trainUnlabelledDirs, trainUnlabelledWordNumbers,
                        LSTMLipReaderModel, batchSize, fileNamePre,
                        percentageOfLabelledData, unlabelledLRPredMaxValueThresh=0.99,
                        criticModel=None, unlabelledCriticPredsYesThresh=0.1):
    print("Adding unlabelled data to labelled data...")
    if criticModel is not None:
        print("...with Critic model")
    else:
        print("...without Critic model")
    unlabelledLRPredMaxValues,  unlabelledCriticPreds, unlabelledLRPredWords, unlabelledActualWords = load_unlabelled_preds(
        trainUnlabelledDirs, trainUnlabelledWordNumbers, batchSize, LSTMLipReaderModel, criticModel
    )
    # Plot
    plot_max_values_and_accuracy(labelledPercent, iterNumber, fileNamePre, unlabelledLRPredMaxValues,
        unlabelledLRPredWords, unlabelledActualWords, unlabelledLRPredMaxValueThresh, percentageOfLabelledData,
        unlabelledCriticPreds, unlabelledCriticPredsYesThresh)
    # Choose those in the unlabelled set that exceed max value thresh
    maxValueFilter = unlabelledLRPredMaxValues > unlabelledLRPredMaxValueThresh
    print("Adding", np.sum(maxValueFilter), "dirs to labelled set")
    if criticModel is not None:
        maxValueFilter = np.logical_and(maxValueFilter,
            unlabelledCriticPreds > unlabelledCriticPredsYesThresh)
        print("Reducing to", np.sum(maxValueFilter), "because of critic.")
    newTrainLabelledIdx = trainUnlabelledIdx[maxValueFilter]
    newTrainDirsLabelled = trainUnlabelledDirs[maxValueFilter]
    newTrainWordNumbersLabelled = trainUnlabelledWordNumbers[maxValueFilter]
    newTrainWords = np.array(unlabelledLRPredWords)[maxValueFilter]
    # Add them to the labelled directories
    for i, directory, wordNum, predWord in zip(newTrainLabelledIdx, newTrainDirsLabelled, newTrainWordNumbersLabelled, newTrainWords):
        trainLabelledIdx = np.append(trainLabelledIdx, i)
        trainLabelledDirs = np.append(trainLabelledDirs, directory)
        trainLabelledWordNumbers = np.append(trainLabelledWordNumbers, wordNum)
        trainLabelledWords = np.append(trainLabelledWords, predWord)
    # Remove them from unlabelled directories
    trainUnlabelledIdx = trainUnlabelledIdx[np.logical_not(maxValueFilter)]
    trainUnlabelledDirs = trainUnlabelledDirs[np.logical_not(maxValueFilter)]
    trainUnlabelledWordNumbers = trainUnlabelledWordNumbers[np.logical_not(maxValueFilter)]
    return trainLabelledIdx, trainUnlabelledIdx, trainLabelledDirs, trainLabelledWordNumbers, trainLabelledWords, trainUnlabelledDirs, trainUnlabelledWordNumbers


def load_unlabelled_preds(trainUnlabelledDirs, trainUnlabelledWordNumbers, batchSize, LSTMLipReaderModel, criticModel=None):
    # Find confidence values of predictions
    unlabelledActualWords = []
    unlabelledLRPreds = []
    unlabelledLRPredWords = []
    if criticModel is not None:
        unlabelledCriticPreds = []
    else:
        unlabelledCriticPreds = None
    genTrainImagesUnlabelled = gen_these_word_images(trainUnlabelledDirs, trainUnlabelledWordNumbers, batchSize=batchSize, shuffle=False)
    trainUnlabelledSteps = len(trainUnlabelledDirs) // batchSize
    for step in tqdm.tqdm(range(trainUnlabelledSteps)):
        vids, words = next(genTrainImagesUnlabelled)
        actualWords = np.argmax(words, axis=1)
        preds = LSTMLipReaderModel.predict(vids)
        LRpredWords = np.argmax(preds, axis=1)
        if criticModel is not None:
            criticPreds = criticModel.predict([vids, np_utils.to_categorical(LRpredWords, wordsVocabSize)])
        for i in range(len(preds)):
            unlabelledActualWords.append(actualWords[i])
            unlabelledLRPreds.append(preds[i])
            unlabelledLRPredWords.append(LRpredWords[i])
            if criticModel is not None:
                unlabelledCriticPreds.append(criticPreds[i])
    genTrainImagesUnlabelledRemaining = gen_these_word_images(trainUnlabelledDirs[trainUnlabelledSteps * batchSize:],
        trainUnlabelledWordNumbers[trainUnlabelledSteps * batchSize:], batchSize=1, shuffle=False)
    trainUnlabelledRemainingSteps = len(trainUnlabelledDirs) - trainUnlabelledSteps * batchSize
    for step in tqdm.tqdm(range(trainUnlabelledRemainingSteps)):
        vids, words = next(genTrainImagesUnlabelledRemaining)
        actualWords = np.argmax(words, axis=1)
        preds = LSTMLipReaderModel.predict(vids)
        LRpredWords = np.argmax(preds, axis=1)
        if criticModel is not None:
            criticPreds = criticModel.predict([vids, np_utils.to_categorical(LRpredWords, wordsVocabSize)])
        for i in range(len(preds)):
            unlabelledActualWords.append(actualWords[i])
            unlabelledLRPreds.append(preds[i])
            unlabelledLRPredWords.append(LRpredWords[i])
            if criticModel is not None:
                unlabelledCriticPreds.append(criticPreds[i])
    # Max confidence value
    unlabelledLRPredMaxValues = np.max(np.array(unlabelledLRPreds), axis=1)
    if criticModel is not None:
        unlabelledCriticPreds = np.reshape(unlabelledCriticPreds, (len(unlabelledCriticPreds)))
        print("Histogram of unlabelledCriticPreds", np.histogram(unlabelledCriticPreds, 20, [0., 1.]))
    return unlabelledLRPredMaxValues, unlabelledCriticPreds, unlabelledLRPredWords, unlabelledActualWords


def plot_max_values_and_accuracy(labelledPercent, iterNumber, fileNamePre, unlabelledLRPredMaxValues, unlabelledLRPredWords, unlabelledActualWords,
        unlabelledLRPredMaxValueThresh, percentageOfLabelledData, unlabelledCriticPreds=None, unlabelledCriticPredsYesThresh=0.1):
    if unlabelledCriticPreds is None:
        unlabelledCriticPreds = np.zeros(len(unlabelledLRPredMaxValues),)
    # boolean
    noCritic = np.all(np.equal(unlabelledCriticPreds, np.zeros(len(unlabelledLRPredMaxValues),)))
    # Sort all according to unlabelledLRPredMaxValues
    sortedUnlabelledLRPredMaxValues, sortedUnlabelledLRPredWords, sortedUnlabelledActualWords, sortedUnlabelledCriticPreds = (
        np.reshape(t, (len(unlabelledLRPredMaxValues),)) for t in zip(*sorted(zip(unlabelledLRPredMaxValues, unlabelledLRPredWords, unlabelledActualWords, unlabelledCriticPreds), reverse=True))
    )
    # Accuracies
    # LR
    sortedUnlabelledAccuracyOnMaxValues = np.cumsum(np.equal(sortedUnlabelledLRPredWords, sortedUnlabelledActualWords)) / (1 + np.arange(len(sortedUnlabelledActualWords)))
    # Critic
    if not noCritic:
        print("Calculating LR+critic accuracies")
        sortedUnlabelledAccuracyWithCritic = np.zeros(sortedUnlabelledAccuracyOnMaxValues.shape)
        sortedUnlabelledPredsNumberWithCritic = np.zeros(sortedUnlabelledAccuracyOnMaxValues.shape)
        for i in tqdm.tqdm(range(len(sortedUnlabelledActualWords))):
            criticFilter = sortedUnlabelledCriticPreds[:i] > unlabelledCriticPredsYesThresh
            sortedUnlabelledPredsNumberWithCritic[i] = np.sum(criticFilter)
            sortedUnlabelledAccuracyWithCritic[i] = np.sum(np.equal(sortedUnlabelledLRPredWords[:i], sortedUnlabelledActualWords[:i])[criticFilter]) / (np.sum(criticFilter) + 1e-15)
    # Plots
    xLen = np.sum(sortedUnlabelledLRPredMaxValues > (unlabelledLRPredMaxValueThresh - 0.04))
    if xLen > 0:
        # All axes
        ax1 = host_subplot(111, axes_class=AA.Axes)
        ax2 = ax1.twiny()
        ax3 = ax1.twiny()
        ax3.axis['top'] = ax3.get_grid_helper().new_fixed_axis(loc='top', axes=ax3, offset=(0, 40))
        ax3.axis['top'].toggle(all=True)
        # LR pred max sorted values
        myX = np.arange(xLen)
        p1, = ax1.plot(myX, sortedUnlabelledLRPredMaxValues[:xLen], label='max value - LR', color='b')
        ax1.set_xlabel("Number of instances considered,\nsorted by predicted max value")
        # Unlabelled accuracy on LR pred max values
        for i in range(len(myX)):
            myX[i] = int(100*myX[i]/len(unlabelledLRPredMaxValues)*(100-percentageOfLabelledData[-1]))/100
        p2, = ax2.plot(myX, sortedUnlabelledAccuracyOnMaxValues[:xLen], label='accuracy - only LR')
        ax2.set_xlabel("Percentage of total data", fontsize=10)
        # Unlabelled accuracy on LR pred max values, with critic
        if not noCritic:
            for i in range(len(myX)):
                myX[i] = int(100*sortedUnlabelledPredsNumberWithCritic[i]/len(unlabelledLRPredMaxValues)*(100-percentageOfLabelledData[-1]))/100
            p3, = ax3.plot(myX, sortedUnlabelledAccuracyWithCritic[:xLen], label='accuracy - LR+critic')
            ax3.set_xlabel("With critic - lesser percentage of total data", fontsize=10)
        yLimMin = min(sortedUnlabelledLRPredMaxValues[:xLen][-1],
            sortedUnlabelledAccuracyOnMaxValues[:xLen][-1])
        if not noCritic:
            yLimMin = min(yLimMin, sortedUnlabelledAccuracyWithCritic[:xLen][-1])
        plt.ylim([yLimMin-0.01, 1.01])
        plt.gca().yaxis.grid(True)
        plt.ylabel("Max value,\nUnlabelled accuracy, trained from " + str(labelledPercent) + "% - iter{0:02d}".format(iterNumber))
        # plt.title("Unlabelled accuracy, trained with " + str(labelledPercent) + "% - iter{0:02d}".format(iterNumber), fontsize=12, y=3.1)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(saveDir, fileNamePre + "-unlabelled-accuracy-max-values.png"))
        time.sleep(1)
        plt.close()


# myX = [0, 1, 2, 3]
# myY1 = [0, 1, 2, 3]
# myY2 = [2, 3, 4, 5]
# myY3 = [4, 5, 6, 7]

# ax1 = host_subplot(111, axes_class=AA.Axes)
# # plt.subplots_adjust(top=25)
# ax2 = ax1.twiny()
# ax3 = ax1.twiny()
# ax3.axis['top'] = ax3.get_grid_helper().new_fixed_axis(loc='top', axes=ax3, offset=(0, 40))
# ax3.axis['top'].toggle(all=True)

# p1, = ax1.plot(myX, myY1, label='1', color='b')

# ax1.set_xlabel("Number of instances considered,\nsorted by predicted max value")
# ax2.set_xlabel("Percentage of total data")
# ax3.set_xlabel("Compressed Percentage of total data")

# for i in range(len(myX)):
#     myX[i] += 10

# p2, = ax2.plot(myX, myY2, label='2', color='g')

# for i in range(len(myX)):
#     myX[i] += 100

# p3, = ax3.plot(myX, myY3, label='3', color='r')

# ax1.axis["bottom"].label.set_color(p1.get_color())
# ax2.axis["top"].label.set_color(p2.get_color())
# ax3.axis["top"].label.set_color(p3.get_color())

# plt.text(0.5, 9.5, "Unlabelled accuracy, trained with ", fontsize=12)

# plt.tight_layout()
# plt.show()

#################################################################
# REFERENCES
#################################################################

# http://parneetk.github.io/blog/neural-networks-in-keras/


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
# sortedUnlabelledPredMaxValues, sortedUnlabelledLRPredWords, sortedUnlabelledActualWords, sortedTrainDirsUnlabelled, sortedTrainWordNumbersUnlabelled = (
#     np.array(t) for t in zip(*sorted(zip(unlabelledPredMaxValues, unlabelledPredWords, unlabelledActualWords, trainUnlabelledDirs, trainUnlabelledWordNumbers), reverse=True)))

# # Plot Accuracy
# unlabelledAccuracyOnMaxValues = np.cumsum(np.equal(sortedUnlabelledLRPredWords, sortedUnlabelledActualWords)) / (1 + np.arange(len(sortedUnlabelledActualWords)))
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
# # sortedUnlabelledPredMaxBy2MaxValues, sortedUnlabelledLRPredWords, sortedUnlabelledActualWords = (
# #     list(t) for t in zip(*sorted(zip(unlabelledPredMaxBy2MaxValues, unlabelledPredWords, unlabelledActualWords), reverse=True)))
# # unlabelledAccuracyOnMaxBy2MaxValues = np.cumsum(np.equal(sortedUnlabelledLRPredWords, sortedUnlabelledActualWords)) / (1 + np.arange(len(sortedUnlabelledActualWords)))
# # plt.plot(unlabelledAccuracyOnMaxBy2MaxValues[:int(labelledPercent/100*len(unlabelledAccuracyOnMaxBy2MaxValues))], label='accuracy')
# # plt.plot(sortedUnlabelledPredMaxBy2MaxValues[:int(labelledPercent/100*len(unlabelledAccuracyOnMaxValues))] / sortedUnlabelledPredMaxBy2MaxValues[0] / 8 + .8 , label='max/2max')
# # plt.legend(loc='best')
# # plt.xlabel("Number of instances, sorted by predicted max value divided by 2nd max value")
# # plt.ylabel("Accuracy")
# # plt.title("Unlabelled Accuracy with Max divided by 2nd Max Values trained with 10%-1", fontsize=12)
# # plt.show()

# # Choose those in the unlabelled set that exceed max value thresh
# unlabelledLRPredMaxValueThresh = 0.99
# maxValueFilter = sortedUnlabelledPredMaxValues > unlabelledLRPredMaxValueThresh
# newTrainDirsLabelled = sortedTrainDirsUnlabelled[maxValueFilter]
# newTrainWordNumbersLabelled = sortedTrainWordNumbersUnlabelled[maxValueFilter]
# newTrainWords = sortedUnlabelledLRPredWords[maxValueFilter]
# # Add them to the training directories
# for directory, wordNum, predWord in zip(newTrainDirsLabelled, newTrainWordNumbersLabelled, newTrainWords):
#     trainLabelledDirs.append(directory)
#     trainLabelledWordNumbers.append(wordNum)
#     trainLabelledWords.append(np_utils.to_categorical(predWord, wordsVocabSize))

# # Remove them from unlabelled directories
# trainUnlabelledDirs = sortedTrainDirsUnlabelled[len(newTrainDirsLabelled):]
# trainUnlabelledWordNumbers = sortedTrainWordNumbersUnlabelled[len(newTrainWordNumbersLabelled):]

