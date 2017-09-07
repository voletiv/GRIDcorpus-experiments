###################################################################
# MAP
###################################################################

# Load best models
LSTMLipReaderModel.load_weights(os.path.join(saveDir, "LSTM-noPadResults-h256-depth2-LSTMactivtanh-enc64-encodedActivsigmoid-Adam-1e-03-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub-epoch078-tl0.4438-ta0.8596-vl0.6344-va0.8103-sil3.2989-sia0.3186.hdf5"))
criticModel, filenamePre = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=False, wordDim=1,
                                useEncWord=True, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=True, oneHotWordDim=52, useOneHotWordFc=False, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)
criticModel.load_weights(os.path.join(saveDir, "C3DCritic-LRnoPadResults-l1f8-l2f16-l3f32-fc1n64-vid64-enc64-oneHotWord52-out64-Adam-5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub-epoch002-tl0.2837-ta0.8783-vl0.4017-va0.8255-sil1.4835-sia0.3520.hdf5"))

# TOTAL

# Generating functions
genTrainImages = genMouthImages(trainDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=False, shuffle=False, keepPadResults=False)
trainSteps = int(len(trainDirs) / batchSize)
genValImages = genMouthImages(valDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=False, shuffle=False, keepPadResults=False)
valSteps = int(len(valDirs) / batchSize)
genSiImages = genMouthImages(siDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=False, shuffle=False, keepPadResults=False)
siSteps = int(len(siDirs) / batchSize)

# TRAIN RESULTS
thresholds = np.arange(0, 1.001, 0.05)
LRTrainPreds = np.empty((0, 1), int)    # Whether the LR Prediction is correct or not
criticTrainPredsPerThresh = np.empty((0, len(thresholds)), int)    # What the Critic predicted
criticPredsThresh = np.zeros((batchSize*6, len(thresholds)))
for step in tqdm.tqdm((range(trainSteps))):
    # Ground truth
    vids, words = next(genTrainImages)
    words = np.argmax(words, axis=1)
    # Lip Reader predictions
    predWordFeatures = encoder.predict(vids)
    predWords = np.argmax(LSTMLipReaderModel.predict(vids), axis=1)
    correctPreds = np.array(words == predWords).astype(int)
    LRTrainPreds = np.vstack((LRTrainPreds, np.expand_dims(correctPreds, axis=1)))
    # Critic predictions
    criticPreds = criticModel.predict([vids, predWordFeatures, np_utils.to_categorical(predWords, wordsVocabSize)])
    for t, thresh in enumerate(thresholds):
        criticPredsThresh[:, t] = np.reshape((criticPreds >= thresh).astype(int), (batchSize*6,))
    criticTrainPredsPerThresh = np.vstack((criticTrainPredsPerThresh, criticPredsThresh))

LRTrainPreds = np.reshape(LRTrainPreds, (len(LRTrainPreds),))

trainTPPerThresh = np.zeros((len(thresholds)))
trainFPPerThresh = np.zeros((len(thresholds)))
trainFNPerThresh = np.zeros((len(thresholds)))
trainTNPerThresh = np.zeros((len(thresholds)))
for t, thresh in enumerate(thresholds):
    trainTPPerThresh[t] = np.sum(np.multiply((LRTrainPreds == 1), (criticTrainPredsPerThresh[:, t] == 1), dtype=int))
    trainFPPerThresh[t] = np.sum(np.multiply((LRTrainPreds == 1), (criticTrainPredsPerThresh[:, t] == 0), dtype=int))
    trainFNPerThresh[t] = np.sum(np.multiply((LRTrainPreds == 0), (criticTrainPredsPerThresh[:, t] == 1), dtype=int))
    trainTNPerThresh[t] = np.sum(np.multiply((LRTrainPreds == 0), (criticTrainPredsPerThresh[:, t] == 0), dtype=int))

lipReaderTrainAcc = np.sum(LRTrainPreds) / len(LRTrainPreds)
criticTrainAcc = (trainTPPerThresh[10] + trainTNPerThresh[10]) / len(criticTrainPredsPerThresh)

# VAL RESULTS
thresholds = np.arange(0, 1.001, 0.05)
LRValPreds = np.empty((0, 1), int)
criticValPredsPerThresh = np.empty((0, len(thresholds)), int)    # What the Critic predicted
criticPredsThresh = np.zeros((batchSize*6, len(thresholds)))
for step in tqdm.tqdm((range(valSteps))):
    # Ground truth
    vids, words = next(genValImages)
    # words = np.argmax(words[:, 0, :], axis=1)
    words = np.argmax(words, axis=1)
    # Lip Reader predictions
    predWordFeatures = encoder.predict(vids)
    # predWords = np.argmax(LSTMLipReaderModel.predict(vids)[:, 0, :], axis=1)
    predWords = np.argmax(LSTMLipReaderModel.predict(vids), axis=1)
    correctPreds = np.array(words == predWords).astype(int)
    LRValPreds = np.vstack((LRValPreds, np.expand_dims(correctPreds, axis=1)))
    # Critic predictions
    criticPreds = criticModel.predict([vids, predWordFeatures, np_utils.to_categorical(predWords, wordsVocabSize)])
    for t, thresh in enumerate(thresholds):
        criticPredsThresh[:, t] = np.reshape((criticPreds >= thresh).astype(int), (batchSize*6,))
    criticValPredsPerThresh = np.vstack((criticValPredsPerThresh, criticPredsThresh))

LRValPreds = np.reshape(LRValPreds, (len(LRValPreds),))

valTPPerThresh = np.zeros((len(thresholds)))
valFPPerThresh = np.zeros((len(thresholds)))
valFNPerThresh = np.zeros((len(thresholds)))
valTNPerThresh = np.zeros((len(thresholds)))
for t, thresh in enumerate(thresholds):
    valTPPerThresh[t] = np.sum(np.multiply((LRValPreds == 1), (criticValPredsPerThresh[:, t] == 1), dtype=int))
    valFPPerThresh[t] = np.sum(np.multiply((LRValPreds == 1), (criticValPredsPerThresh[:, t] == 0), dtype=int))
    valFNPerThresh[t] = np.sum(np.multiply((LRValPreds == 0), (criticValPredsPerThresh[:, t] == 1), dtype=int))
    valTNPerThresh[t] = np.sum(np.multiply((LRValPreds == 0), (criticValPredsPerThresh[:, t] == 0), dtype=int))

lipReaderValAcc = np.sum(LRValPreds) / len(LRValPreds)
criticValAcc = (valTPPerThresh[10] + valTNPerThresh[10]) / len(criticValPredsPerThresh)

# SPEAKER INDEPENDENT RESULTS
thresholds = np.arange(0, 1.001, 0.05)
LRSiPreds = np.empty((0, 1), int)
criticSiPredsPerThresh = np.empty((0, len(thresholds)), int)    # What the Critic predicted
criticPredsThresh = np.zeros((batchSize*6, len(thresholds)))
for step in tqdm.tqdm((range(siSteps))):
    # Ground truth
    vids, words = next(genSiImages)
    words = np.argmax(words, axis=1)
    # Lip Reader predictions
    predWordFeatures = encoder.predict(vids)
    predWords = np.argmax(LSTMLipReaderModel.predict(vids), axis=1)
    correctPreds = np.array(words == predWords).astype(int)
    LRSiPreds = np.vstack((LRSiPreds, np.expand_dims(correctPreds, axis=1)))
    # Critic predictions
    criticPreds = criticModel.predict([vids, predWordFeatures, np_utils.to_categorical(predWords, wordsVocabSize)])
    for t, thresh in enumerate(thresholds):
        criticPredsThresh[:, t] = np.reshape((criticPreds >= thresh).astype(int), (batchSize*6,))
    criticSiPredsPerThresh = np.vstack((criticSiPredsPerThresh, criticPredsThresh))

LRSiPreds = np.reshape(LRSiPreds, (len(LRSiPreds),))

siTPPerThresh = np.zeros((len(thresholds)))
siFPPerThresh = np.zeros((len(thresholds)))
siFNPerThresh = np.zeros((len(thresholds)))
siTNPerThresh = np.zeros((len(thresholds)))
for t, thresh in enumerate(thresholds):
    siTPPerThresh[t] = np.sum(np.multiply((LRSiPreds == 1), (criticSiPredsPerThresh[:, t] == 1), dtype=int))
    siFPPerThresh[t] = np.sum(np.multiply((LRSiPreds == 1), (criticSiPredsPerThresh[:, t] == 0), dtype=int))
    siFNPerThresh[t] = np.sum(np.multiply((LRSiPreds == 0), (criticSiPredsPerThresh[:, t] == 1), dtype=int))
    siTNPerThresh[t] = np.sum(np.multiply((LRSiPreds == 0), (criticSiPredsPerThresh[:, t] == 0), dtype=int))

lipReaderSiAcc = np.sum(LRSiPreds) / len(LRSiPreds)
criticSiAcc = (siTPPerThresh[10] + siTNPerThresh[10]) / len(criticSiPredsPerThresh)

# RESULTS
lrResults = np.concatenate(np.concatenate((LRTrainPreds, LRValPreds)), LRSiPreds)
criticResultsPerThresh = np.vstack((criticTrainPredsPerThresh, criticValPredsPerThresh, criticSiPredsPerThresh))
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

