#########################################################################################
# PLOTS for onlyLRPreds enc, word, oneHotWord, oneHotWord+fc10, enc+oneHotWord
#########################################################################################

myDir35 = '35-C3DCritic-onlyLRPreds-word-l1f8-l2f16-l3f32-fc64-vid64-word-out64-Adam-5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub'
myDir36a = '36a-C3DCritic-onlyLRPreds-enc-l1f8-l2f16-l3f32-fc1n64-vid64-enc64-oHn64-Adam-5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub'
myDir36b = '36b-C3DCritic-onlyLRPreds-enc-l1f8-l2f8-l3f8-fc1n64-vid10-enc64-oHn64-Adam-5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub'
myDir37 = '37a-C3DCritic-onlyLRPreds-oneHotWord-l1f8-l2f16-l3f32-fc64-vid64-oneHotWord52-out64-Adam5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub'
myDir38 = '38a-C3DCritic-onlyLRPreds-oneHotWord+fc-l1f8-l2f16-l3f32-fc64-vid64-oneHotWord52-oneHotHidden10-out64-Adam-5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub'
myDir39a = '39a-C3DCritic-onlyLRPreds-enc+oneHotWord-l1f8-l2f16-l3f32-fc64-vid64-enc64-oneHotWord52-out64-Adam-5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub'
myDir39b = '39b-C3DCritic-onlyLRPreds-enc+oneHotWord-l1f8-l2f8-l3f8-fc64-vid10-enc64-oneHotWord52-out64-Adam-5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub'
myDir40a = '40a-C3DCritic-onlyLRPreds-enc+fc+oneHotWord+fc-l1f8-l2f16-l3f32-fc64-vid64-enc64-encHidden10-oneHotWord52-oneHotHidden10-out64-Adam-5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub'
myDir40b = '40b-C3DCritic-onlyLRPreds-enc+fc+oneHotWord+fc-l1f8-l2f16-l3f32-fc64-vid10-enc64-oneHotWord52-out64-Adam-5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub'
myDir41 = '41a-C3DCritic-onlyLRPreds-predWordDis-l1f8-l2f16-l3f32-fc64-vid64-predWordDis52-out64-Adam-5e-04-GRIDcorpus-s0107-s0909-tMouth-vMouth-NOmeanSub'

import genMouthImages
import C3DCriticModel

### 35
# SPEAKER INDEPENDENT
genSiImages = genMouthImages(siDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=False)
f = os.path.join(mediaDir, myDir35)
silS = []
siaS = []
silE = []
siaE = []
criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=True, wordDim=1,
                                useEncWord=False, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=False, oneHotWordDim=52, useOneHotWordFc=False, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)
for i, file in tqdm.tqdm(enumerate(sorted(glob.glob(os.path.join(f, '*.hdf5'))))):
    print("Loading weights", file)
    criticModel.load_weights(file)
    for step in tqdm.tqdm(range(siValSteps)):
        vids, words = next(genSiImages)
        words = np.argmax(words[:, 0, :], axis=1)
        predWordDis = LSTMLipReaderModel.predict(vids)[:, 0, :]
        predWords = np.argmax(predWordDis, axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = predWords / wordsVocabSize
        outputs = correctPreds
        sil, sia = criticModel.evaluate([inputs1, inputs2], outputs, batch_size=batchSize)
        silS.append(sil)
        siaS.append(sia)
    # Append values
    silE.append(np.mean(silS))
    silS = []
    siaE.append(np.mean(siaS))
    siaS = []

sil35 = list(silE)
sia35 = list(siaE)


### 36a
# SPEAKER INDEPENDENT
genSiImages = genMouthImages(siDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=False)
f = os.path.join(mediaDir, myDir36a)
silS = []
siaS = []
silE = []
siaE = []
criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=False, wordDim=1,
                                useEncWord=True, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=False, oneHotWordDim=52, useOneHotWordFc=False, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)
for i, file in tqdm.tqdm(enumerate(sorted(glob.glob(os.path.join(f, '*.hdf5'))))):
    if i < 7:
        continue
    print("Loading weights", file)
    criticModel.load_weights(file)
    genSiImages = genMouthImages(siDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
                                    wordsVocabSize=wordsVocabSize, useMeanMouthImage=False)
    for step in tqdm.tqdm(range(siValSteps)):
        vids, words = next(genSiImages)
        words = np.argmax(words[:, 0, :], axis=1)
        encWordFeatures = encoder.predict(vids)
        predWordDis = LSTMLipReaderModel.predict(vids)[:, 0, :]
        predWords = np.argmax(predWordDis, axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = encWordFeatures
        outputs = correctPreds
        sil, sia = criticModel.evaluate([inputs1, inputs2], outputs, batch_size=batchSize)
        silS.append(sil)
        siaS.append(sia)
        print('{0:.4f}'.format(sil), '{0:.4f}'.format(sia), '{0:.4f}'.format(np.mean(silS)), '{0:.4f}'.format(np.mean(siaS)))
    # Append values
    silE.append(np.mean(silS))
    silS = []
    siaE.append(np.mean(siaS))
    siaS = []

sil36a = list(silE)
sia36a = list(siaE)


### 37a
# SPEAKER INDEPENDENT
myDir = myDir37a
genSiImages = genMouthImages(siDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=False)
f = os.path.join(mediaDir, myDir)
silS = []
siaS = []
silE = []
siaE = []
criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=False, wordDim=1,
                                useEncWord=False, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=True, oneHotWordDim=52, useOneHotWordFc=False, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)
for i, file in tqdm.tqdm(enumerate(sorted(glob.glob(os.path.join(f, '*.hdf5'))))):
    print("Loading weights", file)
    criticModel.load_weights(file)
    for step in tqdm.tqdm(range(siValSteps)):
        vids, words = next(genSiImages)
        words = np.argmax(words[:, 0, :], axis=1)
        predWordDis = LSTMLipReaderModel.predict(vids)[:, 0, :]
        predWords = np.argmax(predWordDis, axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = np_utils.to_categorical(predWords, wordsVocabSize)
        outputs = correctPreds
        sil, sia = criticModel.evaluate([inputs1, inputs2], outputs, batch_size=batchSize)
        silS.append(sil)
        siaS.append(sia)
    # Append values
    silE.append(np.mean(silS))
    silS = []
    siaE.append(np.mean(siaS))
    siaS = []

sil37a = list(silE)
sia37a = list(siaE)


### 38a
# SPEAKER INDEPENDENT
myDir = myDir38a
genSiImages = genMouthImages(siDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=False)
f = os.path.join(mediaDir, myDir)
silS = []
siaS = []
silE = []
siaE = []
criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=False, wordDim=1,
                                useEncWord=False, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=True, oneHotWordDim=52, useOneHotWordFc=True, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)
for i, file in tqdm.tqdm(enumerate(sorted(glob.glob(os.path.join(f, '*.hdf5'))))):
    print("Loading weights", file)
    criticModel.load_weights(file)
    for step in tqdm.tqdm(range(siValSteps)):
        vids, words = next(genSiImages)
        words = np.argmax(words[:, 0, :], axis=1)
        predWordDis = LSTMLipReaderModel.predict(vids)[:, 0, :]
        predWords = np.argmax(predWordDis, axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = np_utils.to_categorical(predWords, wordsVocabSize)
        outputs = correctPreds
        sil, sia = criticModel.evaluate([inputs1, inputs2], outputs, batch_size=batchSize)
        silS.append(sil)
        siaS.append(sia)
    # Append values
    silE.append(np.mean(silS))
    silS = []
    siaE.append(np.mean(siaS))
    siaS = []

sil38a = list(silE)
sia38a = list(siaE)


### 39a
# SPEAKER INDEPENDENT
myDir = myDir39a
genSiImages = genMouthImages(siDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=False)
f = os.path.join(mediaDir, myDir)
silS = []
siaS = []
silE = []
siaE = []
criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=False, wordDim=1,
                                useEncWord=True, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=True, oneHotWordDim=52, useOneHotWordFc=False, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)
for i, file in tqdm.tqdm(enumerate(sorted(glob.glob(os.path.join(f, '*.hdf5'))))):
    print("Loading weights", file)
    criticModel.load_weights(file)
    for step in tqdm.tqdm(range(siValSteps)):
        vids, words = next(genSiImages)
        words = np.argmax(words[:, 0, :], axis=1)
        encodedWordFeatures = encoder.predict(vids)
        predWordDis = LSTMLipReaderModel.predict(vids)[:, 0, :]
        predWords = np.argmax(predWordDis, axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = encodedWordFeatures
        inputs3 = np_utils.to_categorical(predWords, wordsVocabSize)
        outputs = correctPreds
        sil, sia = criticModel.evaluate([inputs1, inputs2, inputs3], outputs, batch_size=batchSize)
        silS.append(sil)
        siaS.append(sia)
        print('{0:.4f}'.format(sil), '{0:.4f}'.format(sia), '{0:.4f}'.format(np.mean(silS)), '{0:.4f}'.format(np.mean(siaS)))
    # Append values
    silE.append(np.mean(silS))
    silS = []
    siaE.append(np.mean(siaS))
    siaS = []

sil39a = list(silE)
sia39a = list(siaE)


### 39b
# SPEAKER INDEPENDENT
myDir = myDir39b
genSiImages = genMouthImages(siDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=False)
f = os.path.join(mediaDir, myDir)
silS = []
siaS = []
silE = []
siaE = []
criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=8, layer3Filters=8, fc1Nodes=64, vidFeaturesDim=10,
                                useWord=False, wordDim=1,
                                useEncWord=True, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=True, oneHotWordDim=52, useOneHotWordFc=False, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)
for i, file in tqdm.tqdm(enumerate(sorted(glob.glob(os.path.join(f, '*.hdf5'))))):
    print("Loading weights", file)
    criticModel.load_weights(file)
    for step in tqdm.tqdm(range(siValSteps)):
        vids, words = next(genSiImages)
        words = np.argmax(words[:, 0, :], axis=1)
        encodedWordFeatures = encoder.predict(vids)
        predWordDis = LSTMLipReaderModel.predict(vids)[:, 0, :]
        predWords = np.argmax(predWordDis, axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = encodedWordFeatures
        inputs3 = np_utils.to_categorical(predWords, wordsVocabSize)
        outputs = correctPreds
        sil, sia = criticModel.evaluate([inputs1, inputs2, inputs3], outputs, batch_size=batchSize)
        silS.append(sil)
        siaS.append(sia)
        print('{0:.4f}'.format(sil), '{0:.4f}'.format(sia), '{0:.4f}'.format(np.mean(silS)), '{0:.4f}'.format(np.mean(siaS)))
    # Append values
    silE.append(np.mean(silS))
    silS = []
    siaE.append(np.mean(siaS))
    siaS = []

sil39b = list(silE)
sia39b = list(siaE)


### 40a
# SPEAKER INDEPENDENT
myDir = myDir40a
genSiImages = genMouthImages(siDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=False)
f = os.path.join(mediaDir, myDir40a)
silS = []
siaS = []
silE = []
siaE = []
criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=False, wordDim=1,
                                useEncWord=True, encWordDim=64, useEncWordFc=True, encWordFcDim=10,
                                useOneHotWord=True, oneHotWordDim=52, useOneHotWordFc=True, oneHotWordFcDim=10,
                                usePredWordDis=False, predWordDisDim=52,
                                outputHDim=64)
for i, file in tqdm.tqdm(enumerate(sorted(glob.glob(os.path.join(f, '*.hdf5'))))):
    print("Loading weights", file)
    criticModel.load_weights(file)
    for step in tqdm.tqdm(range(siValSteps)):
        vids, words = next(genSiImages)
        words = np.argmax(words[:, 0, :], axis=1)
        encodedWordFeatures = encoder.predict(vids)
        predWordDis = LSTMLipReaderModel.predict(vids)[:, 0, :]
        predWords = np.argmax(predWordDis, axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = encodedWordFeatures
        inputs3 = np_utils.to_categorical(predWords, wordsVocabSize)
        outputs = correctPreds
        sil, sia = criticModel.evaluate([inputs1, inputs2, inputs3], outputs, batch_size=batchSize)
        silS.append(sil)
        siaS.append(sia)
    # Append values
    silE.append(np.mean(silS))
    silS = []
    siaE.append(np.mean(siaS))
    siaS = []

sil40a = list(silE)
sia40a = list(siaE)


### 41a
# SPEAKER INDEPENDENT
myDir = myDir41a
genSiImages = genMouthImages(siDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
                                wordsVocabSize=wordsVocabSize, useMeanMouthImage=False)
f = os.path.join(mediaDir, myDir41a)
silS = []
siaS = []
silE = []
siaE = []
criticModel = C3DCriticModel(layer1Filters=8, layer2Filters=16, layer3Filters=32, fc1Nodes=64, vidFeaturesDim=64,
                                useWord=False, wordDim=1,
                                useEncWord=False, encWordDim=64, useEncWordFc=False, encWordFcDim=10,
                                useOneHotWord=False, oneHotWordDim=52, useOneHotWordFc=False, oneHotWordFcDim=10,
                                usePredWordDis=True, predWordDisDim=52,
                                outputHDim=64)
for i, file in tqdm.tqdm(enumerate(sorted(glob.glob(os.path.join(f, '*.hdf5'))))):
    print("Loading weights", file)
    criticModel.load_weights(file)
    for step in tqdm.tqdm(range(siValSteps)):
        vids, words = next(genSiImages)
        words = np.argmax(words[:, 0, :], axis=1)
        predWordDis = LSTMLipReaderModel.predict(vids)[:, 0, :]
        predWords = np.argmax(predWordDis, axis=1)
        correctPreds = np.array(words == predWords).astype(int)
        inputs1 = vids
        inputs2 = predWordDis
        outputs = correctPreds
        sil, sia = criticModel.evaluate([inputs1, inputs2], outputs, batch_size=batchSize)
        silS.append(sil)
        siaS.append(sia)
    # Append values
    silE.append(np.mean(silS))
    silS = []
    siaE.append(np.mean(siaS))
    siaS = []

sil41a = list(silE)
sia41a = list(siaE)

# RENAME
myDir = myDir39b
sil = sil39b
sia = sia39b
f = os.path.join(mediaDir, myDir)
for i, file in enumerate(sorted(glob.glob(os.path.join(f, '*.hdf5')))):
    os.rename(file, '.'.join(file.split('.')[:-1])+'-sil{0:.4f}-sia{1:.4f}.hdf5'.format(sil[i], sia[i]))


### 35
f35 = os.path.join(mediaDir, myDir35)
tl35 = []
ta35 = []
vl35 = []
va35 = []
sil35 = []
sia35 = []
for file in sorted(glob.glob(os.path.join(f35, '*.hdf5'))):
    tl35.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+2][2:]))
    ta35.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+3][2:]))
    vl35.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+4][2:]))
    va35.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+5][2:]))
    sil35.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+6][3:]))
    sia35.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+7][3:]))


### 36a
f36a = os.path.join(mediaDir, myDir36a)
tl36a = []
ta36a = []
vl36a = []
va36a = []
sil36a = []
sia36a = []
for file in sorted(glob.glob(os.path.join(f36a, '*.hdf5'))):
    tl36a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+2][2:]))
    ta36a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+3][2:]))
    vl36a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+4][2:]))
    va36a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+5][2:]))
    sil36a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+6][3:]))
    sia36a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+7][3:]))


### 36b
f36b = os.path.join(mediaDir, myDir36b)
tl36b = []
ta36b = []
vl36b = []
va36b = []
sil36b = []
sia36b = []
for file in sorted(glob.glob(os.path.join(f36b, '*.hdf5'))):
    tl36b.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+2][2:]))
    ta36b.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+3][2:]))
    vl36b.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+4][2:]))
    va36b.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+5][2:]))
    # sil36b.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+6][3:]))
    # sia36b.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+7][3:]))


### 37a
f37 = os.path.join(mediaDir, myDir37)
tl37 = []
ta37 = []
vl37 = []
va37 = []
sil37 = []
sia37 = []
for file in sorted(glob.glob(os.path.join(f37, '*.hdf5'))):
    tl37.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+2][2:]))
    ta37.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+3][2:]))
    vl37.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+4][2:]))
    va37.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+5][2:]))
    sil37.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+6][3:]))
    sia37.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+7][3:]))


### 38
f38 = os.path.join(mediaDir, myDir38)
tl38 = []
ta38 = []
vl38 = []
va38 = []
sil38 = []
sia38 = []
for file in sorted(glob.glob(os.path.join(f38, '*.hdf5'))):
    tl38.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+2][2:]))
    ta38.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+3][2:]))
    vl38.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+4][2:]))
    va38.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+5][2:]))
    sil38.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+6][3:]))
    sia38.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+7][3:]))


### 39a
f39a = os.path.join(mediaDir, myDir39a)
tl39a = []
ta39a = []
vl39a = []
va39a = []
sil39a = []
sia39a = []
for file in sorted(glob.glob(os.path.join(f39a, '*.hdf5'))):
    tl39a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+2][2:]))
    ta39a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+3][2:]))
    vl39a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+4][2:]))
    va39a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+5][2:]))
    sil39a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+6][3:]))
    sia39a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+7][3:]))


### 39b
f = os.path.join(mediaDir, myDir39b)
tl39b = []
ta39b = []
vl39b = []
va39b = []
sil39b = []
sia39b = []
for file in sorted(glob.glob(os.path.join(f, '*.hdf5'))):
    tl39b.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+2][2:]))
    ta39b.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+3][2:]))
    vl39b.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+4][2:]))
    va39b.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+5][2:]))
    sil39b.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+6][3:]))
    sia39b.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+7][3:]))


### 40a
f40a = os.path.join(mediaDir, myDir40a)
tl40a = []
ta40a = []
vl40a = []
va40a = []
sil40a = []
sia40a = []
for file in sorted(glob.glob(os.path.join(f40a, '*.hdf5'))):
    tl40a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+2][2:]))
    ta40a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+3][2:]))
    vl40a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+4][2:]))
    va40a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+5][2:]))
    sil40a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+6][3:]))
    sia40a.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+7][3:]))


### 41a
f41 = os.path.join(mediaDir, myDir41)
tl41 = []
ta41 = []
vl41 = []
va41 = []
sil41 = []
sia41 = []
for file in sorted(glob.glob(os.path.join(f41, '*.hdf5'))):
    tl41.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+2][2:]))
    ta41.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+3][2:]))
    vl41.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+4][2:]))
    va41.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+5][2:]))
    sil41.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+6][3:]))
    sia41.append(float(file.split('/')[-1].split('.hdf5')[0].split('-')[file.split('/')[-1].split('.hdf5')[0].split('-').index('NOmeanSub')+7][3:]))

# RENAME
myDir = myDir39b
sil = sil39b
sia = sia39b
f = os.path.join(mediaDir, myDir)
for i, file in enumerate(sorted(glob.glob(os.path.join(f, '*.hdf5')))):
    os.rename(file, '.'.join(file.split('.')[:-1])+'-sil{0:.4f}-sia{1:.4f}.hdf5'.format(sil[i], sia[i]))


# plt.plot(tl35, label='tl-35-Word', color='m', linestyle='-.')
# plt.plot(vl35, label='vl-35-Word', color='m', linestyle='-')
# plt.plot(sil35, label='sil-35-Word', color='m', linestyle='--')
plt.plot(tl36a, label='tl-36a-encWord', color='b', linestyle='-.')
plt.plot(vl36a, label='vl-36a-encWord', color='b', linestyle='-')
plt.plot(sil36a, label='sil-36a-encWord', color='b', linestyle='--')
# plt.plot(tl37, label='tl-37-OHWord', color='r', linestyle='-.')
# plt.plot(vl37, label='vl-37-OHWord', color='r', linestyle='-')
# plt.plot(sil37, label='sil-37-OHWord', color='r', linestyle='--')
# plt.plot(tl38, label='tl-38-OHWord-fc', color='g', linestyle='-.')
# plt.plot(vl38, label='vl-38-OHWord-fc', color='g', linestyle='-')
# plt.plot(sil38, label='sil-38-OHWord-fc', color='g', linestyle='--')
# plt.plot(tl39a, label='tl-39a-Enc-OHWord', color='k', linestyle='-.')
# plt.plot(vl39a, label='vl-39a-Enc-OHWord', color='k', linestyle='-')
# plt.plot(sil39a, label='sil-39a-Enc-OHWord', color='k', linestyle='--')
# plt.plot(tl39b, label='tl-39b-small-Enc-OHWord', color='#ffa500', linestyle='-.')
# plt.plot(vl39b, label='vl-39b-Enc-OHWord', color='#ffa500', linestyle='-')
# plt.plot(sil39b, label='sil-39b-Enc-OHWord', color='#ffa500', linestyle='--')
# plt.plot(tl40a, label='tl-40a-Enc-fc-OHWord-fc', color='y', linestyle='-.')
# plt.plot(vl40a, label='vl-40a-Enc-fc-OHWord-fc', color='y', linestyle='-')
# plt.plot(sil40a, label='sil-40a-Enc-fc-OHWord-fc', color='y', linestyle='--')
# plt.plot(tl41, label='tl-41-predWordDis', color='c', linestyle='-.')
# plt.plot(vl41, label='vl-41-predWordDis', color='c', linestyle='-')
# plt.plot(sil41, label='sil-41-predWordDis', color='c', linestyle='--')
leg = plt.legend(loc='lower right', fontsize=11, fancybox=True)
leg.get_frame().set_alpha(0.3)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.xticks(np.arange(0, 18, 1))
# plt.yticks(np.arange(0.4, 0.8, 0.1))
# plt.savefig(os.path.join(mediaDir, "35ato41a-losses-onlyLRPreds-word-enc-OHW-OHWhid-encOHW-encHidOHWHid-predWordDis.png"))
plt.savefig(os.path.join(mediaDir, "36a-losses-onlyLRPreds-word-enc-OHW-OHWhid-encOHW-encHidOHWHid-predWordDis.png"))
plt.close()

# plt.plot(ta35, label='ta-35-Word', color='m', linestyle='-.')
# plt.plot(va35, label='va-35-Word', color='m', linestyle='-')
# plt.plot(sia35, label='sia-35-Word', color='m', linestyle='--')
plt.plot(ta36a, label='ta-36a-encWord', color='b', linestyle='-.')
plt.plot(va36a, label='va-36a-encWord', color='b', linestyle='-')
plt.plot(sia36a, label='sia-36a-encWord', color='b', linestyle='--')
# plt.plot(ta37, label='ta-37-OHWord', color='r', linestyle='-.')
# plt.plot(va37, label='va-37-OHWord', color='r', linestyle='-')
# plt.plot(sia37, label='sia-37-OHWord', color='r', linestyle='--')
# plt.plot(ta38, label='ta-38-OHWord-fc', color='g', linestyle='-.')
# plt.plot(va38, label='va-38-OHWord-fc', color='g', linestyle='-')
# plt.plot(sia38, label='sia-38-OHWord-fc', color='g', linestyle='--')
# plt.plot(ta39a, label='ta-39a-Enc-OHWord', color='k', linestyle='-.')
# plt.plot(va39a, label='va-39a-Enc-OHWord', color='k', linestyle='-')
# plt.plot(sia39a, label='sia-39a-Enc-OHWord', color='k', linestyle='--')
# plt.plot(ta39b, label='ta-39b-Enc-OHWord', color='#ffa500', linestyle='-.')
# plt.plot(va39b, label='va-39b-Enc-OHWord', color='#ffa500', linestyle='-')
# plt.plot(sia39b, label='sia-39b-Enc-OHWord', color='#ffa500', linestyle='--')
# plt.plot(ta40a, label='ta-40a-EncFc-OHWordFc', color='y', linestyle='-.')
# plt.plot(va40a, label='va-40a-EncFc-OHWordFc', color='y', linestyle='-')
# plt.plot(sia40a, label='sia-40a-EncFc-OHWordFc', color='y', linestyle='--')
# plt.plot(ta41, label='ta-41-predWordDis', color='c', linestyle='-.')
# plt.plot(va41, label='va-41-predWordDis', color='c', linestyle='-')
# plt.plot(sia41, label='sia-41-predWordDis', color='c', linestyle='--')
leg = plt.legend(loc='best', fontsize=11, fancybox=True)
leg.get_frame().set_alpha(0.3)
plt.xlabel('epochs')
plt.ylabel('acc')
plt.xticks(np.arange(0, 18, 1))
plt.yticks(np.arange(0.45, 1, 0.05))
plt.gca().yaxis.grid(True)
plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
# plt.savefig(os.path.join(mediaDir, "35ato41a-acc-onlyLRPreds-word-enc-OHW-OHWhid-encOHW-encHidOHWHid-predWordDis.png"))
# plt.savefig(os.path.join(mediaDir, "35ato41a-va-onlyLRPreds-word-enc-OHW-OHWhid-encOHW-encHidOHWHid-predWordDis.png"))
plt.savefig(os.path.join(mediaDir, "36a-acc-onlyLRPreds-word-enc-OHW-OHWhid-encOHW-encHidOHWHid-predWordDis.png"))
plt.close()
