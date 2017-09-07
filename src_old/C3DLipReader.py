#############################################################
# C3D LipReader MODEL
#############################################################

# Manual seeds
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)

layer1Filters = 32
layer2Filters = 32
layer3Filters = 32
fc1Nodes = 2048
vidFeaturesDim = 2048

myInput = Input(shape=(framesPerWord, nOfMouthPixels,))
# 0 1 2 3  2 3 4 5  4 5 6 7  6 7 8 9  8 9 10 11  10 11 12 13
i1 = Lambda(lambda x: x[:, 0:4, :])(myInput)
i2 = Lambda(lambda x: x[:, 2:6, :])(myInput)
i3 = Lambda(lambda x: x[:, 4:8, :])(myInput)
i4 = Lambda(lambda x: x[:, 6:10, :])(myInput)
i5 = Lambda(lambda x: x[:, 8:12, :])(myInput)
i6 = Lambda(lambda x: x[:, 10:14, :])(myInput)

layer1 = Sequential()
layer1.add(Reshape((4, mouthW, mouthH, 1), input_shape=(4, nOfMouthPixels)))
layer1.add(Conv3D(layer1Filters, 3, activation='elu'))
layer1.add(BatchNormalization())
layer1.add(MaxPooling3D(pool_size=(2, 2, 2)))

x1 = layer1(i1)
x2 = layer1(i2)
x3 = layer1(i3)
x4 = layer1(i4)
x5 = layer1(i5)
x6 = layer1(i6)

y1 = concatenate([x1, x2], axis=1)
y2 = concatenate([x3, x4], axis=1)
y3 = concatenate([x5, x6], axis=1)

layer2 = Sequential()
layer2.add(Conv3D(layer2Filters, (2, 3, 3), activation='elu',
                  input_shape=(2, 19, 19, layer1Filters)))
layer2.add(BatchNormalization())
layer2.add(MaxPooling3D(pool_size=(1, 2, 2)))

z1 = layer2(y1)
z2 = layer2(y2)
z3 = layer2(y3)

z = concatenate([z1, z2, z3], axis=1)

layer3 = Sequential()
layer3.add(Conv3D(layer3Filters, (3, 3, 3), activation='elu',
                  input_shape=(3, 3, 3, layer2Filters)))
layer3.add(BatchNormalization())

z = layer3(z)
z = Flatten()(z)
z = Dense(fc1Nodes, activation='elu')(z)
vidFeatures = Dense(vidFeaturesDim, activation='elu')(z)

myWord = Dense(wordsVocabSize, activation='softmax')(vidFeatures)
myWord = Reshape((1, wordsVocabSize))(myWord)
myPad = Dense(wordsVocabSize, activation='softmax')(vidFeatures)
myPad = Reshape((1, wordsVocabSize))(myPad)

myOutput = concatenate([myWord, myPad], axis=1)

c3dLipReaderModel = Model(inputs=myInput, outputs=myOutput)

lr = 1e-4
adam = Adam(lr=lr)
c3dLipReaderModel.compile(optimizer=adam, loss='categorical_crossentropy',
                          metrics=['accuracy'])

c3dLipReaderModel.summary()

filenamePre = 'C3DLipReader-l1f' + str(layer1Filters) + \
    '-l2f' + str(layer2Filters) + \
    '-l3f' + str(layer3Filters) + \
    '-fc1n' + str(fc1Nodes) + \
    '-vid' + str(vidFeaturesDim) + \
    '-Adam-%1.e' % lr + \
    '-GRIDcorpus-s'
print(filenamePre)

# Save Model Architecture
model_yaml = c3dLipReaderModel.to_yaml()
with open(os.path.join(saveDir, "modelArch-" + filenamePre + ".yaml"), "w") as yaml_file:
    yaml_file.write(model_yaml)

# i = np.zeros((2, 14, 20, 20, 1))
# w = np.zeros((2, 1))
# model.predict([i, w])


#############################################################
# COMPLETE FILENAMEPRE
#############################################################

# SPEAKERS LIST IN FILENAME
prevS = -1
for s in speakersList:
    # print(s)
    if prevS == -1:
        filenamePre += '{0:02d}'.format(s)
    elif s - prevS > 1:
        filenamePre += '{0:02d}-s{1:02d}'.format(prevS, s)
    # Prev S
    prevS = s
    # print(filenamePre)

# Last speaker
filenamePre += '{0:02d}'.format(s)

if trainAlign:
    filenamePre += '-tAlign'
else:
    filenamePre += '-tMouth'

if valAlign:
    filenamePre += '-vAlign'
else:
    filenamePre += '-vMouth'

if useMeanMouthImage:
    filenamePre += '-meanSub'
else:
    filenamePre += '-NOmeanSub'

print(filenamePre)


#############################################################
# TRAIN C3D LipReader MODEL
#############################################################

nEpochs = 500
initEpoch = 0
batchSize = 16     # num of speaker vids


class LossHistoryAndCheckpoint(Callback):
    # On train start

    def on_train_begin(self, logs={}):
        self.losses = []
        self.valLosses = []
        self.acc = []
        self.valAcc = []
        self.sil = []
        self.sia = []

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
                float(file.split('/')[-1].split('-')[epochIdx + 1][2:]))
            self.acc.append(
                float(file.split('/')[-1].split('-')[epochIdx + 2][2:]))
            self.valLosses.append(
                float(file.split('/')[-1].split('-')[epochIdx + 3][2:]))
            self.valAcc.append(
                float(file.split('/')[-1].split('-')[epochIdx + 4][2:]))
            self.sil.append(
                float(file.split('/')[-1].split('-')[epochIdx + 5][3:]))
            self.sia.append(
                float(file.split('/')[-1].split('-')[epochIdx + 6][3:-5]))
    
    # At every epoch
    def on_epoch_end(self, epoch, logs={}):
        tl = logs.get('loss')
        ta = logs.get('acc')
        vl = logs.get('val_loss')
        va = logs.get('val_acc')
        # Speaker-Independent
        [l, a] = c3dLipReaderModel.evaluate_generator(genMouthImages(siDirs, batchSize=batchSize, align=False, wordIdx=wordIdx,
                                                                  wordsVocabSize=wordsVocabSize, useMeanMouthImage=False, meanMouthImage=meanMouthImage), siValSteps)
        # Checkpoint
        filepath = os.path.join(saveDir,
                                filenamePre + "-epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-sil{5:.4f}-sia{6:.4f}.hdf5".format(epoch, tl, ta, vl, va, sil, sia))
        checkpoint = ModelCheckpoint(
            filepath, verbose=1, save_best_only=False, save_weights_only=True, period=1)
        print("Saving plots for epoch " + str(epoch))
        self.losses.append(tl)
        self.valLosses.append(vl)
        self.acc.append(ta)
        self.valAcc.append(va)
        plt.plot(self.losses, label='trainingLoss')
        plt.plot(self.valLosses, label='valLoss')
        plt.legend(loc='best')
        plt.xlabel('epochs')
        plt.savefig(os.path.join(saveDir, filenamePre + "-plotLosses.png"))
        plt.close()
        plt.plot(self.acc, label='trainingAcc')
        plt.plot(self.valAcc, label='valAcc')
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.legend(loc='best')
        plt.xlabel('epochs')
        plt.gca().yaxis.grid(True)
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
        plt.savefig(os.path.join(saveDir, filenamePre + "-plotAcc.png"))
        plt.close()

lossHistory = LossHistory()

# # Load previous lipReaderModel
# c3dLipReaderModel.load_weights(os.path.join(
#     saveDir, "C3DLipReader-l1f8-l2f8-l3f16-fc1n256-fc2n512-Adam-1e-02-GRIDcorpus-s0107-s0920-s2234-epoch002-tl1.5618-ta0.5190-vl1.6555-va0.5048.hdf5"))
# initEpoch = 3

# FIT (gen)
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)
c3dLipReaderHistory = c3dLipReaderModel.fit_generator(genMouthImages(trainDirs, batchSize=batchSize, align=trainAlign, wordIdx=wordIdx, wordsVocabSize=wordsVocabSize,
                                                                     useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage),
                                                      steps_per_epoch=trainSteps, epochs=nEpochs, verbose=1, callbacks=[checkpoint, lossHistory],
                                                      validation_data=genMouthImages(valDirs, batchSize=batchSize, align=valAlign, wordIdx=wordIdx, wordsVocabSize=wordsVocabSize,
                                                                                     useMeanMouthImage=useMeanMouthImage, meanMouthImage=meanMouthImage),
                                                      validation_steps=valSteps, workers=1, initial_epoch=initEpoch)
