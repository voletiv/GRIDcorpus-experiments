import tensorflow as tf
# with tf.device('/cpu:0'):
from keras.models import Model, Sequential
from keras.layers import Input, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K

from keras.layers import Masking, LSTM, Dense, concatenate
from keras.layers.wrappers import Bidirectional

#############################################################
# IMPORT PARAMS
#############################################################

from params import *


#############################################################
# GEN MOUTH IMAGES
#############################################################

from gen_mouth_images import *
from load_mouth_images import *


#############################################################
# PARAMS TO BE SET
#############################################################

rootDir = '/home/voletiv/Datasets/GRIDcorpus'
saveDir = rootDir
# rootDir = '/Neutron6/voleti.vikram/GRIDcorpus'
# saveDir = '/Neutron6/voleti.vikram/GRIDcorpusResults/1-h32-d1-Adam-2e-4-s0107-s0920-s2227'

trainAlign = False
valAlign = False
useMeanMouthImage = False


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
siList = [10, 11, 12, 13, 14]
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


#################################################################
# LSTM LipReader MODEL with Encoding
#################################################################

def BiLSTMLipReader(framesPerWord=14,
                    nOfMouthPixels=1600,
                    hiddenDim=100,
                    LSTMactiv='tanh',
                    depth=2,
                    encodedDim=64,
                    encodedActiv='relu',
                    lr=1e-3):
    myInput = Input(shape=(framesPerWord, nOfMouthPixels,))
    maskOutput = Masking(mask_value=0.)(myInput)
    # Bidirectional LSTM
    # If depth > 1
    if depth > 1:
        # First layer
        encoded = Bidirectional(LSTM(hiddenDim, activation=LSTMactiv,
                       return_sequences=True))(maskOutput)
        for d in range(depth - 2):
            encoded = Bidirectional(LSTM(hiddenDim, activation=LSTMactiv,
                           return_sequences=True))(encoded)
        # Last layer
        encoded = Bidirectional(LSTM(hiddenDim, activation=LSTMactiv))(encoded)
    # If depth = 1
    else:
        encoded = Bidirectional(LSTM(hiddenDim, activation=LSTMactiv))(maskOutput)
    # Encoded layer
    encoded = Dense(encodedDim, activation=encodedActiv)(encoded)
    BiLSTMEncoder = Model(inputs=myInput, outputs=encoded)
    # Output
    myWord = Dense(wordsVocabSize, activation='softmax')(encoded)
    BiLSTMLipReaderModel = Model(inputs=myInput, outputs=myWord)
    # Compile
    adam = Adam(lr=lr)
    BiLSTMLipReaderModel.compile(optimizer=adam, loss='categorical_crossentropy',
                               metrics=['accuracy'])
    BiLSTMLipReaderModel.summary()
    # fileNamePre
    fileNamePre = 'BiLSTM' + '-depth' + str(depth) \
        + '-h' + str(hiddenDim) + '-LSTMactiv' + LSTMactiv \
        + '-enc' + str(encodedDim) + '-encodedActiv' + encodedActiv \
        + '-Adam-%1.e' % lr + '-GRIDcorpus-s0107-09-tMouth-valMouth-NOmeanSub'
    return BiLSTMLipReaderModel, BiLSTMEncoder, fileNamePre


hiddenDim = 256
depth = 1
LSTMactiv = 'tanh'
encodedDim = 64
encodedActiv = 'relu'
lr = 1e-3

# Make model
BiLSTMLipReaderModel, BiLSTMEncoder, fileNamePre = BiLSTMLipReader(hiddenDim=hiddenDim,
    depth=depth, LSTMactiv=LSTMactiv, encodedDim=encodedDim, encodedActiv=encodedActiv,
    lr=lr)


#############################################################
# TRAIN LSTM LIPREADER
#############################################################

nEpochs = 100
initEpoch = 0

batchSize = 128     # num of speaker vids
trainSteps = int(len(trainDirs) / batchSize)
valSteps = int(len(valDirs) / batchSize)
siValSteps = int(len(siDirs) / batchSize)


plotColor = 'g'

class CheckpointSIAndPlots(Callback):
    def __init___(self):
        self.plotColor = plotColor
    # On train start
    def on_train_begin(self, logs={}):
        self.losses = []
        self.valLosses = []
        self.siLosses = []
        self.acc = []
        self.valAcc = []
        self.siAcc = []
        # Define epochIndex
        def epochIndex(x):
            x = x.split('/')[-1].split('-')
            return [i for i, word in enumerate(x) if 'epoch' in word][0]
        # Define epochNoInFile
        def epochNoInFile(x):
            epochIdx = epochIndex(x)
            return x.split('/')[-1].split('-')[epochIdx]
        # For all weight files
        for file in sorted(glob.glob(os.path.join(rootDir, "*" + fileNamePre + "*-epoch*")), key=epochNoInFile):
            print(file)
            epochIdx = epochIndex(file)
            self.losses.append(
                float(file.split('/')[-1].split('.')[:-1].split('-')[epochIdx + 1][2:]))
            self.acc.append(
                float(file.split('/')[-1].split('.')[:-1].split('-')[epochIdx + 2][2:]))
            self.valLosses.append(
                float(file.split('/')[-1].split('.')[:-1].split('-')[epochIdx + 3][2:]))
            self.valAcc.append(
                float(file.split('/')[-1].split('.')[:-1].split('-')[epochIdx + 4][2:]))
            self.siLosses.append(
                float(file.split('/')[-1].split('.')[:-1].split('-')[epochIdx + 5][3:]))
            self.siAcc.append(
                float(file.split('/')[-1].split('.')[:-1].split('-')[epochIdx + 6][3:]))
    # At every epoch
    def on_epoch_end(self, epoch, logs={}):
        tl = logs.get('loss')
        ta = logs.get('acc')
        vl = logs.get('val_loss')
        va = logs.get('val_acc')
        # Speaker-Independent
        print("Calculating speaker-independent loss and acc...")
        [sil, sia] = BiLSTMLipReaderModel.evaluate_generator(gen_mouth_images(siDirs, batchSize=batchSize, shuffle=False), siValSteps)
        # Checkpoint
        filepath = os.path.join(saveDir,
                                fileNamePre + "-epoch{0:03d}-tl{1:.4f}-ta{2:.4f}-vl{3:.4f}-va{4:.4f}-sil{5:.4f}-sia{6:.4f}.hdf5".format(epoch, tl, ta, vl, va, sil, sia))
        checkpoint = ModelCheckpoint(
            filepath, verbose=1, save_best_only=False, save_weights_only=True, period=1)
        print("Saving plots for epoch " + str(epoch))
        self.losses.append(tl)
        self.valLosses.append(vl)
        self.siLosses.append(sil)
        self.acc.append(ta)
        self.valAcc.append(va)
        self.siAcc.append(sia)
        plt.plot(self.losses, label='trainingLoss', color=self.plotColor, linestyle='-.')
        plt.plot(self.valLosses, label='valLoss', color=self.plotColor, linestyle='-')
        plt.plot(self.siLosses, label='siLoss', color=self.plotColor, linestyle='--')
        leg = plt.legend(loc='best', fontsize=11, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig(os.path.join(saveDir, fileNamePre + "-plotLosses.png"))
        plt.close()
        plt.plot(self.acc, label='trainingAcc', color=self.plotColor, linestyle='-.')
        plt.plot(self.valAcc, label='valAcc', color=self.plotColor, linestyle='-')
        plt.plot(self.siAcc, label='siAcc', color=self.plotColor, linestyle='--')
        leg = plt.legend(loc='best', fontsize=11, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
        plt.gca().yaxis.grid(True)
        plt.savefig(os.path.join(saveDir, fileNamePre + "-plotAcc.png"))
        plt.close()

checkpointSIAndPlots = CheckpointSIAndPlots()

# # Load previous lipReaderModel
# mediaDir = '/media/voletiv/01D2BF774AC76280/GRIDcorpusResults/1-SimpleSeq2Seq-h256-depth2-Adam-5e-04-GRIDcorpus-s0107-s0909-meanSub-tAlign-vAlign'
# LSTMLipReaderModel.load_weights(os.path.join(
#     saveDir, "SimpleSeq2Seq-h256-depth2-Adam-5e-04-GRIDcorpus-s0107-s0909-tAlign-vAlign-meanSub-epoch059-tl0.0256-ta0.9327-vl0.3754-va0.8746.hdf5"))
# initEpoch = 60

# FIT (gen)
BiLSTMLipReaderHistory = BiLSTMLipReaderModel.fit_generator(gen_mouth_images(trainDirs, batchSize=batchSize, shuffle=True, shuffleWords=True),
                                                        steps_per_epoch=trainSteps, epochs=nEpochs, verbose=1, callbacks=[checkpointSIAndPlots],
                                                        validation_data=gen_mouth_images(valDirs, batchSize=batchSize, shuffle=False, shuffleWords=False),
                                                        validation_steps=valSteps, workers=1, initial_epoch=initEpoch)

# # FIT (load)
# history = model.fit(trainX, trainY, batch_size=batchSize, epochs=nEpochs, verbose=1, callbacks=[checkpointSIAndPlots],
#     validation_data=(valX, valY), initial_epoch=initEpoch)
