import os
import numpy as np
import random as rn
import tensorflow as tf

# with tf.device('/cpu:0'):
from keras.models import Model, Sequential, Input
from keras.layers import Masking, LSTM, Dense, concatenate
from keras.optimizers import Adam

# Import params
from params import *

######################################################################
# LSTM Critic MODEL function with all inputs and 1 output Hidden layer
######################################################################


def LSTM_critic(
        useMask=True,
        hiddenDim=100,
        LSTMactiv='tanh',
        depth=1,
        useLSTMfc=True,
        LSTMfcDim=16,
        LSTMfcActiv='relu',
        oneHotWordDim=wordsVocabSize,
        useOneHotWordFc=False,
        oneHotWordFcDim=16,
        oneHotWordFcActiv='relu',
        outputHDim=64,
        outputActiv='relu',
        lr=5e-4
    ):
    
    # Manual seeds
    os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
    np.random.seed(29)
    rn.seed(29)
    tf.set_random_seed(29)
    
    # Input
    vidInput = Input(shape=(framesPerWord, nOfMouthPixels,))
    
    # Mask
    if useMask:
        LSTMinput = Masking(mask_value=0.)(vidInput)
    else:
        LSTMinput = vidInput
    
    # (Deep) LSTM
    # If depth > 1
    if depth > 1:
        # First layer
        encoded = LSTM(hiddenDim, activation=LSTMactiv,
                       return_sequences=True)(LSTMinput)
        for d in range(depth - 2):
            encoded = LSTM(hiddenDim, activation=LSTMactiv,
                           return_sequences=True)(encoded)
        # Last layer
        encoded = LSTM(hiddenDim, activation=LSTMactiv)(encoded)
    # If depth = 1
    else:
        encoded = LSTM(hiddenDim, activation=LSTMactiv)(LSTMinput)

    # LSTM Fc
    if useLSTMfc:
        vidFeatures = Dense(LSTMfcDim, activation=LSTMfcActiv)(encoded)
    else:
        vidFeatures = encoded

    # Predicted Word input
    oneHotWordInput = Input(shape=(oneHotWordDim,))

    # OHWfc
    if useOneHotWordFc:
        oneHotWordFeatures = Dense(oneHotWordFcDim, activation=oneHotWordFcActiv)(oneHotWordInput)
    else:
        oneHotWordFeatures = oneHotWordInput

    # Full feature
    fullFeature = concatenate([vidFeatures, oneHotWordInput])

    # Output
    y = Dense(outputHDim, activation=outputActiv)(fullFeature)
    myOutput = Dense(1, activation='sigmoid')(y)

    # Model
    criticModel = Model(inputs=[vidInput, oneHotWordInput], outputs=myOutput)

    # lr = 5e-4
    adam = Adam(lr=lr)
    criticModel.compile(optimizer=adam, loss='binary_crossentropy',
                        metrics=['accuracy'])

    criticModel.summary()
    
    fileNamePre ='LSTMCritic-revSeq-Mask-LSTMh' + str(hiddenDim) \
        + '-LSTMactiv' + str(LSTMactiv) + '-depth' + str(depth)
    if useLSTMfc:
        fileNamePre += '-LSTMfc' + str(LSTMfcDim)
    fileNamePre += '-OHWord' + str(oneHotWordDim)
    if useOneHotWordFc:
        fileNamePre += '-OHWordFc' + str(oneHotWordFcDim)
    fileNamePre += '-out' + str(outputHDim) \
        + '-Adam-%1.e' % lr
    print(fileNamePre)
    
    return criticModel, fileNamePre
