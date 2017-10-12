import numpy as np
import random as rn
import os
import tensorflow as tf

# with tf.device('/cpu:0'):
from keras.models import Model, Sequential
from keras.layers import Input, Activation
from keras.layers import Masking, LSTM, Dense
from keras.optimizers import Adam, RMSprop

#################################################################
# IMPORT
#################################################################

from params import *


#################################################################
# LSTM LipReader MODEL with Encoding
#################################################################


def LSTM_lipreader(wordsVocabSize=wordsVocabSize,
                    useMask=False,
                    hiddenDim=100,
                    LSTMactiv='tanh',
                    depth=2,
                    encodedDim=64,
                    encodedActiv='relu',
                    optimizer='adam',
                    lr=1e-3):
    os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
    np.random.seed(29)
    rn.seed(29)
    tf.set_random_seed(29)
    # Input
    myInput = Input(shape=(framesPerWord, nOfMouthPixels,))
    # Mask
    if useMask:
        LSTMinput = Masking(mask_value=0.)(myInput)
    else:
        LSTMinput = myInput
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
    # Encoder
    fcEncoded = Dense(encodedDim, activation=encodedActiv)(encoded)
    LSTMEncoder = Model(inputs=myInput, outputs=fcEncoded)
    # Output
    myWord = Dense(wordsVocabSize, activation='softmax')(fcEncoded)
    # Model
    LSTMLipReaderModel = Model(inputs=myInput, outputs=myWord)
    # Compile
    if optimizer == 'adam':
        optim = Adam(lr=lr)
    elif optimizer == 'rmsprop':
        optim = RMSprop(lr=lr)
    LSTMLipReaderModel.compile(optimizer=optim, loss='categorical_crossentropy',
                               metrics=['accuracy'])
    LSTMLipReaderModel.summary()
    # fileNamePre
    fileNamePre = 'LSTMLipReader-revSeq-Mask-LSTMh' + str(hiddenDim) \
        + '-' + LSTMactiv + '-depth' + str(depth) \
        + '-enc' + str(encodedDim) + '-' + encodedActiv \
        + '-' + optimizer + '-%1.e' % lr + '-tMouth-valMouth-NOmeanSub'
    print(fileNamePre)
    return LSTMLipReaderModel, LSTMEncoder, fileNamePre
