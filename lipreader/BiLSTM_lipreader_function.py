import numpy as np
import random as rn
import os
import tensorflow as tf

# with tf.device('/cpu:0'):
from keras.models import Model, Sequential
from keras.layers import Input, Activation
from keras.layers import Masking, LSTM, Dense
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam

#################################################################
# IMPORT
#################################################################

from params import *


#################################################################
# LSTM LipReader MODEL with Encoding
#################################################################


def BiLSTM_lipreader(hiddenDim=100,
                    LSTMactiv='tanh',
                    depth=2,
                    encodedDim=64,
                    encodedActiv='relu',
                    lr=1e-3):
    
    os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
    np.random.seed(29)
    rn.seed(29)
    tf.set_random_seed(29)
    
    # Input
    myInput = Input(shape=(framesPerWord, nOfMouthPixels,))
    
    # Mask
    maskOutput = Masking(mask_value=0.)(myInput)
    
    # (Deep) Bidirectional LSTM
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
    fileNamePre = 'BiLSTMLipReader' + '-Mask' + '-BiLSTMh' + str(hiddenDim) \
        + '-LSTMactiv' + LSTMactiv + '-BiLSTMdepth' + str(depth) \
        + '-enc' + str(encodedDim) + '-encodedActiv' + encodedActiv \
        + '-Adam-%1.e' % lr + '-GRIDcorpus-s0107-09-tMouth-valMouth-NOmeanSub'
    return BiLSTMLipReaderModel, BiLSTMEncoder, fileNamePre
