import numpy as np
import os

#############################################################
# PARAMS TO BE SET
#############################################################

rootDir = '/home/voletiv/Datasets/GRIDcorpus'
saveDir = rootDir
# rootDir = '/Neutron6/voleti.vikram/GRIDcorpus'
# # saveDir = '/Neutron6/voleti.vikram/GRIDcorpusResults/1-h32-d1-Adam-2e-4-s0107-s0920-s2227'
# saveDir = rootDir


#############################################################
# TRUE PARAMS
#############################################################

framesPerVid = 75
wordsPerVideo = 6
framesPerWord = 14
nOfMouthPixels = 1600
mouthW = 40
mouthH = 40
nOfUniqueWords = 53     # including silent and short pause
# excluding 'sil' and 'sp'
# wordsVocabSize = (nOfUniqueWords - 2) + 1
wordsVocabSize = nOfUniqueWords - 2

# # Unique Words Idx
# uniqueWordsFile = os.path.join('uniqueWords.npy')
# uniqueWords = np.load(uniqueWordsFile)
# # num of unique words, including 'sil' and 'sp'
# nOfUniqueWords = len(uniqueWords)
# # Remove silent
# uniqueWords = np.delete(
#     uniqueWords, np.argwhere(uniqueWords == 'sil'))
# # Remove short pauses
# uniqueWords = np.delete(
#     uniqueWords, np.argwhere(uniqueWords == 'sp'))
# # Vocabulary size
# # excluding 'sil' and 'sp', +1 for padding
# wordsVocabSize = len(uniqueWords) + 1
# # Word indices
# wordIdx = {}
# # Start word indices from 1 (0 for padding)
# for i, word in enumerate(uniqueWords):
#     wordIdx[word] = i+1

# np.save("wordIdx", wordIdx)

currDir = os.path.dirname(os.path.realpath(__file__))

# wordIdx dictionary
wordIdx = np.load(os.path.join(currDir, "wordIdx.npy")).item()
