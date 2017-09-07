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
# excluding 'sil' and 'sp', +1 for padding
wordsVocabSize = (nOfUniqueWords - 2) + 1

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
