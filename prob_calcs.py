import numpy as np
import os

# # BAYES
# P(A and B) = P(A | B) * P(B) = P(B | A) * P(A)
# P(B | A) = P(A | B) * P(B) / P(A)

# VAL DATA
mediaDir = '/media/voletiv/01D2BF774AC76280/GRIDcorpusResults/03-C3DCritic-l1f4-l2f4-l3f4-fc1n4-vid16-OHWord51-OHWordFc16-out64-Adam-1e-03-GRIDcorpus-s0107-09-epoch016-tl0.4776-ta0.7657-vl0.6650-va0.6450-sil0.6415-sia0.6669'
valResults = np.load(os.path.join(mediaDir, 'valDataResults.npz'))

# THRESHOLDS
thresholds = np.arange(0, 1.001, 0.05)

# PROBS

# True Positives = num(C+ | LR+)
valTPPerThresh = valResults["valTPPerThresh"]

# False Negatives = num(C- | LR+)
valFNPerThresh = valResults["valFNPerThresh"]

# False Positives = num(C+ | LR-)
valFPPerThresh = valResults["valFPPerThresh"]

# True Negatives = num(C- | LR-)
valTNPerThresh = valResults["valTNPerThresh"]

# P(LR+) = .85
LRp = ((valTPPerThresh + valFNPerThresh)/(valTPPerThresh + valFNPerThresh + valFPPerThresh + valTNPerThresh))[0]

# P(LR-) = .15
LRn = 1 - LRp

# TPR = Sensitivity = P(C+ | LR+) = TP / (TP+FN) = .63
# TPR = valResults["valSensitivityPerThresh"]
TPR = valTPPerThresh / (valTPPerThresh + valFNPerThresh)

# FPR = Specificity = P(C+ | LR-) = FP / (FP + TN) = .24
# FPR = valResults["valSpecificityPerThresh"]
FPR = valFPPerThresh / (valFPPerThresh + valTNPerThresh)

# P(FN) = P(C- | LR+) = 1 - TPR = 0.37
FNR = 1 - TPR

# P(TN) = P(C- | LR-) = 1 - FPR = 0.76
TNR = 1 - FPR


# TO FIND PROB OF LR+, given C+: P(LR+ | C+)

# P(LR+ | C+) = P(C+ | LR+) * P(LR+) / P(C+)
# P(C+) = P(LR+ and C+) + P(LR- and C+) = P(C+ | LR+)*P(LR+) + P(C+ | LR-)*P(LR-) =  .63*.85 + .24*.15 = .57
Cp = TPR*LRp + FPR*LRn

# Thus,
# P(LR+ | C+) = P(C+ | LR+) * P(LR+) / P(C+) = .63 * .85 / .57 = .94
LRpGivenCp = TPR * LRp / (Cp + 1e-15)


# TO FIND PROB OF LR-, given C-: P(LR+ | C+): P(LR- | C-)
# P(LR- | C-) = P(C- | LR-) * P(LR-) / P(C-)
# P(C-) = P(C- | LR+)*P(LR+) + P(C- | LR-)*P(LR-) = .37*.85 + .76*.15 = .43
Cn = FNR*LRp + TNR*LRn

# Thus,
# P(LR- | C-) = P(C- | LR-) * P(LR-) / P(C-) = .76 * .15 / .43 = .26
LRnGivenCn = TNR * LRn / (Cn + 1e-15)


# WHAT IF LIPREADER IS BAD
LRp = .5
LRn = 1 - LRp


