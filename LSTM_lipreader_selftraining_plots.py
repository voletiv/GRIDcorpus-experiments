# SELF-TRAINING with FINE-TUNING or REMODELLING, LR or LR+Critic

a = np.load("/media/voletiv/01D2BF774AC76280/GRIDcorpusResults/SELF-TRAINING/5-20pc-finetuning-LR/LSTMLipReader-revSeq-Mask-LSTMh256-tanh-depth2-enc64-relu-adam-1e-03-tMouth-valMouth-NOmeanSub-GRIDcorpus-s0107-10-si-s1314-20PercentSelfLearning.npz")
b = np.load("/media/voletiv/01D2BF774AC76280/GRIDcorpusResults/SELF-TRAINING/6-20pc-remodelling-LR/LSTMLipReader-revSeq-Mask-LSTMh256-tanh-depth2-enc64-relu-adam-1e-03-tMouth-valMouth-NOmeanSub-GRIDcorpus-s0107-10-si-s1314-20PercentSelfTraining.npz")
c = np.load("/media/voletiv/01D2BF774AC76280/GRIDcorpusResults/SELF-TRAINING/7-20pc-finetuning-critic/LSTMLipReader-revSeq-Mask-LSTMh256-tanh-depth2-enc64-relu-adam-1e-03-tMouth-valMouth-NOmeanSub-GRIDcorpus-s0107-10-si-s1314-20PercentSelfTraining.npz")
d = np.load("/media/voletiv/01D2BF774AC76280/GRIDcorpusResults/SELF-TRAINING/8-20pc-remodelling-critic/LSTMLipReader-revSeq-Mask-LSTMh256-tanh-depth2-enc64-relu-adam-1e-03-tMouth-valMouth-NOmeanSub-GRIDcorpus-s0107-10-si-s1314-20PercentSelfTraining.npz")

LRftPc = a["percentageOfLabelledData"]
LRrmPc = b["percentageOfLabelledData"]
CftPc = c["percentageOfLabelledData"]
CrmPc = d["percentageOfLabelledData"]

LRftAaPc = a["apparentLabelledTrainAccuraciesThruPcOfLabelledData"]
LRrmAaPc = b["apparentLabelledTrainAccuraciesThruPcOfLabelledData"]
CftAaPc = c["apparentLabelledTrainAccuraciesThruPcOfLabelledData"]
CrmAaPc = d["apparentLabelledTrainAccuraciesThruPcOfLabelledData"]

LRftTaPc = a["trueLabelledTrainAccuraciesThruPcOfLabelledData"]
LRrmTaPc = b["trueLabelledTrainAccuraciesThruPcOfLabelledData"]
CftTaPc = c["trueLabelledTrainAccuraciesThruPcOfLabelledData"]
CrmTaPc = d["trueLabelledTrainAccuraciesThruPcOfLabelledData"]

LRftVaPc = a["valAccuraciesThruPcOfLabelledData"]
LRrmVaPc = b["valAccuraciesThruPcOfLabelledData"]
CftVaPc = c["valAccuraciesThruPcOfLabelledData"]
CrmVaPc = d["valAccuraciesThruPcOfLabelledData"]

LRftSiaPc = a["siAccuraciesThruPcOfLabelledData"]
LRrmSiaPc = b["siAccuraciesThruPcOfLabelledData"]
CftSiaPc = c["siAccuraciesThruPcOfLabelledData"]
CrmSiaPc = d["siAccuraciesThruPcOfLabelledData"]


plt.plot(LRftPc, LRftAaPc, label="LR-ft-Apparent-Acc", color='b', marker='.')
plt.plot(LRftPc, LRftTaPc, label="LR-ft-True-Acc", color='b', marker='o')
plt.plot(LRftPc, LRftVaPc, label="LR-ft-Val-Acc", color='b', marker='D')
plt.plot(LRftPc, LRftSiaPc, label="LR-ft-SI-Acc", color='b', marker='+')

plt.plot(LRrmPc, LRrmAaPc, label="LR-rm-Apparent-Acc", color='g', marker='.')
plt.plot(LRrmPc, LRrmTaPc, label="LR-rm-True-Acc", color='g', marker='o')
plt.plot(LRrmPc, LRrmVaPc, label="LR-rm-Val-Acc", color='g', marker='D')
plt.plot(LRrmPc, LRrmSiaPc, label="LR-rm-SI-Acc", color='g', marker='+')

plt.plot(CftPc, CftAaPc, label="LR+C-ft-Apparent-Acc", color='r', marker='.')
plt.plot(CftPc, CftTaPc, label="LR+C-ft-True-Acc", color='r', marker='o')
plt.plot(CftPc, CftVaPc, label="LR+C-ft-Val-Acc", color='r', marker='D')
plt.plot(CftPc, CftSiaPc, label="LR+C-ft-SI-Acc", color='r', marker='+')

plt.plot(CrmPc, CrmAaPc, label="LR+C-rm-Apparent-Acc", color='k', marker='.')
plt.plot(CrmPc, CrmTaPc, label="LR+C-rm-True-Acc", color='k', marker='o')
plt.plot(CrmPc, CrmVaPc, label="LR+C-rm-Val-Acc", color='k', marker='D')
plt.plot(CrmPc, CrmSiaPc, label="LR+C-rm-SI-Acc", color='k', marker='+')

plt.xlim([0, 100])
plt.xticks(np.arange(0, 101, 10), fontsize=8, rotation=90)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.gca().yaxis.grid(True, alpha=0.5)
plt.xlabel('% of labelled data')
plt.ylabel('accuracy')
leg = plt.legend(loc='best', fontsize=14, fancybox=True)
leg.get_frame().set_alpha(0.3)
plt.title("Accuracies with semi-supervised learning")
plt.tight_layout()
plt.show()


# SELF-TRAINING with 10%, 20% with LR or LR+Critic (only finetuning)

e = np.load("/media/voletiv/01D2BF774AC76280/GRIDcorpusResults/SELF-TRAINING/10-10pc-finetuning-LR/LSTMLipReader-revSeq-Mask-LSTMh256-tanh-depth2-enc64-relu-adam-1e-03-tMouth-valMouth-NOmeanSub-GRIDcorpus-s0107-10-si-s1314-10PercentSelfTraining.npz")
f = np.load("/media/voletiv/01D2BF774AC76280/GRIDcorpusResults/SELF-TRAINING/9-10pc-finetuning-critic/LSTMLipReader-revSeq-Mask-LSTMh256-tanh-depth2-enc64-relu-adam-1e-03-tMouth-valMouth-NOmeanSub-GRIDcorpus-s0107-10-si-s1314-10PercentSelfTraining.npz")
g = np.load("/media/voletiv/01D2BF774AC76280/GRIDcorpusResults/SELF-TRAINING/5-20pc-finetuning-LR/LSTMLipReader-revSeq-Mask-LSTMh256-tanh-depth2-enc64-relu-adam-1e-03-tMouth-valMouth-NOmeanSub-GRIDcorpus-s0107-10-si-s1314-20PercentSelfTraining.npz")
h = np.load("/media/voletiv/01D2BF774AC76280/GRIDcorpusResults/SELF-TRAINING/7-20pc-finetuning-critic/LSTMLipReader-revSeq-Mask-LSTMh256-tanh-depth2-enc64-relu-adam-1e-03-tMouth-valMouth-NOmeanSub-GRIDcorpus-s0107-10-si-s1314-20PercentSelfTraining.npz")

LRft10Pc = e["percentageOfLabelledData"]
CriticFt10pc = f["percentageOfLabelledData"]
LRft20Pc = g["percentageOfLabelledData"]
CriticFt20pc = h["percentageOfLabelledData"]

LRft10PcAa = e["apparentLabelledTrainAccuraciesThruPcOfLabelledData"]
CriticFt10pcAa = f["apparentLabelledTrainAccuraciesThruPcOfLabelledData"]
LRft20PcAa = g["apparentLabelledTrainAccuraciesThruPcOfLabelledData"]
CriticFt20pcAa = h["apparentLabelledTrainAccuraciesThruPcOfLabelledData"]

LRft10PcTa = e["trueLabelledTrainAccuraciesThruPcOfLabelledData"]
# CriticFt10pcTa = f["trueLabelledTrainAccuraciesThruPcOfLabelledData"]
LRft20PcTa = g["trueLabelledTrainAccuraciesThruPcOfLabelledData"]
CriticFt20pcTa = h["trueLabelledTrainAccuraciesThruPcOfLabelledData"]

LRft10PcVa = e["valAccuraciesThruPcOfLabelledData"]
CriticFt10pcVa = f["valAccuraciesThruPcOfLabelledData"]
LRft20PcVa = g["valAccuraciesThruPcOfLabelledData"]
CriticFt20pcVa = h["valAccuraciesThruPcOfLabelledData"]

LRft10PcSia = e["siAccuraciesThruPcOfLabelledData"]
CriticFt10pcSia = f["siAccuraciesThruPcOfLabelledData"]
LRft20PcSia = g["siAccuraciesThruPcOfLabelledData"]
CriticFt20pcSia = h["siAccuraciesThruPcOfLabelledData"]

plt.plot(LRft10Pc, LRft10PcAa, label="LR-ft-10%-Apparent-Acc", color='b', marker='.')
plt.plot(LRft10Pc, LRft10PcTa, label="LR-ft-10%-True-Acc", color='b', marker='o')
plt.plot(LRft10Pc, LRft10PcVa, label="LR-ft-10%-Val-Acc", color='b', marker='D')
plt.plot(LRft10Pc, LRft10PcSia, label="LR-ft-10%-SI-Acc", color='b', marker='+')

plt.plot(CriticFt10pc, CriticFt10pcAa, label="LR+C-ft10%--Apparent-Acc", color='g', marker='.')
# plt.plot(CriticFt10pc, CriticFt10pcTa, label="LR+C-ft-10%-True-Acc", color='g', marker='o')
plt.plot(CriticFt10pc, CriticFt10pcVa, label="LR+C-ft-10%-Val-Acc", color='g', marker='D')
plt.plot(CriticFt10pc, CriticFt10pcSia, label="LR+C-ft-10%-SI-Acc", color='g', marker='+')

plt.plot(LRft20Pc, LRft20PcAa, label="LR-ft-20%-Apparent-Acc", color='r', marker='.')
plt.plot(LRft20Pc, LRft20PcTa, label="LR-ft-20%-True-Acc", color='r', marker='o')
plt.plot(LRft20Pc, LRft20PcVa, label="LR-ft-20%-Val-Acc", color='r', marker='D')
plt.plot(LRft20Pc, LRft20PcSia, label="LR-ft-20%-SI-Acc", color='r', marker='+')

plt.plot(CriticFt20pc, CriticFt20pcAa, label="LR+C-ft-20%-Apparent-Acc", color='k', marker='.')
plt.plot(CriticFt20pc, CriticFt20pcTa, label="LR+C-ft-20%-True-Acc", color='k', marker='o')
plt.plot(CriticFt20pc, CriticFt20pcVa, label="LR+C-ft-20%-Val-Acc", color='k', marker='D')
plt.plot(CriticFt20pc, CriticFt20pcSia, label="LR+C-ft-20%-SI-Acc", color='k', marker='+')

plt.xlim([0, 100])
plt.xticks(np.arange(0, 101, 10), fontsize=8, rotation=90)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.gca().yaxis.grid(True, alpha=0.5)
plt.xlabel('% of labelled data')
plt.ylabel('accuracy')
leg = plt.legend(loc='best', fontsize=14, fancybox=True)
leg.get_frame().set_alpha(0.3)
plt.title("Accuracies with semi-supervised learning")
plt.tight_layout()
plt.show()



# FOR saving those of not saved
tlE = []
taE = []
vlE = []
vaE = []
silE = []
siaE = []
iterNum = -1
for file in sorted(glob.glob(os.path.join(mediaDir, "*"))):
    if 'epoch' in file:
        if int('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[-8][4:]) != iterNum:
            if iterNum != -1:
                tlE.append(tl)
                vlE.append(vl)
                silE.append(sil)
                taE.append(ta)
                vaE.append(va)
                siaE.append(sia)
            iterNum = int('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[-8][4:])
            print(iterNum)
            tl = []
            ta = []
            vl = []
            va = []
            sil = []
            sia = []
        tl.append(float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[-6][2:]))
        ta.append(float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[-5][2:]))
        vl.append(float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[-4][2:]))
        va.append(float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[-3][2:]))
        sil.append(float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[-2][3:]))
        sia.append(float('.'.join(file.split('/')[-1].split('.')[:-1]).split('-')[-1][3:]))
# Save
for i in range(len(tlE)):
    allApparentLabelledTrainLossesThruSelfLearning.append(tlE[i])
    allValLossesThruSelfLearning.append(vlE[i])
    allSiLossesThruSelfLearning.append(silE[i])
    allApparentLabelledTrainAccuraciesThruSelfLearning.append(taE[i])
    allValAccuraciesThruSelfLearning.append(vaE[i])
    allSiAccuraciesThruSelfLearning.append(siaE[i])
    apparentLabelledTrainLossesThruPcOfLabelledData.append(tlE[i][-1])
    valLossesThruPcOfLabelledData.append(vlE[i][-1])
    siLossesThruPcOfLabelledData.append(silE[i][-1])
    apparentLabelledTrainAccuraciesThruPcOfLabelledData.append(taE[i][-1])
    valAccuraciesThruPcOfLabelledData.append(vaE[i][-1])
    siAccuraciesThruPcOfLabelledData.append(siaE[i][-1])

