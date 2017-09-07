# C3D Critic MODEL function with all inputs and 1 output Hidden layer
def C3DCriticModel(
        layer1Filters=8,
        layer2Filters=16,
        layer3Filters=32,
        fc1Nodes=64,
        vidFeaturesDim=64,
        useWord=False,
        wordDim=1,
        useEncWord=False,
        encWordDim=64,
        useEncWordFc=False,
        encWordFcDim=10,
        useOneHotWord=False,
        oneHotWordDim=52,
        useOneHotWordFc=False,
        oneHotWordFcDim=10,
        usePredWordDis=False,
        predWordDisDim=52,
        outputHDim=64,
        lr=5e-4
    ):
    # Manual seeds
    os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
    np.random.seed(29)
    rn.seed(29)
    tf.set_random_seed(29)
    vidInput = Input(shape=(framesPerWord, nOfMouthPixels,))
    # 0 1 2 3  2 3 4 5  4 5 6 7  6 7 8 9  8 9 10 11  10 11 12 13
    i1 = Lambda(lambda x: x[:, 0:4, :])(vidInput)
    i2 = Lambda(lambda x: x[:, 2:6, :])(vidInput)
    i3 = Lambda(lambda x: x[:, 4:8, :])(vidInput)
    i4 = Lambda(lambda x: x[:, 6:10, :])(vidInput)
    i5 = Lambda(lambda x: x[:, 8:12, :])(vidInput)
    i6 = Lambda(lambda x: x[:, 10:14, :])(vidInput)
    layer1 = Sequential()
    layer1.add(Reshape((4, mouthW, mouthH, 1), input_shape=(4, nOfMouthPixels)))
    layer1.add(Conv3D(layer1Filters, 3, activation='relu'))
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
    layer2.add(Conv3D(layer2Filters, (2, 3, 3), activation='relu',
                      input_shape=(y1.get_shape()[1], y1.get_shape()[2], y1.get_shape()[3], layer1Filters)))
    layer2.add(BatchNormalization())
    layer2.add(MaxPooling3D(pool_size=(1, 2, 2)))
    z1 = layer2(y1)
    z2 = layer2(y2)
    z3 = layer2(y3)
    z = concatenate([z1, z2, z3], axis=1)
    layer3 = Sequential()
    layer3.add(Conv3D(layer3Filters, (3, 3, 3), activation='relu',
                      input_shape=(z.get_shape()[1], z.get_shape()[2], z.get_shape()[3], layer2Filters)))
    layer3.add(BatchNormalization())
    z = layer3(z)
    z = Flatten()(z)
    z = Dense(fc1Nodes, activation='relu')(z)
    vidFeatures = Dense(vidFeaturesDim, activation='relu')(z)
    # Other Inputs
    if useWord:
        wordFeatures = Input(shape=(wordDim,))
        print("...using wordFeatures")
    if useEncWord:
        encWordFeatures = Input(shape=(encWordDim,))
        print("...using encWordFeatures")
    if useEncWordFc:
        encWordFcFeatures = Dense(encWordFcDim, activation='relu')(encWordFeatures)
        print("...using encWordFcFeatures")
    if useOneHotWord:
        oneHotWordFeatures = Input(shape=(oneHotWordDim,))
        print("...using oneHotWordFeatures")
    if useOneHotWordFc:
        oneHotWordFcFeatures = Dense(oneHotWordFcDim, activation='relu')(oneHotWordFeatures)
        print("...using oneHotWordFcFeatures")
    if usePredWordDis:
        predWordDis = Input(shape=(predWordDisDim,))
        print("...using predWordDis")
    # Full feature
    if useWord:
        fullFeature = concatenate([vidFeatures, wordFeatures])
        print("...fullFeature wordFeatures")
    if useEncWord and not useEncWordFc and not useOneHotWord:
        fullFeature = concatenate([vidFeatures, encWordFeatures])
        print("...fullFeature encWordFeatures")
    if useEncWord and useEncWordFc and not useOneHotWord:
        fullFeature = concatenate([vidFeatures, encWordFcFeatures])
        print("...fullFeature encWordFeatures, encWordFcFeatures")
    if useOneHotWord and not useOneHotWordFc and not useEncWord:
        fullFeature = concatenate([vidFeatures, oneHotWordFeatures])
        print("...fullFeature oneHotWordFeatures")
    if useOneHotWord and useOneHotWordFc and not useEncWord:
        fullFeature = concatenate([vidFeatures, oneHotWordFcFeatures])
        print("...fullFeature oneHotWordFeatures, oneHotWordFcFeatures")
    if useEncWord and not useEncWordFc and useOneHotWord and not useOneHotWordFc:
        fullFeature = concatenate([vidFeatures, encWordFeatures, oneHotWordFeatures])
        print("...fullFeature encWordFeatures, oneHotWordFeatures")
    if useEncWord and useEncWordFc and useOneHotWord and useOneHotWordFc:
        fullFeature = concatenate([vidFeatures, encWordFcFeatures, oneHotWordFcFeatures])
        print("...fullFeature encWordFeatures, encWordFcFeatures, oneHotWordFeatures, oneHotWordFcFeatures")
    if usePredWordDis:
        fullFeature = concatenate([vidFeatures, predWordDis])
        print("...fullFeature predWordDis")
    # Output
    y = Dense(outputHDim, activation='relu')(fullFeature)
    print("...y")
    myOutput = Dense(1, activation='sigmoid')(y)
    print("...myOutput")
    # Model
    if useWord:
        criticModel = Model(inputs=[vidInput, wordFeatures], outputs=myOutput)
        print("...model wordFeatures")
    if useEncWord and not useOneHotWord:
        criticModel = Model(inputs=[vidInput, encWordFeatures], outputs=myOutput)
        print("...model encWordFeatures")
    if useOneHotWord and not useEncWord:
        criticModel = Model(inputs=[vidInput, oneHotWordFeatures], outputs=myOutput)
        print("...model oneHotWordFeatures")
    if useEncWord and useOneHotWord:
        criticModel = Model(inputs=[vidInput, encWordFeatures, oneHotWordFeatures], outputs=myOutput)
        print("...model encWordFeatures, oneHotWordFeatures")
    if usePredWordDis:
        criticModel = Model(inputs=[vidInput, predWordDis], outputs=myOutput)
        print("...model predWordDis")
    # lr = 5e-4
    adam = Adam(lr=lr)
    criticModel.compile(optimizer=adam, loss='binary_crossentropy',
                        metrics=['accuracy'])
    filenamePre ='C3DCritic-l1f' + str(layer1Filters) + \
    '-l2f' + str(layer2Filters) + \
    '-l3f' + str(layer3Filters) + \
    '-fc1n' + str(fc1Nodes) + \
    '-vid' + str(vidFeaturesDim)
    if useWord:
        filenamePre += '-word' + str(wordDim)
    if useEncWord:
        filenamePre += '-encWord' + str(encWordDim)
    if useOneHotWord:
        filenamePre += '-OHWord' + str(oneHotWordDim)
    if useOneHotWordFc:
        filenamePre += '-OHWordFc' + str(oneHotWordFcDim)
    if usePredWordDis:
        filenamePre += '-predWordDisDim' + str(predWordDisDim)
    filenamePre += '-out' + str(outputHDim) + \
        '-Adam-%1.e' % lr + \
        '-GRIDcorpus-s'
    print(filenamePre)
    return criticModel, filenamePre

