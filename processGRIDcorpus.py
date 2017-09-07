import cv2
import os
import numpy as np
import glob
import tqdm
import matplotlib
# matplotlib.use('agg')     # Use this for remote terminals
import matplotlib.pyplot as plt
import random as rn

from matplotlib.patches import Rectangle
from imutils.face_utils import FaceAligner, shape_to_np
import dlib


#############################################################
# TRUE PARAMS
#############################################################

from params import *

#############################################################
# DETECT AND SAVE MOUTH REGIONS
#############################################################

detector = dlib.get_frontal_face_detector()

# predictor = dlib.shape_predictor(
#     '/home/voletiv/GitHubRepos/gazr/share/shape_predictor_68_face_landmarks.dat')
# fa = FaceAligner(predictor, desiredFaceWidth=100)

# # Opencv classifier for face detection
# faceCascade = cv2.CascadeClassifier(
#     '/home/voletiv/dev/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')


# Align face, and return just the face scaled to 128 pixels width
def alignFace(frame, x, y, w, h):
    return fa.align(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))


# Convert dlib rectangle to bounding box (x, y, w, h)
def rect_to_bb(rect):
    if isinstance(rect, dlib.rectangle):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
    else:
        x, y, w, h = rect
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


# Find rectangle bounding face
def findFaceRect(frame, mode='dlib'):
    # If mode is opencv
    if mode == 'opencv':
        # Detect (only 1) face = (x, y, w, h)
        faceRect = faceCascade.detectMultiScale(
            frame, scaleFactor=1.1, minNeighbors=5)
        # If a face is found
        if faceRect != ():
            # If more than 1 face is found
            if len(faceRect) > 1:
                n = len(faceRect)
                # print(str(n) + " mouths found")
                widths = []
                for i in range(n):
                    widths.append(faceRect[i][2])
                # Find face with max width
                return faceRect[np.argmax(widths)]
            # Else, if only 1 face is found
            else:
                return faceRect[0]
        # Else if no faceRect is found
        else:
            return ()
    # If mode is dlib
    elif mode == 'dlib':
        # Detect face using dlib detector
        faceRect = detector(frame, 1)
        # If at least 1 face is found
        if len(faceRect) > 0:
            return faceRect[0]
        # If no face is found
        else:
            return ()


def expandFaceRect(faceX, faceY, faceW, faceH, scale=1.5):
    w = int(faceW * scale)
    h = int(faceH * scale)
    x = faceX - int((w - faceW) / 2)
    y = faceY - int((h - faceH) / 2)
    return x, y, w, h


# Gaussian kernel
# https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Find mean pixel value of mouth area in expanded face, assuming faceW = 128
def findMouthMeanInFaceRect(face, wReduceFactor=0.6, wLeftOffsetFactor=0.15, hReduceFactor=0.5, hBottomOffsetFactor=0.15, showACh=False, aChThresh=0.9):
    # Reduce frame width to find mouth in constrained area
    (faceH, faceW, _) = face.shape
    # plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)); plt.show()
    wDelta = wReduceFactor * faceW
    newFaceW = faceW - int(wDelta) + int(wLeftOffsetFactor * faceW)
    newFaceX = int(wDelta / 2)
    hDelta = hReduceFactor * faceH
    newFaceY = int(hReduceFactor * faceH)
    newFaceH = faceH - int(hDelta) - int(hBottomOffsetFactor * faceH)
    # Extract smaller face
    smallFace = np.array(
        face[newFaceY:newFaceY + newFaceH, newFaceX:newFaceX + newFaceW, :])
    # plt.imshow(cv2.cvtColor(smallFace, cv2.COLOR_BGR2RGB)); plt.show()
    # Convert face to LAB, extract A channel
    aCh = cv2.cvtColor(smallFace, cv2.COLOR_BGR2Lab)[:, :, 1]
    (aChW, aChH) = aCh.shape
    # Element-wise multiply with gaussian kernel with center pixel at
    # 30% height, and sigma 500
    gaussKernel = matlab_style_gauss2D(
        (aChW, 2 * 0.7 * aChH), sigma=500)[:, -aChH:]
    aCh = np.multiply(aCh, gaussKernel)
    # Rescale to [0, 1]
    aCh = (aCh - aCh.min()) / (aCh.max() - aCh.min())
    # Find mean of those pixels > 0.9
    # plt.imshow(aCh > 0.9, cmap='gray'); plt.show()
    if showACh:
        plt.imshow(aCh, cmap='gray'); plt.show()
    # Here, the X & Y axes of np array are the Y & X of Rectangle respectively
    mouthY, mouthX = np.where(aCh > aChThresh)
    mouthXMean = mouthX.mean()
    mouthYMean = mouthY.mean()
    # plt.imshow(cv2.cvtColor(smallFace, cv2.COLOR_BGR2RGB)); ca = plt.gca(); ca.add_patch(Rectangle((mouthXMean - 2, mouthYMean - 2), 4, 4, edgecolor='g', facecolor='g')); plt.show()
    # plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)); ca = plt.gca(); ca.add_patch(Rectangle((newFaceX + mouthXMean - 2, newFaceY + mouthYMean - 2), 4, 4, edgecolor='g', facecolor='g')); plt.show()
    return (newFaceX + mouthXMean, newFaceY + mouthYMean, aCh)


# Extract mouth image
def extractMouthImage(frameFile, align=False, showACh=False, aChThresh=0.9, mode='dlib'):
    # Read image (as BGR)
    frame = cv2.imread(frameFile)
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); plt.show()
    # Find rectangle bounding face
    faceRect = findFaceRect(frame, mode=mode)
    if faceRect != ():
        faceX, faceY, faceW, faceH = rect_to_bb(faceRect)
    # If a face is not found, return empty
    else:
        return faceRect
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); ca = plt.gca(); ca.add_patch(Rectangle((faceX, faceY), faceW, faceH, edgecolor='r', fill=False)); plt.show()
    # If asked to align faces
    if align:
        # Align face, and return just the face scaled to 100
        # pixels width
        face = alignFace(frame, faceX, faceY, faceW, faceH)
    # If no need to align
    else:
        # Extract 1.5 times face frame
        if mode == 'opencv':
            (x, y, w, h) = expandFaceRect(
                faceX, faceY, faceW, faceH, scale=1.5)
        # Else if dlib
        elif mode == 'dlib':
            (x, y, w, h) = expandFaceRect(
                faceX, faceY, faceW, faceH, scale=1.7)
        # If x < 0
        if x < 0:
            w -= (0 - x)
            x = 0
        # If y < 0
        if y < 0:
            h -= (0 - y)
            y = 0
        # If y + h exceeds frame height
        if y + h > frame.shape[0]:
            h = frame.shape[0] - y
        # If x + w exceeds frame width
        if x + w > frame.shape[1]:
            w = frame.shape[1] - x
        # plt.imshow(cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)); plt.show()
        # Return just the face scaled to 128 pixels width
        face = cv2.resize(
            np.array(frame[y:y + h, x:x + w]), (128, int(h * 128. / w)))
    # plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)); plt.show()
    # Find mean pixel value of mouth area - pixel value is
    # plottable on face, not frame
    mouthXMean, mouthYMean, aCh = findMouthMeanInFaceRect(
        face, wReduceFactor=0.5, hReduceFactor=0.5, hBottomOffsetFactor=0.15, showACh=showACh, aChThresh=aChThresh)
    # plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)); ca = plt.gca(); ca.add_patch(Rectangle((mouthXMean-2, mouthYMean-2), 4, 4, edgecolor='g', facecolor='g')); plt.show()
    # plt.imshow(aCh, cmap='gray'); plt.show()
    # To make mouth mean plottable on frame
    frameMouthYMean = mouthXMean + faceY
    frameMouthXMean = mouthYMean + faceX
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); ca = plt.gca(); ca.add_patch(Rectangle((frameMouthXMean-2, frameMouthYMean-2), 4, 4, edgecolor='g', facecolor='g')); plt.show()
    # In case mouthYMean cannot cover 40 pixels of mouth around it
    if (int(mouthYMean) + 20 > face.shape[0]):
        mouthYMean = face.shape[0] - 20
    # plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)); ca = plt.gca(); ca.add_patch(Rectangle((mouthXMean-2, mouthYMean-2), 4, 4, edgecolor='g', facecolor='g')); plt.show()
    # Extract mouth as a grayscale 40x40 region around the mean
    mouth = cv2.cvtColor(
        face[int(mouthYMean) - 20:int(mouthYMean) + 20, int(mouthXMean) - 20:int(mouthXMean) + 20, :], cv2.COLOR_BGR2GRAY)
    # plt.imshow(mouth, cmap='gray'); plt.show()
    # Minimize contrast
    mouth = (mouth - mouth.min()) / (mouth.max() - mouth.min())
    return mouth, face, aCh


# Extract And Save Mouth Images
align, key, startDir, showACh, aChThresh, mode = False, False, 's25', False, 0.94, 'dlib'


def extractAndSaveMouthImages(rootDir, align=False, key=False, startDir='s02/pbwxzs', showACh=False, aChThresh=0.94, mode='dlib'):
    GRIDcorpusDirs = sorted(glob.glob(os.path.join(rootDir, '*/')))
    # For each speaker
    for speakerDir in tqdm.tqdm(GRIDcorpusDirs):
        print(speakerDir)
        # speakerDirs = glob.glob(speaker + '*/')
        speakerVids = sorted(glob.glob(os.path.join(speakerDir, '*/')))
        # For each video
        for videoDir in tqdm.tqdm(speakerVids):
            # Current directory name
            if startDir in videoDir:
                key = True
            # If key
            if key:
                # if len(glob.glob(currDir + '/*Frame*.jpg')) == 75 and
                # len(glob.glob(currDir + '/*Mouth*.jpg')) < 75:
                if len(glob.glob(os.path.join(videoDir, '*Frame*.jpg'))) == 75:
                    print(video)
                    # For each frame
                    for count, frameFile in enumerate(tqdm.tqdm(sorted(glob.glob(os.path.join(videoDir, '*Frame*.jpg'))))):
                        # Extract the mouth
                        try:
                            mouth, face, aCh = extractMouthImage(
                                frameFile, align=align, showACh=showACh, aChThresh=aChThresh, mode=mode)
                            # plt.imshow(mouth, cmap='gray'); plt.show()
                            # plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), cmap='gray'); plt.show()
                            # plt.imshow(aCh, cmap='gray'); plt.show()
                            # If a mouth has been found
                            if mouth != ():
                                # Convert mouth to uint8
                                mouthUint8 = (mouth * 255).astype(np.uint8)
                                # Image name
                                if align:
                                    # Write the name
                                    name = '/'.join(frameFile.split('/')[:-1]) + '/' + \
                                        frameFile.split('/')[-1][:-11] + \
                                        "Aligned%02d.jpg" % count
                                # If not aligned
                                else:
                                    # Write the name
                                    name = '/'.join(frameFile.split('/')[:-1]) + '/' + \
                                        frameFile.split('/')[-1][:-11] + \
                                        "Mouth%02d.jpg" % count
                                # imwrite
                                isWritten = cv2.imwrite(name, mouthUint8)
                        # But if it doesn't work
                        except KeyboardInterrupt:
                            print("Ctrl+C was pressed!")
                            # return
                        except:
                            print(frameFile)

# # Check if all have 75 Mouth images
# for directory in tqdm.tqdm(sorted(glob.glob(os.path.join(rootDir, 's15/*/')))):
#     # If num of mouth images != 75
#     if len(glob.glob(os.path.join(directory, '*Mouth*'))) != 75:
#         print(directory)

# SHORT VERSION:
# for speakerDir in tqdm.tqdm(GRIDcorpusDirs):
speakerVids = sorted(glob.glob(os.path.join(speakerDir, '*/')))
for video in tqdm.tqdm(speakerVids):
         for count, frameFile in enumerate(tqdm.tqdm(sorted(glob.glob(os.path.join(video, '*Frame*.jpg'))))):
            print(frameFile)
            mouth, face, aCh = extractMouthImage(frameFile, align=align, showACh=showACh, aChThresh=aChThresh, mode=mode)
            if mouth != ():
                 mouthUint8 = (mouth * 255).astype(np.uint8)
                 name = '/'.join(frameFile.split('/')[:-1]) + '/' + frameFile.split('/')[-1][:-11] + "Mouth%02d.jpg" % count
                 isWritten = cv2.imwrite(name, mouthUint8)



#############################################################
# SAVE NPY
#############################################################

nOfFrames = 75

# Read ALL words
for file in glob.glob(os.path.join(rootDir, '*/*.align')):
    with open(file) as f:
        for line in f:
            words.append(line[:-1].split(' ')[-1])


# Find the unique set of words
uniqueWords = np.unique(words)
nOfUniqueWords = len(uniqueWords)

# Make dictionaries
for i, word in enumerate(uniqueWords):
    word2numDict[word] = i
    num2wordDict[i] = word

categoricalEncoding = np_utils.to_categorical(
    list(range(nOfUniqueWords)), nOfUniqueWords)

# Init the y
y = np.zeros((nOfFrames, nOfUniqueWords))

GRIDcorpusDirs = sorted(glob.glob(os.path.join(rootDir, '*/')))
# For each speaker
for speakerDir in tqdm.tqdm(GRIDcorpusDirs):
    print(speakerDir)
    # speakerDirs = glob.glob(speaker + '*/')
    speakerVids = sorted(glob.glob(speakerDir + '*.mpg'))
    # For each video
    for video in tqdm.tqdm(speakerVids):
        currDir = video[:-4]
        # if len(glob.glob(currDir + '/*Frame*.jpg')) == 75 and
        # len(glob.glob(currDir + '/*Mouth*.jpg')) == 75:
        if len(glob.glob(currDir + '/*Frame*.jpg')) == 75:
            # Align text file
            alignFile = currDir + ".align"
            # Read the lines
            lines = []
            with open(alignFile, 'r') as f:
                for line in f:
                    # lines.append(line[:-1].split(' '))
                    lineArray = line[:-1].split(' ')
                    fromFrame = int(int(lineArray[0]) / 100)
                    toFrame = int(int(lineArray[1]) / 100)
                    theWord = lineArray[2]
                    y[fromFrame:toFrame + 1] = categoricalEncoding[word2numDict[theWord]]
            np.save(currDir + '/' + currDir[len(speakerDir):] + "Y.npy", y)

#############################################################
# CONVERT VIDEOS TO IMAGES
# SAVE IMAGES IN INDIVIDUAL FOLDERS BELONGING TO EACH VIDEO
#############################################################

# Extract video into frames
def extractVidIntoFrames(video, speakerDir):
    # Capture the video
    vidCap = cv2.VideoCapture(video)
    count = 0
    # Read each frame
    success, frame = vidCap.read()
    if success:
        # If dir does not exist
        if not os.path.isdir(video.split('.')[0]):
            # Make the directory for the video
            os.mkdir(video.split('.')[0])
        # Only if all the frames have not been extracted
        # (even if dir was previously made)
        if len(os.listdir(video.split('.')[0])) < 75:
            # While a new frame can be read
            while success:
                # Write image
                isWritten = cv2.imwrite(
                    video.split('.')[0] + '/' + video.split('.')[0][len(speakerDir)] + "Frame%02d.jpg" % count, frame)
                success, frame = vidCap.read()
                count += 1

# For each speaker
# speakerDir = os.path.join(rootDir, 's28/')
for speakerDir in tqdm.tqdm(sorted(glob.glob(os.path.join(rootDir, '*/')))):
    print(speakerDir)
    # speakerDirs = glob.glob(speaker + '*/')
    speakerVids = sorted(glob.glob(os.path.join(speakerDir, '*.mpg')))
    # For each video
    for video in tqdm.tqdm(speakerVids):
        extractVidIntoFrames(video, speakerDir)

# # # Check if all train have 75 Mouth images
# for directory in tqdm.tqdm(np.append(trainDirs, valDirs)):
# # for directory in tqdm.tqdm(sorted(glob.glob(os.path.join(rootDir, 's18/*/')))):
#     # If num of mouth images != 75
#     if len(glob.glob(os.path.join(directory, '*Frame*'))) != 75:
#         print(directory)

s = 's34/'

speakerDir = os.path.join(rootDir, s)
for video in tqdm.tqdm(sorted(glob.glob(os.path.join(speakerDir, '*.mpg')))):
    extractVidIntoFrames(video, speakerDir)

for directory in tqdm.tqdm(sorted(glob.glob(os.path.join(rootDir, s, '*/')))):
    if len(glob.glob(os.path.join(directory, '*Frame*'))) != 75:
        print(directory)
