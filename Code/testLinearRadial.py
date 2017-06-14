import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
import numpy.matlib
import scipy.io as sio
import os
import cv2
import matplotlib.pyplot as plt
import dlib
import requests
from PIL import Image
from imutils import face_utils
import matplotlib.path as mpltPath
import csv
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
import threading
from printclass import *

fullpathprocessed = '/Users/clmeiste/TimAikenDocs/SeniorProject/Database/Processed/featurevectors/'
fullpathSVM = '/Users/clmeiste/TimAikenDocs/SeniorProject/Classifiers/SVM/'
    
qualityBad = fullpathprocessed + 'qualityBad.bin'
qualityOK = fullpathprocessed + 'qualityOk.bin'
qualityBadData = fullpathprocessed + 'qualityBad.csv'
qualityOKData = fullpathprocessed + 'qualityOK.csv'
qualitySVMradial = fullpathSVM + 'qualityRBF.xml'
qualitySVMlinear = fullpathSVM + 'qualityLinear.xml'
    
contrastBad = fullpathprocessed + 'constrastBad.bin'
contrastOK = fullpathprocessed + 'constrastOK.bin'
contrastBadData = fullpathprocessed + 'constrastBad.csv'
contrastOKData = fullpathprocessed + 'constrastOK.csv'
contrastSVMradial = fullpathSVM + 'contrastRBF.xml'
contrastSVMlinear = fullpathSVM + 'contrastLinear.xml'
    
darkBad = fullpathprocessed + 'darkBad.bin'
darkOK = fullpathprocessed + 'darkOK.bin'
darkBadData = fullpathprocessed + 'darkBad.csv'
darkOKData = fullpathprocessed + 'darkOK.csv'
darkSVMradial = fullpathSVM + 'darkRBF.xml'
darkSVMlinear = fullpathSVM + 'darkLinear.xml'
    
eyesBad = fullpathprocessed + 'eyesBad.bin'
eyesOK = fullpathprocessed + 'eyesOK.bin'
eyesBadData = fullpathprocessed + 'eyesBad.csv'
eyesOKData = fullpathprocessed + 'eyesOK.csv'
eyesSVMradial = fullpathSVM + 'eyesRBF.xml'
eyesSVMlinear = fullpathSVM + 'eyesLinear.xml'
    
flatBad = fullpathprocessed + 'flatBad.bin'
flatOK = fullpathprocessed + 'flatOK.bin'
flatBadData = fullpathprocessed + 'flatBad.csv'
flatOKData = fullpathprocessed + 'flatOK.csv'
flatSVMradial = fullpathSVM + 'flatRBF.xml'
flatSVMlinear = fullpathSVM + 'flatLinear.xml'

def testSVM(badfile, okfile, badDataFile, okDataFile, saveLocation):
    
    svm = cv2.ml.SVM_load(saveLocation)
    
    badData = np.fromfile(badfile).reshape((-1, 8))
    badDataTesting = badData[int(badData.shape[0] * 0.8):,:]
    badClassTesting = np.ones(badDataTesting.shape[0])
    
    okData = np.fromfile(okfile).reshape((-1,8))
    okDataTesting = okData[int(okData.shape[0] * 0.8):,:]
    okClassTesting = np.zeros(okDataTesting.shape[0])
    
    testingData = np.float32(np.vstack((badDataTesting, okDataTesting)))
    truth = np.int32(np.concatenate((badClassTesting, okClassTesting)))
    
    output = svm.predict(testingData)[1].ravel()
    
    correct = np.sum(output == truth)
    total = output.size
    
    correctBad = np.sum(output[:badClassTesting.size] == truth[:badClassTesting.size])
    incorrectBad = np.sum(output[:badClassTesting.size] != truth[:badClassTesting.size])
    truePositive = np.float32(correctBad) / badClassTesting.size
    falseNegative = np.float32(incorrectBad) / badClassTesting.size
    
    correctOK = np.sum(output[badClassTesting.size:] == truth[badClassTesting.size:])
    incorrectOK = np.sum(output[badClassTesting.size:] != truth[badClassTesting.size:])
    trueNegative = np.float32(correctOK) / (output.size - badClassTesting.size)
    falsePositive = np.float32(incorrectOK) / (output.size - badClassTesting.size)
    
    score = np.float32(correct) / total
    
    with open(okDataFile, 'rb') as okCSV, open(badDataFile, 'rb') as badCSV:
        
        readerOK = np.asarray(list(csv.reader(okCSV)))[int(okData.shape[0] * 0.8):,0]
        readerBad = np.asarray(list(csv.reader(badCSV)))[int(badData.shape[0] * 0.8):,0]
        
        filesIncorrectBad = readerBad[output[:badClassTesting.size] != truth[:badClassTesting.size]]
        filesIncorrectOK = readerOK[output[badClassTesting.size:] != truth[badClassTesting.size:]]
        
        print(saveLocation)
        
        print("False negative")
        for f in filesIncorrectBad[0:5]:
            print f
        
        print("False positive")
        for f in filesIncorrectOK[0:5]:
            print f
    
    return score, truePositive, falseNegative, falsePositive, trueNegative

def testLinearRadial():
    
    eS, eTP, eFN, eFP, eTN = testSVM(eyesBad, eyesOK, eyesBadData, eyesOKData, eyesSVMlinear)
    fS, fTP, fFN, fFP, fTN = testSVM(flatBad, flatOK, flatBadData, flatOKData, flatSVMlinear)
    dS, dTP, dFN, dFP, dTN = testSVM(darkBad, darkOK, darkBadData, darkOKData, darkSVMlinear)
    cS, cTP, cFN, cFP, cTN = testSVM(contrastBad, contrastOK, contrastBadData, contrastOKData, contrastSVMlinear)
    qS, qTP, qFN, qFP, qTN = testSVM(qualityBad, qualityOK, qualityBadData, qualityOKData, qualitySVMlinear)
    printClassifier("Linear Eyes too dark", eS, eTP, eFN, eFP, eTN)
    printClassifier("Linear Face too flat", fS, fTP, fFN, fFP, fTN)
    printClassifier("Linear Face too dark", dS, dTP, dFN, dFP, dTN)
    printClassifier("Linear Face too contrasty", cS, cTP, cFN, cFP, cTN)
    printClassifier("Linear Quality of photo", qS, qTP, qFN, qFP, qTN)
    
    eS, eTP, eFN, eFP, eTN = testSVM(eyesBad, eyesOK, eyesBadData, eyesOKData, eyesSVMradial)
    fS, fTP, fFN, fFP, fTN = testSVM(flatBad, flatOK, flatBadData, flatOKData, flatSVMradial)
    dS, dTP, dFN, dFP, dTN = testSVM(darkBad, darkOK, darkBadData, darkOKData, darkSVMradial)
    cS, cTP, cFN, cFP, cTN = testSVM(contrastBad, contrastOK, contrastBadData, contrastOKData, contrastSVMradial)
    qS, qTP, qFN, qFP, qTN = testSVM(qualityBad, qualityOK, qualityBadData, qualityOKData, qualitySVMradial)
    printClassifier("RBF Eyes too dark", eS, eTP, eFN, eFP, eTN)
    printClassifier("RBF Face too flat", fS, fTP, fFN, fFP, fTN)
    printClassifier("RBF Face too dark", dS, dTP, dFN, dFP, dTN)
    printClassifier("RBF Face too contrasty", cS, cTP, cFN, cFP, cTN)
    printClassifier("RBF Quality of photo", qS, qTP, qFN, qFP, qTN)
    