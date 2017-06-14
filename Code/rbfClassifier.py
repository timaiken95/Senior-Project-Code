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
from sklearn.cross_validation import train_test_split

fullpathprocessed = '/Users/clmeiste/TimAikenDocs/SeniorProject/Database/Processed/featurevectors/'
fullpathSVM = '/Users/clmeiste/TimAikenDocs/SeniorProject/Classifiers/SVM/'
    
qualityBad = fullpathprocessed + 'qualityBad.bin'
qualityOK = fullpathprocessed + 'qualityOk.bin'
qualitySVMradial = fullpathSVM + 'qualityRBF.xml'
qualitySVMlinear = fullpathSVM + 'qualityLinear.xml'
    
contrastBad = fullpathprocessed + 'constrastBad.bin'
contrastOK = fullpathprocessed + 'constrastOK.bin'
contrastSVMradial = fullpathSVM + 'contrastRBF.xml'
contrastSVMlinear = fullpathSVM + 'contrastLinear.xml'
    
darkBad = fullpathprocessed + 'darkBad.bin'
darkOK = fullpathprocessed + 'darkOK.bin'
darkSVMradial = fullpathSVM + 'darkRBF.xml'
darkSVMlinear = fullpathSVM + 'darkLinear.xml'
    
eyesBad = fullpathprocessed + 'eyesBad.bin'
eyesOK = fullpathprocessed + 'eyesOK.bin'
eyesSVMradial = fullpathSVM + 'eyesRBF.xml'
eyesSVMlinear = fullpathSVM + 'eyesLinear.xml'
    
flatBad = fullpathprocessed + 'flatBad.bin'
flatOK = fullpathprocessed + 'flatOK.bin'
flatSVMradial = fullpathSVM + 'flatRBF.xml'
flatSVMlinear = fullpathSVM + 'flatLinear.xml'

def createRadialSVMs():
    trainSVMRadial(flatBad, flatOK, flatSVMradial)
    trainSVMRadial(eyesBad, eyesOK, eyesSVMradial)
    trainSVMRadial(darkBad, darkOK, darkSVMradial)
    trainSVMRadial(contrastBad, contrastOK, contrastSVMradial)
    trainSVMRadial(qualityBad, qualityOK, qualitySVMradial)

def trainSVMRadial(badfile, okfile, saveLocation):
    
    badData = np.fromfile(badfile).reshape((-1, 8))
    badDataTraining = badData[0:int(badData.shape[0] * 0.8),:]
    badClassTraining = np.ones((badDataTraining.shape[0], 1))
    
    okData = np.fromfile(okfile).reshape((-1,8))
    okDataTraining = okData[0:int(okData.shape[0] * 0.8),:]
    okClassTraining = np.zeros((okDataTraining.shape[0], 1))
    
    trainingData = np.float32(np.vstack((badDataTraining, okDataTraining)))
    responses = np.int32(np.vstack((badClassTraining, okClassTraining)))[:,0]
    
    badMultiply = float(okClassTraining.size) / (badClassTraining.size + okClassTraining.size)
    okMultiply = float(badClassTraining.size) / (badClassTraining.size + okClassTraining.size)
    
    bestScore = 0
    bestC = 0
    bestG = 0
    
    for _ in range (0,5):

        data_train, data_test, labels_train, labels_test = train_test_split(trainingData, responses, test_size=0.20, random_state=42)
        
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        for c in C_range:
            for g in gamma_range:
                svm = cv2.ml.SVM_create()
                svm.setType(cv2.ml.SVM_C_SVC)
                svm.setKernel(cv2.ml.SVM_RBF)
                svm.setGamma(g)
                svm.setC(c)
                svm.setClassWeights(np.array([badMultiply, okMultiply]))
            
                svm.train(data_train, cv2.ml.ROW_SAMPLE, labels_train)
            
                output = svm.predict(data_test)[1].ravel()
    
                correct = np.sum(output == labels_test)
                total = output.size
                score = np.float32(correct) / total

                correctBad = np.sum(output[:labels_test.size] == labels_test[:labels_test.size])
                scoreBad = np.float32(correctBad) / labels_test.size
                
                totalScore = 3 * scoreBad + score
                
                if totalScore > bestScore:
                    bestScore = totalScore
                    bestC = c
                    bestG = g
                
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setGamma(bestG)
    svm.setC(bestC)
    svm.setClassWeights(np.array([badMultiply, okMultiply]))
    svm.train(trainingData, cv2.ml.ROW_SAMPLE, responses)
    
    svm.save(saveLocation)
    
    