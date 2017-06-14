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
from lasagne import layers
from lasagne.updates import adam
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet
import lasagne

from printclass import *

fullpathprocessed = '/Users/clmeiste/TimAikenDocs/SeniorProject/Database/Processed/smallimages/'
fullpathSVM = '/Users/clmeiste/TimAikenDocs/SeniorProject/Classifiers/SVM/'
fullpathNN = '/Users/clmeiste/TimAikenDocs/SeniorProject/Classifiers/CNN/'
    
qualityBad = fullpathprocessed + 'qualityBad.npy'
qualityOK = fullpathprocessed + 'qualityOk.npy'
qualityBadData = fullpathprocessed + 'qualityBad.csv'
qualityOKData = fullpathprocessed + 'qualityOK.csv'
qualitySVMradial = fullpathSVM + 'qualityRBF.xml'
qualitySVMlinear = fullpathSVM + 'qualityLinear.xml'
qualityNNparams = fullpathNN + 'qualityParams'
    
contrastBad = fullpathprocessed + 'contrastBad.npy'
contrastOK = fullpathprocessed + 'contrastOK.npy'
contrastBadData = fullpathprocessed + 'contrastBad.csv'
contrastOKData = fullpathprocessed + 'contrastOK.csv'
contrastSVMradial = fullpathSVM + 'contrastRBF.xml'
contrastSVMlinear = fullpathSVM + 'contrastLinear.xml'
contrastNNparams = fullpathNN + 'contrastParams'
    
darkBad = fullpathprocessed + 'darkBad.npy'
darkOK = fullpathprocessed + 'darkOK.npy'
darkBadData = fullpathprocessed + 'darkBad.csv'
darkOKData = fullpathprocessed + 'darkOK.csv'
darkSVMradial = fullpathSVM + 'darkRBF.xml'
darkSVMlinear = fullpathSVM + 'darkLinear.xml'
darkNNparams = fullpathNN + 'darkParams'
    
eyesBad = fullpathprocessed + 'eyesBad.npy'
eyesOK = fullpathprocessed + 'eyesOK.npy'
eyesBadData = fullpathprocessed + 'eyesBad.csv'
eyesOKData = fullpathprocessed + 'eyesOK.csv'
eyesSVMradial = fullpathSVM + 'eyesRBF.xml'
eyesSVMlinear = fullpathSVM + 'eyesLinear.xml'
eyesNNparams = fullpathNN + 'eyesParams'
    
flatBad = fullpathprocessed + 'flatBad.npy'
flatOK = fullpathprocessed + 'flatOK.npy'
flatBadData = fullpathprocessed + 'flatBad.csv'
flatOKData = fullpathprocessed + 'flatOK.csv'
flatSVMradial = fullpathSVM + 'flatRBF.xml'
flatSVMlinear = fullpathSVM + 'flatLinear.xml'
flatNNparams = fullpathNN + 'flatParams'

def createNN(badfile, okfile, outfile):
    # SIZE = number of columns in feature matrix
    # OUTPUTS = number of possible outputs (for binary classification this would be 2)
    
    badData = np.load(badfile).reshape((-1, 1, 100, 100))

    badDataTraining = badData[0:int(badData.shape[0] * 0.8),:,:]
    badClassTraining = np.ones((badDataTraining.shape[0], 1))

    okData = np.load(okfile).reshape((-1, 1, 100, 100))

    okDataTraining = okData[0:int(badClassTraining.size),:,:]
    okClassTraining = np.zeros((okDataTraining.shape[0], 1))

    trainingData = np.float32(np.vstack((badDataTraining, okDataTraining)))
    responses = np.int32(np.vstack((badClassTraining, okClassTraining)))[:,0]
    
    net1 = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d3', layers.Conv2DLayer),
            ('conv2d4', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('fc1', layers.DenseLayer),
            ('fc2', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape= (None, 1, 100, 100),  # this code won't compile without SIZE being set
        
        conv2d1_num_filters=32,
        conv2d1_filter_size=(3, 3),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),    
        
        conv2d2_num_filters=32,
        conv2d2_filter_size=(3, 3),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d2_W=lasagne.init.GlorotUniform(), 
        
        maxpool1_pool_size=(2, 2),
        
        conv2d3_num_filters=32,
        conv2d3_filter_size=(3, 3),
        conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d3_W=lasagne.init.GlorotUniform(),
        
        conv2d4_num_filters=32,
        conv2d4_filter_size=(3, 3),
        conv2d4_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d4_W=lasagne.init.GlorotUniform(),
        
        maxpool2_pool_size=(2, 2),
        
        fc1_num_units=200,
        fc1_nonlinearity=lasagne.nonlinearities.rectify,
        
        fc2_num_units=200,
        fc2_nonlinearity=lasagne.nonlinearities.rectify, 
        
        output_nonlinearity=softmax,  # output layer uses identity function
        output_num_units=2,  # this code won't compile without OUTPUTS being set

        # optimization method:
        update=adam,
        update_learning_rate=0.00001, 
        #update_momentum=0.9,

        regression=False,  # If you're doing classification you want this off
        max_epochs=100,  # more epochs can be good, 
        verbose=1, # enabled so that you see meaningful output when the program runs
        )

    
    net1.fit(trainingData, responses)
    net1.save_params_to(outfile)
    
    testNN(badfile, okfile, net1)

def testNN(badfile, okfile, badDataFile, okDataFile, nnfile):
    
    net1 = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d3', layers.Conv2DLayer),
            ('conv2d4', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('fc1', layers.DenseLayer),
            ('fc2', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape= (None, 1, 100, 100),  # this code won't compile without SIZE being set
        
        conv2d1_num_filters=32,
        conv2d1_filter_size=(3, 3),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),    
        
        conv2d2_num_filters=32,
        conv2d2_filter_size=(3, 3),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d2_W=lasagne.init.GlorotUniform(), 
        
        maxpool1_pool_size=(2, 2),
        
        conv2d3_num_filters=32,
        conv2d3_filter_size=(3, 3),
        conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d3_W=lasagne.init.GlorotUniform(),
        
        conv2d4_num_filters=32,
        conv2d4_filter_size=(3, 3),
        conv2d4_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d4_W=lasagne.init.GlorotUniform(),
        
        maxpool2_pool_size=(2, 2),
        
        fc1_num_units=200,
        fc1_nonlinearity=lasagne.nonlinearities.rectify,
        
        fc2_num_units=200,
        fc2_nonlinearity=lasagne.nonlinearities.rectify, 
        
        output_nonlinearity=softmax,  # output layer uses identity function
        output_num_units=2,  # this code won't compile without OUTPUTS being set

        # optimization method:
        update=adam,
        update_learning_rate=0.00001, 
        #update_momentum=0.9,

        regression=False,  # If you're doing classification you want this off
        max_epochs=100,  # more epochs can be good, 
        verbose=1, # enabled so that you see meaningful output when the program runs
        )
    
    net1.load_params_from(nnfile)
    
    badData = np.load(badfile).reshape((-1, 1, 100, 100))
    badDataTesting = badData[int(badData.shape[0] * 0.8):,:]
    badClassTesting = np.ones(badDataTesting.shape[0])
    
    okData = np.load(okfile).reshape((-1, 1, 100, 100))
    okDataTesting = okData[int(okData.shape[0] * 0.8):,:]
    okClassTesting = np.zeros(okDataTesting.shape[0])
    
    testingData = np.float32(np.vstack((badDataTesting, okDataTesting)))
    truth = np.int32(np.concatenate((badClassTesting, okClassTesting)))
    
    output = net1.predict(testingData)
    
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
        
        print(nnfile)
        
        print("False negative")
        for f in filesIncorrectBad[0:5]:
            print f
        
        print("False positive")
        for f in filesIncorrectOK[0:5]:
            print f
    
    return score, truePositive, falseNegative, falsePositive, trueNegative

def testAllCNNs():
    eS, eTP, eFN, eFP, eTN = testNN(eyesBad, eyesOK, eyesBadData, eyesOKData, eyesNNparams)
    fS, fTP, fFN, fFP, fTN = testNN(flatBad, flatOK, flatBadData, flatOKData, flatNNparams)
    dS, dTP, dFN, dFP, dTN = testNN(darkBad, darkOK, darkBadData, darkOKData, darkNNparams)
    cS, cTP, cFN, cFP, cTN = testNN(contrastBad, contrastOK, contrastBadData, contrastOKData, contrastNNparams)
    qS, qTP, qFN, qFP, qTN = testNN(qualityBad, qualityOK, qualityBadData, qualityOKData, qualityNNparams)
    printClassifier("CNN Eyes too dark", eS, eTP, eFN, eFP, eTN)
    printClassifier("CNN Face too flat", fS, fTP, fFN, fFP, fTN)
    printClassifier("CNN Face too dark", dS, dTP, dFN, dFP, dTN)
    printClassifier("CNN Face too contrasty", cS, cTP, cFN, cFP, cTN)
    printClassifier("CNN Quality of photo", qS, qTP, qFN, qFP, qTN)
