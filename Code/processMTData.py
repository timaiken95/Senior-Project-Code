
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
from featuresForImage import *

fullpathclassified = '/Users/clmeiste/TimAikenDocs/SeniorProject/Database/Classified/'
fullpathprocessed = '/Users/clmeiste/TimAikenDocs/SeniorProject/Database/Processed/smallimages/'

def createMatrix(inFileCSV, outFileBinary, outFileCSV):
    with open(inFileCSV, 'rb') as csvinfile, open(outFileCSV, 'wb') as csvoutfile:
        
        reader = csv.reader(csvinfile, delimiter=',', quotechar='|')
        writer = csv.writer(csvoutfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        results = []
        url_data = list(reader)
        
        with ThreadPoolExecutor(max_workers = 20) as executor:
            futures = []
            lock = threading.Lock()
        
            for i, elem in enumerate(url_data):
                url = url_data[i][0]
                futures.append(executor.submit(createFeatureVectorForImage, url, True))
            
            for future in as_completed(futures):
                newVec, url = future.result()
                if newVec.size == 8 or newVec.size == 10000:
                    results.append(newVec)
                    writer.writerow([url])
        
        finalResults = np.asarray(results)
        print(finalResults.shape)
        np.save(outFileBinary, finalResults)

def processAllData():  
    eyesBad = 'eyesBad'
    eyesOK = 'eyesOK'
    darkBad = 'darkBad'
    darkOK = 'darkOK'
    contrastBad = 'constrastBad'
    contrastOK = 'constrastOK'
    flatBad = 'flatBad'
    flatOK = 'flatOK'
    qualityBad = 'qualityBad'
    qualityOK = 'qualityOK'
    
    createMatrix(fullpathclassified + eyesBad + ".csv",
                fullpathprocessed + eyesBad,
                fullpathprocessed + eyesBad + ".csv")
    createMatrix(fullpathclassified + eyesOK + ".csv",
                 fullpathprocessed + eyesOK,
                 fullpathprocessed + eyesOK + ".csv")
    createMatrix(fullpathclassified + darkBad + ".csv",
                 fullpathprocessed + darkBad,
                 fullpathprocessed + darkBad + ".csv")
    createMatrix(fullpathclassified + darkOK + ".csv",
                 fullpathprocessed + darkOK,
                 fullpathprocessed + darkOK + ".csv")
    createMatrix(fullpathclassified + contrastBad + ".csv",
                 fullpathprocessed + contrastBad,
                 fullpathprocessed + contrastBad + ".csv")
    createMatrix(fullpathclassified + contrastOK + ".csv",
                 fullpathprocessed + contrastOK,
                 fullpathprocessed + contrastOK + ".csv")
    createMatrix(fullpathclassified + flatBad + ".csv",
                 fullpathprocessed + flatBad,
                 fullpathprocessed + flatBad + ".csv")
    createMatrix(fullpathclassified + flatOK + ".csv",
                 fullpathprocessed + flatOK,
                 fullpathprocessed + flatOK + ".csv")
    createMatrix(fullpathclassified + qualityBad + ".csv",
                 fullpathprocessed + qualityBad,
                 fullpathprocessed + qualityBad + ".csv")
    createMatrix(fullpathclassified + qualityOK + ".csv",
                 fullpathprocessed + qualityOK,
                 fullpathprocessed + qualityOK + ".csv")
