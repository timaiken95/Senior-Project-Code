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

facialTriangles = [np.array([0, 17, 18, 19, 20, 21, 27, 39, 40, 41, 36]), # left eye
                    np.array([16, 45, 46, 47, 42, 27, 22, 23, 24, 25, 26]), # right eye
                    np.array([0, 36, 41, 40, 39, 27, 31, 49, 60, 48, 4, 3, 2, 1]), # left cheek
                    np.array([16, 15, 14, 13, 12, 54, 64, 53, 35, 27, 42, 47, 46, 45]), # right cheek
                    np.array([27, 35, 34, 33, 32, 31]), # nose
                    np.array([31, 32, 33, 34, 35, 53, 64, 54, 55, 56, 57, 58, 59, 48, 60, 49]), # mouth
                    np.array([4, 48, 59, 58, 57, 8, 7, 6, 5]), # left chin
                    np.array([12, 11, 10, 9, 8, 57, 56, 55, 54]) # right chin
                    ]

fullpathhaar = '/Users/clmeiste/anaconda/pkgs/opencv3-3.1.0-py27_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
fullpathlandmarks = '/Users/clmeiste/TimAikenDocs/SeniorProject/Code/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(fullpathlandmarks)

def normalize(arr):
    rng = np.max(arr.flatten()) - np.min(arr.flatten())
    amin = np.min(arr.flatten())
    toReturn = (arr - amin) * 255.0 / rng
    return toReturn

def createFeatureVectorForImage(url, affine=False):

    img = requests.get(url = url, stream = True)
    if img.status_code == 200:

        img.raw.decode_content = True
        im = Image.open(img.raw)
        
        if np.array(im).ndim > 2:
            gray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
        else:
            gray = np.array(im)
        
        gray = normalize(gray)
        gray = np.uint8(gray)
        
        face_cascade = cv2.CascadeClassifier(fullpathhaar)
        faces = face_cascade.detectMultiScale(gray,
                                            scaleFactor=1.05,
                                            minNeighbors=10,
                                            minSize=(128, 128))
        
        if len(faces) == 1:
            (x,y,w,h) = faces[0]
            rect = dlib.rectangle(long(x), long(y), long(x + w), long(y + h))
            shape = predictor(gray, rect)
            landmarks = face_utils.shape_to_np(shape)
            
            if not affine:
                features = computeFeatures(gray, landmarks)
            else:
                features = computeAffineSmall(gray, landmarks)
            
            return features, url
            
    return np.array([]), url

def computeFeatures(img, landmarks):
    
    vec = []
    
    for r in range(0,len(facialTriangles)):
        region = landmarks[facialTriangles[r],:]
        leftbound = max(np.min(region[:,0]) - 1, 0)
        rightbound = min(np.max(region[:,0]) + 1, img.shape[1] - 1)
        topbound = max(np.min(region[:,1]) - 1, 0)
        bottombound = min(np.max(region[:,1]) + 1, img.shape[0] - 1)
        
        pixels = []
        regionPath = mpltPath.Path(region)
        for x in range (leftbound, rightbound + 1):
            for y in range(topbound, bottombound + 1):
                if(regionPath.contains_point((x, y))):
                    pixels.append(img[y,x])
        
        val = np.mean(np.asarray(pixels))
        vec.append(val)
        
    return np.asarray(vec)

def computeAffineSmall(img, landmarks):
    toSelect = np.array([0, 16, 8])
    
    transformedPoints = np.float32(np.array([[0, 20],
                                          [100, 20],
                                          [50, 100]]))
    
    originalPoints = np.float32(landmarks[toSelect,:])
    
    affineTransform = cv2.getAffineTransform(originalPoints, transformedPoints)
    result = cv2.warpAffine(img, affineTransform, (100, 100))
    
    return result.flatten()
    
    