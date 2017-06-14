import requests
import json
import os
import numpy as np
import cv2
import glob
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
import threading
import csv
import urllib
import imutils
import dlib
import matplotlib.pyplot as plt
from PIL import Image

fullpathraw = '/Users/clmeiste/TimAikenDocs/SeniorProject/Database/Raw/'
fullpathclassified = '/Users/clmeiste/TimAikenDocs/SeniorProject/Database/Classified/'
fullpathhaar = '/Users/clmeiste/anaconda/pkgs/opencv3-3.1.0-py27_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'

def processInput(row, inputKey, answerKey, dictionary):
    image = row[inputKey]
    if dictionary.has_key(image):
        dictionary[image].append(row[answerKey])
    else:
        dictionary[image] = [row[answerKey]]
    
    return dictionary

def writeBinaryOutput(dictionary, filePathBad, filePathOK):
     with open(filePathBad, 'wb') as badFile, open(filePathOK, 'wb') as okFile:
        csvwriterBad = csv.writer(badFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriterOK = csv.writer(okFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        for key in dictionary:
            results = dictionary[key]
            badPercent = results.count('YES') / float(len(results))
            goodPercent = results.count('NO') / float(len(results))
            
            if badPercent >= 0.7:
                csvwriterBad.writerow([key])
            elif goodPercent >= 0.7:
                csvwriterOK.writerow([key])
                
def mtFileToData(csvpath):
    with open(csvpath, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        
        eyes = {}
        dark = {}
        contrast = {}
        flat = {}
        quality = {}
        
        for row in reader:
            
            eyes = processInput(row, 'Input.url1', 'Answer.Q1Answer', eyes)
            dark = processInput(row, 'Input.url2', 'Answer.Q2Answer', dark)
            contrast = processInput(row, 'Input.url3', 'Answer.Q3Answer', contrast)
            flat = processInput(row, 'Input.url4', 'Answer.Q4Answer', flat)
            quality = processInput(row, 'Input.url5', 'Answer.Q5Answer', quality)
        
        eyesBad = fullpathclassified + 'eyesBad.csv'
        eyesOK = fullpathclassified + 'eyesOK.csv'
        darkBad = fullpathclassified + 'darkBad.csv'
        darkOK = fullpathclassified + 'darkOK.csv'
        contrastBad = fullpathclassified + 'constrastBad.csv'
        contrastOK = fullpathclassified + 'constrastOK.csv'
        flatBad = fullpathclassified + 'flatBad.csv'
        flatOK = fullpathclassified + 'flatOK.csv'
        qualityBad = fullpathclassified + 'qualityBad.csv'
        qualityOK = fullpathclassified + 'qualityOK.csv'
        
        writeBinaryOutput(eyes, eyesBad, eyesOK)
        writeBinaryOutput(dark, darkBad, darkOK)
        writeBinaryOutput(contrast, contrastBad, contrastOK)
        writeBinaryOutput(flat, flatBad, flatOK)
        writeBinaryOutput(quality, qualityOK, qualityBad)

def sort():
    path = fullpathclassified + 'mtResults.csv'
    mtFileToData(path)
        