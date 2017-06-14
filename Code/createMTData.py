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

key = "acccd1fff36c7565632302b4035e0eb3"
secretkey = "015a07f86dd8b895"
fullpathdatabase = '/Users/clmeiste/TimAikenDocs/SeniorProject/Database/'

fullpathraw = '/Users/clmeiste/TimAikenDocs/SeniorProject/Database/Raw/'
fullpathclassified = '/Users/clmeiste/TimAikenDocs/SeniorProject/Database/Classified/'

fullpathhaar = '/Users/clmeiste/anaconda/pkgs/opencv3-3.1.0-py27_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'

detector = dlib.get_frontal_face_detector()

def check_photo(url, lock, csvwrite):
    img = requests.get(url = url, stream = True)
    if img.status_code == 200:

        img.raw.decode_content = True
        im = Image.open(img.raw)
        grey = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
        
        face_cascade = cv2.CascadeClassifier(fullpathhaar)
        faces = face_cascade.detectMultiScale(grey,
                                            scaleFactor=1.05,
                                            minNeighbors=10,
                                            minSize=(128, 128))
        if len(faces) == 1:
            lock.acquire()
            csvwrite.writerow([url])
            lock.release()
        
    return "done"

def flickr_search(query, pathWrite):
    
    pages = []
    for page in range(1,9):
        params = {'method': 'flickr.photos.search',
                    'api_key': key,
                    'text': query,
                    'per_page': 500,
                    'page': page,
                    'format': 'json'
                }
    
        urlStart = 'http://flickr.com/services/rest/?'
        url = urlStart + urllib.urlencode(params)

        r = requests.get(url = url)
        start, json_string = r.text.split("(", 1)
        json_string = json_string[:-1]
        j = json.loads(json_string)
        pages.append(j)
    
    with ThreadPoolExecutor(max_workers = 20) as executor:
        futures = []
        lock = threading.Lock()
        
        with open(pathWrite + 'photoURLs.csv', 'wb') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
            i = 0
            for json_data in pages:
                i += 1
                photos = json_data['photos']['photo']
                for photo in photos:
                    photo_url = "https://farm" + str(photo['farm']) + ".staticflickr.com/"
                    photo_url += str(photo['server']) + "/" + str(photo['id']) + "_"
                    photo_url += str(photo['secret']) + "_b.jpg"
                    futures.append(executor.submit(check_photo, photo_url, lock, csvwriter))
            
            count = 0
            for future in as_completed(futures):
                count += 1
                print(count)
                #print(future.result())

def toMTFile(csvinpath, csvoutpath):
    with open(csvoutpath, 'wb') as csvoutfile, open(csvinpath, 'rb') as csvinfile:
        
        csvreader = csv.reader(csvinfile, delimiter=' ', quotechar='|')
        csvwriter = csv.writer(csvoutfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        csvwriter.writerow(['url1', 'url2', 'url3', 'url4', 'url5'])
        
        url_data = list(csvreader)
        
        length = len(url_data)
        for i, row in enumerate(url_data):
            url1 = url_data[i][0]
            url2 = url_data[(i + 1) % length][0]
            url3 = url_data[(i + 2) % length][0]
            url4 = url_data[(i + 3) % length][0]
            url5 = url_data[(i + 4) % length][0]
            
            csvwriter.writerow([url1, url2, url3, url4, url5])

def create():
    flickr_search("portrait", fullpathraw)
    inPath = fullpathraw + 'photoURLs.csv'
    outPath = fullpathraw + 'mtURLs.csv'
    toMTFile(inPath, outPath)
            