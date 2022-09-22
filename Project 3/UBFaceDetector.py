'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from helper import show_image

import cv2
import numpy as np
import os
import sys
import glob
import warnings
warnings.filterwarnings("error")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import face_recognition

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(input_path):
    result_list = []
    absolute_path = "/Users/volkopat/opt/anaconda3/envs/ComputerVision/lib/python3.7/site-packages/cv2/data/"
    for filename in glob.glob(input_path + '/*.jpg'):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(absolute_path + 'haarcascade_frontalface_alt.xml')
        # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(img, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            result_list.append({"iname": filename.split('/')[2], "bbox": [int(x), int(y), int(w), int(h)]})
    
    return result_list


'''
K: number of clusters
'''
def cluster_faces(input_path, K):
    result_list = []
    cropped = []
    boxes = []
    files = []
    absolute_path = "/Users/volkopat/opt/anaconda3/envs/ComputerVision/lib/python3.7/site-packages/cv2/data/"
    for filename in glob.glob(input_path + '/*.jpg'):
        img = cv2.imread(filename)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(absolute_path + 'haarcascade_frontalface_alt.xml')
        # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(img, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cropped.append(img[y:y+h, x:x+w])
            boxes.append([x, y, x+w, y+h])
            files.append(filename.split('/')[1])
    
    encoded = []
    for i in range(len(cropped)):
        x = face_recognition.face_encodings(cropped[i])
        encoded.append(x[0])
    
    kmeans = KMeans(n_clusters = K)
    kmeans.fit(encoded)
    
    clusters = kmeans.labels_
    # clusters = KMeans(int(K), encoded)
    
    # Handle RuntimeWarning
    # while True:
    #     try:
    #         clusters = KMeans(int(K), encoded)
    #     except RuntimeWarning:
    #         continue
    #     break
    
    files = np.array(files)
    cluster_no, counts = np.unique(clusters, return_counts = True)
    
    # Handle wrong clusters
    # while True:
    #     if any(i > 9 for i in counts) == True:
    #         while True:
    #             try:
    #                 clusters = KMeans(int(K), encoded)
    #             except RuntimeWarning:
    #                 continue
    #             break
    #         cluster_no, counts = np.unique(clusters, return_counts = True)
    #     else: break
    
    file_list = []
    
    for i in range(len(cluster_no)):
        file_list.append([])
    for i in range(len(clusters)):
        file_list[clusters[i]].append(files[i])
        
        # Handle IndexError
        # while True:
        #     try:
        #         file_list[clusters[i]].append(files[i])
        #     except IndexError:
        #         clusters = KMeans(int(K), encoded)
        #     break 
        
    for i in range(len(cluster_no)):
        result_list.append({"cluster_no": int(cluster_no[i]), "elements": file_list[i]})
    
    DisplayImages(input_path, result_list)
    
    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""
# Self Coded KMeans algorithm from scratch
# def KMeans(K, encoded):
#     encoded = np.array(encoded)
#     n = encoded.shape[0]
#     c = encoded.shape[1]
#     mean = np.mean(encoded, axis = 0)
#     std = np.std(encoded, axis = 0)
    
#     centers = np.random.randn(K, c) * std + mean
#     centers_old = np.zeros(centers.shape)
#     centers_new = centers.copy() 
#     clusters = np.zeros(n)
#     distances = np.zeros((n,K))
#     error = np.linalg.norm(centers_new - centers_old)
#     for _ in range(100): #100 epochs
#         for i in range(K):
#             distances[:,i] = np.linalg.norm(encoded - centers_new[i],  axis = 1)
#         clusters = np.argmin(distances, axis = 1)
#         centers_old = centers.copy()
#         for i in range(K):
#             centers_new[i] = np.mean(encoded[clusters == i], axis=0)
#         error = np.linalg.norm(centers_new - centers_old)
        
#     return clusters

# To display Image Clusters

def DisplayImages(input_path, results):
    i = 0
    imgs = []
    while i < len(results):
        merged_img = []
        for j in results[i]['elements']:
            img = cv2.imread(input_path + "/" + j)
            merged_img.append(img)
        imgs.append(np.concatenate(merged_img))
        i = i+1
    
    plt.figure(figsize=(20,25))
    for i in range(0, len(imgs)):
        plt.subplot(1,5, i+1)
        plt.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
    plt.show()

# Dead code: Tried Viola Jones Algorithm from scratch:

# def Intensities(img, x1, y1, x2, y2):
#     a = img[y1][x1]
#     b = img[y1][x2]
#     c = img[y2][x1]
#     d = img[y2][x2]
#     intensity = (d + a)- (b + c)
    
#     return intensity

# def HaarFeatures(img, haar, x, y, hx, hy): # Haar Features
#     mx = hx - 1
#     my = hy - 1
#     if haar == 1: # Top White Bottom Black
#         white = Intensities(img, x, y, x + mx, y + math.floor(my/2))
#         black = Intensities(img, x, y + math.ceil(my/2), x + mx, y + my)
#         result = white - black

#     elif haar == 2: # Left White Right Black
#         white = Intensities(img, x, y, x + math.floor(mx/2), y + my)
#         black = Intensities(img, x + math.ceil(mx/2), y, x + mx, y + my)
#         result = white - black

#     elif haar == 3: # Top and Bottom White Middle Black
#         white1 = Intensities(img, x, y, x + mx, y + math.floor(my/3))
#         black = Intensities(img, x, y + math.ceil(my/3), x + mx, y + math.floor((2 * my)/3))
#         white2 = Intensities(img, x, y + math.ceil((2 * my)/3), x + mx, y + my)
#         result = (white1 + white2) - black

#     elif haar == 4: # Left and Right White Middle Black
#         white1 = Intensities(img, x, y, x + math.floor(mx/3), y + my)
#         black = Intensities(img, x + math.ceil(mx/3), y, x + math.floor((2 * mx)/3), y + my)
#         white2 = Intensities(img, x + math.ceil((2 * mx)/3), y, x + mx, y + my)
#         result = (white1 + white2) - black

#     elif haar == 5: #Chessboard
#         white1 = Intensities(img, x, y, x + math.floor(mx/2), y + math.floor(my/2))
#         black1 = Intensities(img, x + math.ceil(mx/2), y, x + mx, y + math.floor(my/2))
#         white2 = Intensities(img, x, y + math.ceil(my/2), x + math.floor(mx/2), y + my)
#         black2 = Intensities(img, x + math.ceil(mx/2), y + math.ceil(my/2), x + mx, y + my)
#         result = (white1 + white2) - (black1 + black2)
    
#     return result

# def Integral(img):
#     x = img.shape[0]
#     y = img.shape[1]
#     result = np.array([img[0, 0]])
#     result = result[np.newaxis, :]
#     for j in range(1, y):
#         result = np.append(result, [[img[0, j] + result[0, j-1]]], axis = 1)
#     for i in range(1, x):
#         count = 0
#         temp = np.array([])
#         temp = temp[np.newaxis, :]
#         for j in range(0, y):
#             count = count + img[i, j]
#             temp = np.append(temp, [[count + result[i-1, j]]], axis = 1)
#         result = np.append(result, temp, axis = 0)
    
#     return result

# def FeatureVote(img):
#     score = Integral(img)
#     if score < polarity * threshold: return weight * 1
#     else: return weight * -1