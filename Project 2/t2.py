# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def Crop(img):
    Y, X, _ = np.nonzero(img)
    minrow = np.min(X)
    mincol = np.min(Y)
    maxrow = np.max(X)
    maxcol = np.max(Y)
    img = img[mincol:maxcol, minrow:maxrow]
    return img

def SIFTFeatureDetection(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kps, features = sift.detectAndCompute(img, None)
    features = features / np.linalg.norm(features)
    kps = cv2.KeyPoint_convert(kps)
    return kps, features

def CrossCorrelationMatching(features1, features2, T):
    X = [] 
    Y = []
    M = len(features1)
    N = len(features2)
    D = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            D[i][j] = np.correlate(features1[i], features2[j])
    D = 1 - (D/D.max())
    
    for i in range(0,len(D)):
        if D[i].min() < T: # Tune
            X.append(i)
            Y.append(D[i].argmin())
            
    return X, Y

def WarpStitching(img1, img2, M):
    width1, height1 = img1.shape[:2]
    width2, height2 = img2.shape[:2]

    X = np.float32([[0, 0], [0, width1], [height1, width1], [height1, 0]]).reshape(-1,1,2)
    Y = np.float32([[0, 0], [0, width2], [height2, width2], [height2, 0]]).reshape(-1,1,2)
    Y = cv2.perspectiveTransform(Y, M)
    
    Z = np.concatenate((X, Y), axis = 0)
    minZ = Z.min(axis = 0).flatten()
    maxZ = Z.max(axis = 0).flatten()
    [minrow, mincol] = np.int32(minZ - 0.5)
    [maxrow, maxcol] = np.int32(maxZ + 0.5)
    
    H = np.array([[1, 0 , -minrow], [0, 1, -mincol], [0, 0, 1]])
    
    warped = cv2.warpPerspective(img2, H.dot(M),(maxrow - minrow, maxcol - mincol))
    warped[-mincol:width1-mincol, -minrow:height1-minrow] = img1
    stitched = warped
    
    return stitched


def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    
    final = imgs[0]
    for i in range(1, len(imgs)):
        kps1, features1 = SIFTFeatureDetection(final)
        kps2, features2 = SIFTFeatureDetection(imgs[i])
        X, Y = CrossCorrelationMatching(features1, features2, T = 0.01)
        match_list1 = kps1[X]
        match_list2 = kps2[Y]
        M, _ = cv2.findHomography(np.array(match_list1), np.array(match_list2), cv2.RANSAC, 5.0)
        final = WarpStitching(imgs[i], final, M)

    final = Crop(final)
    cv2.imwrite(savepath, final)
    
    M = len(imgs)
    overlap_arr = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            kps1, features1 = SIFTFeatureDetection(imgs[i])
            kps2, features2 = SIFTFeatureDetection(imgs[j])
            X, Y = CrossCorrelationMatching(features1, features2, T = 0.01)
            if len(X) > 50:
                overlap_arr[i][j] = 1
    
    return overlap_arr

if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', N=5, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
