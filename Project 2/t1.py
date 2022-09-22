#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt

def SIFTFeatureDetection(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kps, features = sift.detectAndCompute(img, None)
    features = features / np.linalg.norm(features)
    kps = cv2.KeyPoint_convert(kps)
    return kps, features

def CrossCorrelationMatching(features1, features2, T): # Gave best results

    X = []
    Y = []
    M = len(features1)
    N = len(features2)
    distance = np.zeros((1, M))
    min_distance = np.zeros((1, N))

    n_matches = 0
            
    for i in range(1, N):
        for j in range(1, M):
            diff = features2[i][:] - features1[j][:]
            distance[0][j] = np.linalg.norm(diff)
        min_index = np.argsort(distance[0])
        distance[0] = np.sort(distance[0])
        min_distance[0][i] = distance[0][1]
        if distance[0][1] < T * distance[0][2]:
            n_matches = n_matches + 1
            X.append(i)
            Y.append(min_index[1])
    
    return X, Y
            

# def CrossCorrelationMatching1(features1, features2, T):
#     X = [] 
#     Y = []
#     M = len(features1)
#     N = len(features2)
#     D = np.zeros((M, N))
#     for i in range(M):
#         for j in range(N):
#             D[i][j] = np.correlate(features1[i], features2[j])
#     D = 1 - (D/D.max())
    
#     for i in range(0,len(D)):
#         if D[i].min() < T: # Tune
#             X.append(i)
#             Y.append(D[i].argmin())
            
#     return X, Y

# def CrossCorrelationMatching2(features1, features2, T):    
#     X = []
#     Y = []
#     M = len(features1)
#     N = len(features2)
#     for i in range(N):
#         matches = []
#         for j in range(M):
#             difference = features2[i]-features1[j]
#             product = np.dot(difference, difference)
#             matches.append(product)
#         minimum = np.min(matches)
#         if(minimum < T):
#             X.append(i)
#             Y.append(matches.index(minimum))

#     return X, Y

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
    
    warped = cv2.warpPerspective(img2, M,(maxrow - minrow, maxcol - mincol), dst = img1.copy(), flags = cv2.INTER_NEAREST, borderMode = cv2.BORDER_TRANSPARENT)
    
    return warped

def Replace(img, final):
    M = img.shape[0]
    N = img.shape[1]
    for i in range(0, M-1):
        for j in range(0, N-1):
            final_pixel = np.sum(final[i][j])
            img_pixel = np.sum(img1[i][j])
            if final_pixel < img_pixel:
                final[i][j] = img1[i][j]           
                
    return final


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    
    kps1, features1 = SIFTFeatureDetection(img1)
    kps2, features2 = SIFTFeatureDetection(img2)
    X, Y = CrossCorrelationMatching(features1, features2, T = 0.5)
    
    match_list1 = kps2[X]
    match_list2 = kps1[Y]

    H, _ = cv2.findHomography(np.array(match_list1), np.array(match_list2), cv2.RANSAC, 5.0)
    
    final = WarpStitching(img1, img2, H)
    final = Replace(img1, final)

    cv2.imwrite(savepath, final)

    return final

if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

