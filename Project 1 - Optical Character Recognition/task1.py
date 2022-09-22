"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments
    results = []

    labels, templates = enrollment(characters)

    bbox, characters = detection(test_img)
    
    recog = recognition(labels, templates, characters)
    
    for i in range(len(recog)):
        entry = {"bbox": (bbox[i][1], bbox[i][0], bbox[i][3], bbox[i][2]), "name" : recog[i]}
        results.append(entry)
    
    return results

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    
    # Separating the character names and the image arrays from Character
    chars = []
    labels = []
    for character in characters:
        labels.append(character[0])
        chars.append(character[1])
        
    # Resizing the chars by trimming the surrounding area
    
    templates = []
    for img in chars:
        img = ConvertToBW(img)
        img = ConvertToBinary(img)
        minrow, maxrow, mincol, maxcol = TemplateResize(img)
        img = np.array(img)
        templates.append(img[minrow:maxrow, mincol:maxcol])
        
    # Displaying the templates as a demo
    
    # plt.figure(figsize=(10,5))
    # for i in range(0, len(templates)):
    #     plt.subplot(2,3,i+1)
    #     plt.imshow(templates[i])
    # plt.show()
    
    return labels, templates
    
def detection(img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    
    # Extracting the features using Connected Component Labeling
    
    img = ConvertToBW(img)
    img = ConvertToBinary(img)
    img = np.array(img)
    components = ConnectedComponents(img)
    minrow, maxrow, mincol, maxcol = CharacterExtraction(components)
    bbox, characters = Cropping(minrow, maxrow, mincol, maxcol, components)
    
    # Displaying the components as a demo
    
    # plt.figure(figsize=(30,15))
    # for i in range(0, len(characters)):
    #     plt.subplot(10,14,i+1)
    #     plt.imshow(characters[i])
    # plt.show()
    
    return bbox, characters
    

def recognition(labels, templates, characters):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

    recognized = []
    for i in range(len(characters)):
        corr = []
        maximum = 0
        pos = 0
        for j in range(len(templates)):
            corr.append(NormalizedCrossCorrelation(characters[i], templates[j]))
        for i in range(len(corr)):
            if corr[i] >= maximum:
                maximum = corr[i]
                pos = i
        if maximum > 0.8:
            recognized.append(labels[pos])
        else:
            recognized.append("UNKNOWN")
    
    return recognized

def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)

################################ ENROLLMENT ##############################

# Changing all values to 0 for Black and 255 for white

def ConvertToBW(img):
    
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] > 127:
                img[i][j] = 255
            else: img[i][j] = 0
    
    return img

# Convert to Binary

def ConvertToBinary(img):
    
    binaryimg = [[0 for c in range(img.shape[1])] for r in range(img.shape[0])]
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] == 0:
                binaryimg[i][j] = 1
    
    return binaryimg

################################ DETECTION ##############################

# Connected Components Labeling

def ConnectedComponents(img):
    
    components = np.ones([img.shape[0],img.shape[1]])
    count = 0
    link = []
    id = 0
    
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] == 0:
                components[i][j] = 0
            else:
                new_left = components[i-1][j]
                new_top = components[i][j-1]
                if new_left == 0 and new_top == 0:
                    count += 1
                    components[i][j] = count
                else:
                    if np.min([new_left,new_top]) == 0 or new_left == new_top:
                        components[i][j] = np.max([new_left,new_top])
                    else:
                        components[i][j] = np.min([new_left,new_top])
                        if id == 0:
                            link.append([new_left,new_top])
                            id += 1
                        else:
                            check = 0
                            for k in range(id):
                                tmp = set(link[k]).intersection(set([new_left,new_top]))
                                if len(tmp) != 0 :
                                    link[k] = set(link[k]).union([new_left,new_top])
                                    np.array(link)
                                    check += 1
                            if check == 0:
                                id += 1
                                np.array(link)
                                link.append(set([new_left,new_top]))
    
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(id):
                if (components[i][j] in link[k]) and components[i][j] != 0 :
                    components[i][j] = min(link[k])


    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(id):
                if components[i][j] == min(link[k]):
                    components[i][j] = k+1
                    
    return components

# Resizing the Character list

def TemplateResize(img):
    
    rowDict = []
    colDict = []
    for i in range(len(img)):
        for j in range(len(img[i])):
            if(img[i][j] == 0): continue
            else:
                rowDict.append(i)
                colDict.append(j)
     
    minrow = min(rowDict)
    mincol = min(colDict)
    maxrow = max(rowDict)
    maxcol = max(colDict)
    
    return minrow, maxrow, mincol, maxcol

# Extracting characters from test image

def CharacterExtraction(components):
    
    rowDict = defaultdict(list)
    colDict = defaultdict(list)
    for i in range(len(components)):
        for j in range(len(components[i])):
            if(components[i][j] == 0): continue
            else:
                rowDict[components[i][j]].append(i)
                colDict[components[i][j]].append(j)

    minrow = defaultdict(list)
    mincol = defaultdict(list)
    maxrow = defaultdict(list)
    maxcol = defaultdict(list)

    for i in rowDict:
        minrow[i] = min(rowDict[i])
        maxrow[i] = max(rowDict[i])
    for i in colDict:
        mincol[i] = min(colDict[i])
        maxcol[i] = max(colDict[i])
    
    return minrow, maxrow, mincol, maxcol

# Getting test image coordinates x,y,w,h

def Cropping(minrow, maxrow, mincol, maxcol, components):
    
    bbox = []
    characters = []
    for i in mincol:
        bbox.append([minrow[i],mincol[i],maxrow[i] - minrow[i], maxcol[i] - mincol[i]])
        characters.append(components[minrow[i]:maxrow[i], mincol[i]:maxcol[i]])
    
    return bbox, characters

################################ RECOGNITION ##############################

# Finding the mean of the matrix

def MatrixMean(template):
    s = 0
    for i in range(template.shape[0]):
        for j in range(template.shape[1]):
            s += template[i][j]
    
    matrixShape = template.shape[0] * template.shape[1]
    matrixMean = s/matrixShape
    
    return matrixMean

# Normalizing the matrix

def MatrixNormalize(template):
    template = np.square(template)
    template = np.sum(template)
    template = np.sqrt(template)
    
    return template

# Resizing the arrays by padding

def MatrixResize(template, image):
    M = template.shape
    N = image.shape
    if M[0] > N[0] and M[1] > N[1]:
        image = np.pad(image,((M[0]-N[0],0),(M[1]-N[1],0)))
    if M[0] < N[0] and M[1] < N[1]:
        template = np.pad(template,((N[0]-M[0],0),(N[1]-M[1],0)))
    if M[0] > N[0] and M[1] < N[1]:
        image = np.pad(image,((M[0]-N[0],0),(0,0)))
        template = np.pad(template,((0,0),(N[1]-M[1],0)))
    if M[0] < N[0] and M[1] > N[1]:
        template = np.pad(template,((N[0]-M[0],0),(0,0)))
        image = np.pad(image,((0,0),(M[1]-N[1],0)))
    if M[0] > N[0] and M[1] == N[1]:
        image = np.pad(image,((M[0]-N[0],0),(0,0)))
    if M[0] < N[0] and M[1] == N[1]:
        template = np.pad(template,((N[0]-M[0],0),(0,0)))
    if M[0] == N[0] and M[1] > N[1]:
        image = np.pad(image,((0,0),(M[1]-N[1],0)))
    if M[0] == N[0] and M[1] < N[1]:
        template = np.pad(template,((0,0),(N[1]-M[1],0)))
    
    return template, image

# Normalized Cross Correlation to find similarity between template and image

def NormalizedCrossCorrelation(template, image):
    template, image = MatrixResize(template, image)
    correlation = np.sum(template * image)
    image = MatrixNormalize(image)
    template = MatrixNormalize(template)
    normal = image * template

    return correlation / normal

def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
