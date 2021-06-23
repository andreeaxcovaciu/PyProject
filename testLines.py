# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:52:17 2021

@author: covac
"""

#obtinerea unghiurilor muchiilor
#magnitudine = puterea muchiei
#inclinarea
#CANNY

#INCERCARI DE PROCESARE

#img = cv2.imread('D:\\XFlxN.png', 0)
#img = Image.open('D:\\XFlxN.png').convert('RGBA')  
#img = np.array(img, dtype = np.uint16)
#img *= 256

#img1 = Image.open('D:\\_WorkTmpFilesTst\\08_07\\Clasa2\\1_4.jpg')  
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)
#fig = plt.figure(figsize=(30,6))
#ax = fig.add_subplot(121); ax.imshow(img, cmap=plt.cm.bone); #plt.show(

    
    #kernel = np.ones((3,2),np.uint8)
    #opening = cv2.morphologyEx(img_b, cv2.MORPH_OPEN, kernel)
    #functii morfologice de curatare
    #punctele albe 
    #image = invert(opening)



#image = cv2.Canny(image,50,150,3)
#skeleton  = invert(skeleton)
#bw = cv2.threshold(skeleton, 127, 255, cv2.THRESH_BINARY) 
#print('image: ',  skeleton.depth)


#ALTE INCERCARI

# a,b = lines.shape
# for i in range(a):
#     rho = lines[i][0][0]
#     theta = lines[i][0][1]
#     a = math.cos(theta)
#     b = math.sin(theta)
#     x0, y0 = a*rho, b*rho
#     pt1 = ( int(x0+100*(-b)), int(y0+100*(a)) )
#     pt2 = ( int(x0-100*(-b)), int(y0-100*(a)) )
    # cv2.line(result, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
# import functools

# distance = []
# for linet in lines:
#     my_result = functools.reduce(lambda sub, elem: sub * 10 + elem, linet)
#     distance.append(np.linalg.norm(linet[:,:2] - linet[:,2:]))

# print('max distance:',max(distance),'\nmin distance:',min(distance))

# # Adjusting the best distance 
 # bestDistance=1110

# numberOfLines=[]
# count=0
# for im in distance:
    
#     if im > bestDistance:
#         numberOfLines.append(im)
#         count=count+1

# print('Number of lines:',count)

# contours, hierarchy= cv2.findContours(skeleton.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# number_of_objects_in_image= len(contours)

# #1st line is the total numbers lines of the image
# features=number_of_objects_in_image

# print ("The number of objects in this image: ", str(number_of_objects_in_image))



import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import sys
import shutil

import numpy as np
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage.morphology import thin


#img = cv2.imread('D:\\_WorkTmpFilesTst\\clasasvm4\\4_16.jpeg', 0)
#path = 'D:\\_WorkTmpFilesTst\\24bits\\'


#Cod de culoare in plot

#FUNCTIONS

def showImages(img,skeleton,lines, str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    
    ax[1].imshow(skeleton, cmap=plt.cm.gray)
    #ax[1].set_title('Canny edges')
    
    ax[2].imshow(skeleton * 0)
    for line in lines:
        p0, p1 = line

       # computeSlopes(p0,p1)       
     
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, img.shape[1]))
    ax[2].set_ylim((img.shape[0], 0))
    ax[2].set_title(str)
    
    for a in ax:
        a.set_axis_off()
    
    plt.tight_layout()
    plt.show() 
    
    
def imageprocessingThreshold(img):
    ret,img_b = cv2.threshold(img.astype(np.uint8),
                              0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image = invert(img_b)
    
    skeleton = thin(image.reshape(img.shape[0], img.shape[1]))
    skeleton = skeleton.astype('uint8')
    
    
    return skeleton
 
    
def imageprocessingMeijering(img):
    from skimage.filters import meijering
    
    ret,img_b = cv2.threshold(img.astype(np.uint8),
                              0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
   
    image = meijering(img_b, sigmas=[2], alpha = 5)
        
    skeleton = thin(image.reshape(img.shape[0], img.shape[1]))
    skeleton = skeleton.astype('uint8')
     
    return skeleton
    
def slope(x1, y1, x2, y2): 
    if (x1 == x2):
        return 0
    else:
        return (y2-y1)/(x2-x1)


def computeSlopes(x,y):
    
    import math 
    
    slopE = []
    finalAngle=[]
    for i in range(len(x)):
        s = math.degrees(math.atan(slope(int(x[i][0]),
                                         int(x[i][1]),
                                         int(y[i][0]),
                                         int(y[i][1]))))
        if (s < 0) and (s > -180):
            slopE.append(s + 180)
        else:
            slopE.append(s)
       
    for i in range(len(slopE)):
    
    #finalAngle = 0
        if (0 <= slopE[i] < 22.5) or (157.5 <= slopE[i] <= 180):
            finalAngle.append(0)
    #finalAngle = 45
        elif (22.5 <= slopE[i] < 67.5):
            finalAngle.append(45)
    #finalAngle = 90
        elif (67.5 <= slopE[i] < 112.5):
            finalAngle.append(90)
    #finalAngle = 135
        elif (112.5 <= slopE[i] < 157.5):
            finalAngle.append(135)
            
    return finalAngle

def computeDistance(x,y):
    distance = []
    
    for i in range(len(x)):
        distance.append(np.sqrt(pow(int(x[i][0]) - int(x[i][1]), 2) + pow(int(y[i][0]) - int(y[i][1]),2)))
        
    return distance

def computeProbabilistic(img):
    from skimage.transform import probabilistic_hough_line
    lines =  probabilistic_hough_line(img, threshold=10, 
                                      line_length=3, line_gap=5)
    return lines

def imagesFeatures(path):
    img = cv2.imread(path, 0)
    
    skeleton = imageprocessingMeijering(img)
    skeleton2 = imageprocessingThreshold(img)
      
    import math
    
    result = img.copy()
    result = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        
    x = []
    y = []
    
    lines =  computeProbabilistic(skeleton)
    lines1 = computeProbabilistic(skeleton2)
    
    
    for i in range(len(lines)):
        point1 = lines[i][0]
        point2 = lines[i][1]
            
        x.append(point1)
        y.append(point2)
        
    finalAngle = computeSlopes(x,y)
        
    #vector cu toate unghiurile care au grade specifice: 0, 45, 90, 135 
    degrees = []
    degrees.extend([finalAngle.count(0), finalAngle.count(45), 
                    finalAngle.count(90), finalAngle.count(135)])
    
    #vector cu procentajele 
    percentageParallel = []
    percentageParallel.extend([finalAngle.count(0)*100/len(lines), 
                               finalAngle.count(45)*100/len(lines), 
                               finalAngle.count(90)*100/len(lines), 
                               finalAngle.count(135)*100/len(lines)])
    
    #lungimea liniilor
    
    distance = computeDistance(x,y);  
    average = sum(distance)/len(distance) 
     
    counterLong = 0
    counterShort = 0
    for i in range(len(distance)):
        if (distance[i] > average):
            counterLong +=1
        else: 
            counterShort +=1   
    
    features = []
    
    #total lines, long lines, short lines, parallel lines, percentage of long line, and percentage of parallel lines
    features.append(len(lines))        
    features.append(counterLong)
    features.append(counterShort)
    features.append(degrees)
    features.append(percentageParallel)
    features.append(counterLong*100/len(lines))
    
    showImages(img, skeleton, lines, 'no filter - img 1')
    showImages(img, skeleton2, lines1, 'meijering - img 1' )
    return features

path = r'D:\_WorkTmpFilesTst\08_07\Clasa3\3_18.jpg'   
features = imagesFeatures(path)


 #comparatie intre imagini
    #vector care descriu imaginea
    #clasificator randomtree
    
    # showImages(img, skeleton, lines, 'no filter - img 1')
    # showImages(img, skeleton2, lines1, 'meijering - img 1' )
    
    #showImages(img2, skeleton3, lines11, 'no filter - img 2')
    #showImages(img2, skeleton4, lines12, 'meijering - img 2' )
    
    #showImages(img3, skeleton5, lines21, 'no filter - img 3')
    #showImages(img3, skeleton6, lines22, 'meijering - img 3' )
    
    
    #plt.hist(distance)
    #plt.show()
# save resulting images
#cv2.imwrite('fabric_equalized_thresh.jpg',img_b)
# minLineLength = 10
# maxLineGap = 13

# lines = cv2.HoughLinesP(skeleton,1,np.pi/180, 2,minLineLength,maxLineGap)

# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

# cv2.imwrite('houghlines.jpg',img)

# fig = plt.figure(figsize=(30,6))
# #ax = fig.add_subplot(132); ax.imshow(image, cmap=plt.cm.bone)
# ax = fig.add_subplot(131); ax.imshow(skeleton, cmap=plt.cm.bone); #plt.show()
# ax = fig.add_subplot(133); ax.imshow(result, cmap=plt.cm.bone)