# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:55:08 2020

@author: covac
"""
import cv2
import numpy as np
import os
from sklearn.feature_extraction.image import extract_patches_2d
from matplotlib import pyplot as plt
from random import random

#version2
def get_1DImages(path):
   
    path2 = 'D:/_WorkTmpFilesTst/3Clase'
    x = []
    y = []
    #folders = np.array(['et0', 'et1','et2', 'et3 ', 'et4'])
   # folders = np.array([ 'clasasvm0','clasasvm1', 'clasasvm2', 'clasasvm3', 'clasasvm4'])
    #folders =np.array(['Clasa0' , 'Clasa1', 'Clasa2', 'Clasa3', 'Clasa4'])
    #folders = np.array(['1.Decembrie-All','2.Ianuarie-All', '3.Februarie-All', '4.Martie-All', '5.Aprilie-All', '6.Mai-All', '7.Iunie-All'])
    folders =np.array(['Clasa1',  'Clasa2', 'Clasa3'])
    import glob   
    roi = 256
    
    # if (os.path.isdir(path2)):
    #     import shutil
    #     shutil.rmtree(path2)
        
    # os.mkdir(path2) # refacere director principal (gol)
    # for folder in folders:
    #     os.mkdir(os.path.join(path2,folder))
        
     # filename = os.rename(path +folders[i] + '.jpg', path +folders[i] + str(i) + '.jpg' )
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i in range(0,len(folders)):
        
        filesCls = glob.glob(path+'/*'+ folders[i]+'/*.jpg')
      #  print(filesCls)
        print(folders)
        for k in range (0, len(filesCls)):
          
            img1 = cv2.imread(filesCls[k],0)
           
            #img = clahe.apply(img1)
            
            [H, W] = img1.shape
            img_roi = img1[int((H-roi)/2):int((H+roi)/2),int((W-roi)/2):int((W+roi)/2)]
            
            #imgFeat1D = np.reshape(img,(img.shape[0]*img.shape[1],))
           # dirname = path2+str(folders[i]) + '/' +str(i)+ '_'+ str(k) + '.jpg'
            #cv2.imwrite(dirname, img_roi)
            

            x.append(img_roi)
            y.append(i)
    
    y = np.asarray(y)
    X = np.asarray(x)
    # cols = 4; line = 4; pl = 1;
    # fig = plt.figure(figsize=(cols*2,line*2))
    # for i in range(0, line):
    #     for j in range (0, cols):
           
    #         fig.add_subplot(line,cols, (i*cols+j+1)); #pl=pl+1;
    #         rdmImg = int(random()* X.shape[0])
    #         plt.imshow(X[rdmImg], cmap='gray');# plt.title(str(i) + str(i) )
    #         plt.show();
    
    print('X.shape: ',X.shape,  ' y.shape: ',y.shape)
    return [X, y]

#orgPath = r'D:\Technical University of Cluj-Napoca\Camelia Florea - L20_AndreeaCovaciu\Clasificare-Actualizare'       
#orgPath = r'D:\Licenta\All pics\Andreea' 
#orgPath = r'C:\Users\covac\Downloads\_WorkTmpFilesTst'
orgPath = r'D:\_WorkTmpFilesTst\08_07'
[X, y] = get_1DImages(orgPath)

#from numpy import savez_compressed
#(X_train, y_train) = [X, y]
#save_data = [X, y]
np.save(r"D:\_WorkTmpFilesTst\08_07\set_X2D_1_256_08_07_2clase_taken.npy", X)
np.save(r"D:\_WorkTmpFilesTst\08_07\set_y2D_1_256_08_07_2clase_taken.npy", y)
#np.save(r"D:\_WorkTmpFilesTst\Toate\set_y2D_1_256_ToateSVM3.npy", y)