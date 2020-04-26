# -*- coding: utf-8 -*-
"""
@author: covaciu
"""

import os
import numpy as np
import cv2

      
def get_imlist(path):
  """  Returns a list of filenames for
    all jpg images in a directory. """
  Files = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
  return print('\n'.join(Files))
   
def get_1DImages(path):
    """

    Parameters
    ----------
    path : The directory
    you want to use
    -------
    Prints the images' matrix
    and the images' labels

    """
    x = []
    y = []

    dirFiles = os.listdir(path)      #all filenames of that particular dir -- image
    
    for k in range(0,len(dirFiles)):
          
            fileName = path+"\\"+dirFiles[k]
            
            if (os.path.isfile(fileName)):
               
                imgFeat1D = cv2.imread(fileName)
                x.append(imgFeat1D)
               
                if (fileName.endswith("etyn.jpg")):
                    y.append("etyn")
                elif (fileName.endswith("etym.jpg")):
                    y.append("etym")
                elif (fileName.endswith("etyo.jpg")):
                    y.append("etyo")

    Y = np.asarray(y)
    X = np.asarray(x)
    
    print (X)
    print (Y)
    return 
           
                    
get_1DImages(r'C:\Users\covac\Desktop\Analize-All\2.Ianuarie-All')