# -*- coding: utf-8 -*-
"""
@author: covaciu
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
#command for plotting: %matplotlib inline

      
def get_imlist(path):
  """  Returns a list of filenames for
    all jpg images in a directory. """
  Files = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
  return print('\n'.join(Files))
   
def get_1DImages(path):
    """
    Name
    ----------
    get_1DImages(path)
    
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
               
                img = cv2.imread(fileName,0)
                imgFeat1D = np.reshape(img,(img.shape[0]*img.shape[1],))
                x.append(imgFeat1D)
               
                if (fileName.endswith("etyn.jpg")):
                    y.append("etyn")
                elif (fileName.endswith("etym.jpg")):
                    y.append("etym")
                elif (fileName.endswith("etyo.jpg")):
                    y.append("etyo")

    y = np.asarray(y)
    X = np.asarray(x)
    
#    print (X)
#    print (y)
    print('X.shape: ',X.shape,  ' y.shape: ',y.shape)
    return

def histogram(filename):     
    """
    Name
    -------
    histogram()
    
     
    Parameters
    ----------
    filename : The image you want 
    the histogram for
    
    Returns
    -------
    The histogram of an image.

    """

    img = cv2.imread(filename,0)
    hist_values = cv2.calcHist(img, channels=[0],mask=None,histSize=[256],ranges=[0,256])
    plt.plot(hist_values)

    return 
           
                    
get_1DImages(r'C:\Users\covac\Desktop\Analize-All\2.Ianuarie-All')
histogram(r'C:\Users\covac\Desktop\Analize-All\2.Ianuarie-All\zi30_1_AC_etyn.jpg')
