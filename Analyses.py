# -*- coding: utf-8 -*-
"""
@author: covaciu
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
#command for plotting: %matplotlib inline

# In[1] - 
      
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
    import glob
    x = []
    y = []
    clsLabels = np.array(['etyn','etym', 'etyo'])
    
    for i in range(0,len(clsLabels)):
        filesCls = glob.glob('D:/Licenta/Analize-All/*/*'+ clsLabels[i]+".jpg")
        for k in range (0, len(filesCls)):
            img = cv2.imread(filesCls[k], 0)
            imgFeat1D = np.reshape(img,(img.shape[0]*img.shape[1],))
            x.append(imgFeat1D)
            y.append(i)
  
    y = np.asarray(y)
    X = np.asarray(x)
    
#    print (X)
#    print (y)
    print('X.shape: ',X.shape,  ' y.shape: ',y.shape)
    return [X, y]

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
    hist_values = cv2.calcHist([img], channels=[0],mask=None,histSize=[256],ranges=[0,256])
    plt.plot(hist_values)

    return 
# In[1] - Prgram principal            
orgPath = r'D:\Licenta\Analize-All\5.Aprilie-All'  
    
[X, y] = get_1DImages(orgPath)
