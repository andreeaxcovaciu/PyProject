# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:49:24 2020

@author: covac
"""


import numpy as np
import image_slicer
 
import os


#from PIL import Image

def slice(path):
    clsLabels = np.array(['etyn','etym', 'etyo'])
    import glob   
    for i in range(0,len(clsLabels)):
        filesCls = glob.glob(path +'/*/*' + clsLabels[i]+".jpg")
        for k in range (0, len(filesCls)):
            tiles = image_slicer.slice(filesCls[k], 25, save = False)
            image_slicer.save_tiles(tiles, directory = r'C:\Users\covac\Desktop\img\All_tiles_in1', prefix = filesCls[k], format = 'jpeg')
            import shutil
            shutil.move(filesCls[k], r'C:\Users\covac\Desktop\img\All_pics_1' + os.path.basename(filesCls[k]))           
         
path = r'C:\Users\covac\Desktop\img'          
slice(path)
            