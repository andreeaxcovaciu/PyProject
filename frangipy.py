# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:19:47 2020

@author: covac
"""


import matplotlib.pyplot as plt
from skimage.filters import frangi, hessian
import cv2



roi = 256
# firstPath = "D:/_WorkTmpFilesTst/"
imagefile = r'../_DataSet_ImgsSalivaFerning/Analize_All_Andreea/2.Ianuarie-All\zi09_2_AC_etym.jpg'
image_org = cv2.imread(imagefile,0)
[H, W] = image_org.shape
image_org = image_org[int((H-roi)/2):int((H+roi)/2),int((W-roi)/2):int((W+roi)/2)]
image_eq = cv2.equalizeHist(image_org)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
image_cl = clahe.apply(image_org)

# from sklearn.cluster import KMeans
# md_km_q = KMeans(n_clusters = 4, random_state=0)

# labels = md_km_q.fit_predict(image_org.ravel().reshape(-1, 1)) 
# img_quant = md_km_q.cluster_centers_.astype("uint8")[labels]
# img_quant = img_quant.reshape(image_org.shape[0], image_org.shape[1])
img_quant = 255- (image_cl * (2**5/255)).astype("uint8")

# frangi
image_fr = 255 - frangi(img_quant) #scale_step = 1, beta1 = 1, beta2 = 6.5

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(321)
ax.imshow(image_org, cmap=plt.cm.gray)
ax.set_title('Original image')

ax = fig.add_subplot(322)
ax.imshow(image_eq, cmap=plt.cm.gray)
ax.set_title('equalizeHist result')

ax = fig.add_subplot(323)
ax.imshow(image_cl, cmap=plt.cm.gray)
ax.set_title('clahe result')

ax = fig.add_subplot(324)
ax.imshow(image_fr, cmap=plt.cm.gray)
ax.set_title('Frangi filter result')

ax = fig.add_subplot(325)
ax.imshow(hessian(image_fr), cmap=plt.cm.gray)
ax.set_title('Hybrid Hessian filter result')

ax = fig.add_subplot(326)
ax.imshow(img_quant, cmap=plt.cm.gray)
ax.set_title('img_quant')


# for a in ax:
#     a.axis('off')

plt.tight_layout()
plt.show()
