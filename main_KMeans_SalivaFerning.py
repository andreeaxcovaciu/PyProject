# -*- coding: utf-8 -*-
# import seaborn as sns; 
import matplotlib as plt
# def ImageIllustrationPredictionResults(X, y, y_pred, cls_labels): 
#     nrCls = len(cls_labels)
#     pozErr = (y != y_pred) # errors in label prediction
#     X_err = X [pozErr]
#     y_err_or = y[pozErr]
#     y_err_pr = y_pred[pozErr]
#     print("Set date/imagini eronate: ", X_err.shape)
    
#     lines = 4
#     cols = max(nrCls, 10)
#     fig = plt.figure(figsize=(cols*1.6,lines*2.2))
#     for i in range(0, min(lines*cols, X_err.shape[0])):
#         fig.add_subplot(lines, cols, i+1); plt.imshow(X_err[i], cmap=plt.cm.bone); 
#         plt.title([str(y_err_or[i]) + '/ p:' + str(y_err_pr[i])]);
#     plt.show()


# def ImagePlotConfusionMatrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#   """
#   This function prints and plots the confusion matrix.
#   Normalization can be applied by setting `normalize=True`.
#   """
#   if normalize:
#       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#       print("Normalized confusion matrix")
#   else:
#       print('Confusion matrix, without normalization')

#   # print(cm)

#   plt.imshow(cm, interpolation='nearest', cmap=cmap)
#   plt.title(title)
#   plt.colorbar()
#   tick_marks = np.arange(len(classes))
#   plt.xticks(tick_marks, classes, rotation=45)
#   plt.yticks(tick_marks, classes)

#   fmt = '.2f' if normalize else 'd'
#   thresh = cm.max() / 2.
#   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#       plt.text(j, i, format(cm[i, j], fmt),
#                horizontalalignment="center",
#                color="white" if cm[i, j] > thresh else "black")

#   plt.tight_layout()
#   plt.ylabel('True label')
#   plt.xlabel('Predicted label')
#   plt.show()


import itertools
# firstPath = "D:\_WorkTmpFilesTst\"
firstPath = "D:/_WorkTmpFilesTst/"
# -----------------------------------------------------------------------------------
# In[0] Incarcare baza de date 
import numpy as np
import matplotlib.pyplot as plt
from random import random
%matplotlib inline

import time
start_time = time.time()

# Vectorul de trasaturi Xpa - patch-uri extrase din tot setul de date
Xpa = np.load('npyData/07_07/set_all_patches_X.npy')

print(' Setul de patche-uri incarcat, Xpa.shape ', Xpa.shape)

# Afisare un numar de imagini, aleator
hp = Xpa.shape[1]; wp = Xpa.shape[2]
print(' Imaginile au rezolutia: ', hp ,'x', wp )
print(' Afisare imagini (random) din setul de date incarcat: ')
cols = 4; line = 4; pl = 1;
fig = plt.figure(figsize=(cols*1.2,line*1.8))
for i in range(0, line):
    for j in range (0, cols):
        fig.add_subplot(line,cols, (i*cols+j+1)); #pl=pl+1;
        rdmImg = int(random()* Xpa.shape[0])
        plt.imshow(Xpa[rdmImg], cmap='gray'); #plt.title(str([rdmImg]))
plt.show();

# In[] Descriptori de textura - ferigarea 
# Extragerea vectorului de trasaturi
import cv2
from skimage.filters import frangi
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
#from PIL import Image

desc = LocalBinaryPatterns(26, 8)

from sklearn.cluster import KMeans
md_km_q = KMeans(n_clusters = 4, random_state=0)



# centers_kmeans = model_kmeans.cluster_centers_ # centrele claselor


def dataFeature(img, desc, af):
    
    ret,img_bn = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    img_fr = frangi(img_bn.reshape(img.shape[0], img.shape[1]), scale_step = 1, gamma = 1.5, beta = 1.5) #frangi filter
    img_fr_n = 255 - ( img_fr*255/0.05).astype(np.uint8)
    
    
    if (af == True):
        if (i%2 == 0):
            fig = plt.figure(figsize=(30,6))
            ax = fig.add_subplot(131); ax.imshow(img_i, cmap=plt.cm.bone); #plt.show()
            ax = fig.add_subplot(132); ax.imshow(img_bn, cmap=plt.cm.bone); #plt.show()
            ax = fig.add_subplot(133); ax.imshow(img_fr_n, cmap=plt.cm.bone); #plt.show()
            
    imftr = desc.describe(img_fr_n)
    return imftr


ftl = 28 # fiecare imagine este descrisa prin ftl valori
img_fts = np.zeros((Xpa.shape[0],ftl))
# img_frX = np.zeros((Xpa.shape))

af = False
for i in range(0, Xpa.shape[0]): #X.shape[0]
    img_i = Xpa[i]
    imftr = dataFeature(img_i, desc, af)
    img_fts[i] = imftr

Xpaft = img_fts 
print('  - Interval original: [', Xpaft[0].min(), ' ; ', Xpaft[0].max(),']')
# In[3] Aplicare KMeans - creare model, antrenare pe setul de date, extragere centre clase


nrClase = 60
model_kmeans = KMeans(n_clusters = nrClase, random_state=0)

ypaft_labels = model_kmeans.fit_predict(Xpaft)

centers_kmeans = model_kmeans.cluster_centers_ # centrele claselor
np.save("npyData/07_07/centers_kmeans_Xpa_Andreea.npy", centers_kmeans)

print(' KMeans - cluster centers')
print('     - kmeans.cluster_centers_.shape', model_kmeans.cluster_centers_.shape)
savePatchesGroups = True # daca dorim sa salvam pache-urile dupa grupare KMeans
if (savePatchesGroups == True):
    # gruparea imaginilor dupa clasele alocate
    path = firstPath+'SalivaImgsAnl/Clasif_kmeans3/' # stocare locala
    folders = [] # definiere cate un folder pentru fiecare clasa
    for i in range(nrClase):
        folders.append("clasa"+ str(i))
    
    # Daca directorul deja exista - se sterge si reface
    import os
    if (os.path.isdir(path)):
        import shutil
        shutil.rmtree(path) # stergere folder cu toate subfolder-urile
        
    os.mkdir(path) # refacere director principal (gol)
    for folder in folders:
        os.mkdir(os.path.join(path,folder)) # creare director pe clasa
    
    # Salvare patch-uri dupa gruparea facura de KMeans
    # claseVals = []
    for i in range(0, ypaft_labels.shape[0]):
        dirname = path+'clasa'+str(ypaft_labels[i]) + '/' + str(i) + '.jpg'
        cv2.imwrite(dirname, Xpa[i, :, :])
        # claseVals.append(ypaft_labels[ypaft_labels==ypaft_labels[i]].shape)
    # print(claseVals)
# In[] Antrenare BoW pe date a caror eticheta o cunoastem 
from sklearn.feature_extraction.image import extract_patches_2d
# save np.load

X = np.load('npyData/07_07/set_analize_all_labeled_X.npy', allow_pickle=True)
y = np.load('npyData/07_07/set_analize_all_labeled_y.npy', allow_pickle=True)

X = X[y!=1]
y = y[y!=1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

Xlb = X_train
ylb = y_train

#Xlb = np.load(r'D:\_WorkTmpFilesTst\Toate/set_X2D_1_256_3clase_Toate.npy')
#ylb = np.load(r'D:\_WorkTmpFilesTst\Toate/set_y2D_1_256_3clase_Toate.npy')


from sklearn.neighbors import KNeighborsClassifier
model_kNN = KNeighborsClassifier(n_neighbors=1) 
X_train_kNN = centers_kmeans # definire dictionar - prototip
y_train_kNN = np.arange(0, X_train_kNN.shape[0])

 # fiecare centru din KMeans - are o alta eticheta 
model_kNN.fit(X_train_kNN, y_train_kNN) # antrenare model kNN pe setul de date

Xsvm = np.zeros((Xlb.shape[0], X_train_kNN.shape[0]-1))

patch_size = (wp, hp)
rng = np.random.RandomState(0)

for i in range (Xlb.shape[0]):
    dataSubImgs = extract_patches_2d(Xlb[i], patch_size , max_patches=120, random_state=rng)
    
    dsi_ft = np.zeros((dataSubImgs.shape[0],ftl))

    
    for j in range(0,dataSubImgs.shape[0]):
        img_j = dataSubImgs[j]
        imftr = dataFeature(img_j, desc, af)
        dsi_ft[j] = imftr 
     
    y_pred = model_kNN.predict(dsi_ft)
    hisy = np.histogram(y_pred, bins=y_train_kNN)
    Xsvm[i] = hisy[0]
 
    opt_par_svm = False; # True or False


# In[]

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
X_train_svm = Xsvm
y_train_svm = ylb
# definire/modelare clasificator

cls_labels = np.unique(ylb).astype(str);
nrCls = len(cls_labels);
print(' Numar clase: ', nrCls)
print(' Etichete: ', cls_labels)

nkernel =  'rbf'
model_svm = SVC(kernel = nkernel, random_state = 0)
model_svm.fit(X_train_svm, y_train_svm)
print(' Model SVM - kernel:', model_svm.kernel,', gamma:', model_svm.gamma,
      ', C: ', model_svm.C,', degree: ', model_svm.degree)



#antrenare clasificator folosind setul de date de antrenament
model_svm.fit(X_train_svm, y_train_svm)

# predictie pe setul de antrenare
y_train_pred = model_svm.predict(X_train_svm)
fig = plt.figure(figsize=(16, 6))

# Evaluare rezultate pe setul pe care s-a realizat antrenarea
from sklearn.metrics import confusion_matrix, classification_report
import _PyUtilFcts.UtilsPredictionResults as upr

cm_train = confusion_matrix(y_train_svm, y_train_pred)
upr.ImagePlotConfusionMatrix(cm_train, cls_labels)


print( " \n ------ Evaluare performante clasificator ------ ")

print(' \n   Classification report - train dataset : \n\n', classification_report(y_train_svm, y_train_pred))


#print(metrics.classification_report(cm_train, y_ia_pred))


# In[4]
#Xia = np.load('npyData/set_ALL_X.npy')

#Xia = np.load('npyData/07_07/set_ALL_X.npy')
#yia = np.load('npyData/06_07/set_2D_1_256_3clase_Toate.npy')
Xia = X_test
yia = y_test

print('Preprocesare date ... ')
Xiasvm = np.zeros((Xia.shape[0], X_train_kNN.shape[0]-1))
for i in range (Xia.shape[0]):
    dataSubImgs = extract_patches_2d(Xia[i], patch_size , max_patches=120, random_state=rng)
    dsi_ft = np.zeros((dataSubImgs.shape[0],ftl))
    for j in range(0,dataSubImgs.shape[0]):
        img_j = dataSubImgs[j] # plt.imshow(img_j); plt.show()
        imftr = dataFeature(img_j, desc, af)
        dsi_ft[j] = imftr 
    y_pred = model_kNN.predict(dsi_ft)
    hisy = np.histogram(y_pred, bins=y_train_kNN)
    Xiasvm[i] = hisy[0]

print ('Predictie clasa ...')
# predictie pe setul de test

y_ia_pred = model_svm.predict(Xiasvm)
y_labels = np.unique(y_train_svm)

from sklearn import metrics
ylb = ylb.astype(int)

fig = plt.figure(figsize=(16, 6))

cm = metrics.confusion_matrix(yia,y_ia_pred)
upr.ImagePlotConfusionMatrix(cm, cls_labels,  title='Validation confusion matrix')
upr.ImageIllustrationPredictionResults(Xia, yia, y_ia_pred, cls_labels)
print(' \n   Classification report - test dataset : \n\n', classification_report(yia, y_ia_pred))



#salvare model

import pickle


# filename = 'finalized_SVM_model_2classes.sav'
# pickle.dump(model_svm, open(r'npyData/07_07/'+filename, 'wb'))

print('Salvare imagini dupa clasa: ...')

saveImgsCls = True # daca dorim sa salvam pache-urile dupa grupare KMeans
if (saveImgsCls == True):
    # gruparea imaginilor dupa clasele alocate
    path = firstPath+'SalivaImgsAnl/Clasif_Svm3/' # stocare locala
    folders = [] # definiere cate un folder pentru fiecare clasa
    for i in range(nrCls):
        folders.append("clasasvm"+ str(i))
    
    # Daca directorul deja exista - se sterge si reface
    import os
    if (os.path.isdir(path)):
        import shutil
        shutil.rmtree(path) # stergere folder cu toate subfolder-urile
        
    os.mkdir(path) # refacere director principal (gol)
    for folder in folders:
        os.mkdir(os.path.join(path,folder)) # creare director pe clasa
    
    # Salvare patch-uri dupa gruparea facura de KMeans
    # claseVals = []
    for i in range(0, y_ia_pred.shape[0]):
        dirname = path+'clasasvm'+str(y_ia_pred[i]) + '/' + str(i) + '.jpg'
        cv2.imwrite(dirname, Xia[i])
        
                # claseVals.append(y_ia_pred[y_ia_pred==y_ia_pred[i]].shape)
    # print(claseVals)
 
print("--- %s seconds ---" % (time.time() - start_time))
      
#  # Afisare centrelor claselor KMeans
# if ('images' in dsXy.keys()): 
#     print('     - digits.images.shape', dsXy.images.shape)
#     centers_image = model_kmeans.cluster_centers_.reshape(model_kmeans.n_clusters, dsXy.images.shape[1], dsXy.images.shape[2])
    
#     fig, ax = plt.subplots(1, model_kmeans.n_clusters, figsize=(model_kmeans.n_clusters*1.6, 3))
    
#     for axi, center in zip(ax.flat, centers_image):
#         axi.set(xticks=[], yticks=[])
#         axi.imshow(center, cmap=plt.cm.bone)
#     plt.show()
# In[4] Evaluare rezultate

# from scipy.stats import mode

# # se coreleaza labelurile returnate de KMeans cu cele reale
# labels = np.zeros_like(clusters)
# for i in range(model_kmeans.n_clusters):
#     mask = (clusters == i)
#     labels[mask] = mode(y[mask])[0] # returneaza cele mai frecvente valori dintr-un vector

# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# print( " \n ------ Evaluare performante clasificator ------ ")
# # print( "accuracy: ", accuracy_score(dsXy.target, labels))
# cm = confusion_matrix(y, labels)
# fig = plt.figure(figsize=(16, 6))
# ImagePlotConfusionMatrix(cm, cls_labels, "CM")
# print('    Classification report: \n\n', classification_report(y, labels))

# # In[] Optizare descriere date prin aplicare TSNE 

# from sklearn.manifold import TSNE

# # Project the data: this step will take several seconds
# tsne = TSNE(n_components=2, init='random', random_state=0)
# X_tsne = tsne.fit_transform(X)

# print("\n*********************************************************\n")
# print( " DUPA aplicare TSNE !!!")
# print(" Vectorul de trasaturi, X_tsne.shape: ", X_tsne.shape)

# # Compute the clusters
# model_kmeans_tnse = KMeans(n_clusters= nrCls, random_state=0)
# clusters = model_kmeans_tnse.fit_predict(X_tsne)
# print(clusters[clusters==0].shape,clusters[clusters==1].shape,clusters[clusters==2].shape)
# # Permute the labels
# labels = np.zeros_like(clusters)
# for i in range(10):
#     mask = (clusters == i)
#     labels[mask] = mode(y[mask])[0]

# print( " \n ------ Evaluare performante clasificator ------ ")
# # Compute the accuracy
# # print( "accuracy: ", accuracy_score(dsXy.target, labels))
# cm = confusion_matrix(y, labels)
# fig = plt.figure(figsize=(16, 6))
# ImagePlotConfusionMatrix(cm, cls_labels, "CM, optimizat")
# print('    Classification report: \n\n', classification_report(y, labels))
