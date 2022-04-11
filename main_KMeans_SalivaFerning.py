import matplotlib as plt
import itertools
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
Xpa = np.load(r'D:\_WorkTmpFilesTst\Disertatie\set_29_03_TOATE.npy')

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

# In[1] Descriptori de textura - ferigarea 
# Extragerea vectorului de trasaturi
import cv2
from skimage.filters import frangi
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
#from PIL import Image

desc = LocalBinaryPatterns(26, 8)

from sklearn.cluster import KMeans
md_km_q = KMeans(n_clusters = 4, random_state=0)


def dataFeature(img, desc, af):
    
    #applying binarization
    ret,img_bn = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    img_fr = frangi(img_bn.reshape(img.shape[0], img.shape[1]), scale_step = 1, gamma = 1.5, beta = 1.5) #frangi filter

    img_fr_n = 255 * (img_fr -np.min(img_fr))/(np.max(img_fr) - np.min(img_fr))
    
    
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
# In[2] Aplicare KMeans - creare model, antrenare pe setul de date, extragere centre clase

#setting the number classes
nrClase = 60

#calling the KMeans function
model_kmeans = KMeans(n_clusters = nrClase, random_state=0)

ypaft_labels = model_kmeans.fit_predict(Xpaft)

centers_kmeans = model_kmeans.cluster_centers_ # centrele claselor

#saving the centroids
np.save("npyData/07_07/centers_kmeans_Xpa_Andreea_k_Disertatie.npy", centers_kmeans)

print(' KMeans - cluster centers')
print('     - kmeans.cluster_centers_.shape', model_kmeans.cluster_centers_.shape)

#saving the extracted patches in their classes
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
# In[3] Antrenare BoW pe date a caror eticheta o cunoastem 
from sklearn.feature_extraction.image import extract_patches_2d
# save np.load

X = np.load(r'D:\_WorkTmpFilesTst\Disertatie\set_29_03_X.npy', allow_pickle=True)
y = np.load(r'D:\_WorkTmpFilesTst\Disertatie\set_29_03_y.npy', allow_pickle=True)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

Xlb = X_train
ylb = y_train



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


# In[4]

#implementing SVM
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


# In[5]

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



import pickle


print('Salvare imagini dupa clasa: ...')

saveImgsCls = True # daca dorim sa salvam pache-urile dupa grupare KMeans
if (saveImgsCls == True):
    # gruparea imaginilor dupa clasele alocate
    path = firstPath+'SalivaImgsAnl/Clasificare_Again/' # stocare locala
    folders = [] # definiere cate un folder pentru fiecare clasa
    for i in range(nrCls):
        folders.append("clasasvm"+ str(i))
    
    Daca directorul deja exista - se sterge si reface
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
        
 
print("--- %s seconds ---" % (time.time() - start_time))
<<<<<<< Updated upstream

=======
      
>>>>>>> Stashed changes
