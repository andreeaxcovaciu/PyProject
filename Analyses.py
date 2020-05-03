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

#import glob
# In[1] - Prgram principal            
orgPath = r'D:\Licenta\Analize-All\2.Ianuarie-All'   
clsLabels = {'etyn','etym', 'etyo'}        

# dirFiles = os.listdir(orgPath)

# dirsPath = orgPath + dirFiles[0]

# imgsFile = os.listdir(dirsPath)


     
[X, y] = get_1DImages(orgPath)
# histogram(r'C:\Users\covac\Desktop\Analize-All\2.Ianuarie-All\zi10_1_AC_etym.jpg')


from sklearn.model_selection import train_test_split
# divizare date in set de antrenare si set de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
print(' Setul de date este impartit: ', X_train.shape[0],' train, ',  X_test.shape[0],' test' )

# In[] SVM
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics # Making the Confusion Matrix
param_grid = {'kernel':[ 'rbf','poly','linear'], 'degree':[2,3,4],'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 
classifier = GridSearchCV(SVC(),param_grid,refit=True,verbose=1)

classifier.fit(X_train, y_train)
print(classifier.best_params_)

# nkernel =  'poly' #  'linear','poly', 'rbf'
# classifier = SVC(kernel = nkernel,degree=2 ,random_state = 0)
# classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predTrn = classifier.predict(X_train)
y_predTst = classifier.predict(X_test)

print('\n ---- Setul de antrenare ------- ')
print(' SVM - Accuracy: ', metrics.accuracy_score(y_train, y_predTrn))
print(' Confusion Matrix pe setul de antrenare: \n', metrics.confusion_matrix(y_train,y_predTrn))
print('\n ---- Setul de testare ------- ')
print(' SVM - Accuracy: ', metrics.accuracy_score(y_test, y_predTst))
print(' Confusion Matrix pe setul de testare: \n', metrics.confusion_matrix(y_test,y_predTst))

# In[]
# save the model to disk
import pickle
filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# some time later...

# load the model from disk
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
