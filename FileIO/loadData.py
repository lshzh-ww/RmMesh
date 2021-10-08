import scipy.io
import numpy as np
from PyQt5.QtWidgets import QInputDialog

def matFile(self,filename):
    data=scipy.io.loadmat(filename)
    allKeys=data.keys()
    text, ok = QInputDialog.getText(self, 'Which matrix?',str(allKeys))
    if ok:
        M = data.get(str(text))
    else:
        return None
    M=np.swapaxes(M,0,2)
    return M

def saveMatFile(self,loadFilename,saveFilename,saveData):
    data=scipy.io.loadmat(loadFilename)
    allKeys=data.keys()
    text, ok = QInputDialog.getText(self, 'Which matrix',str(allKeys))
    if ok:
        data[str(text)]=np.swapaxes(saveData,0,2)
    else:
        return None
    scipy.io.savemat(saveFilename,data)
    
    return 0

def loadSpacialScan(self,filename):
    data=scipy.io.loadmat(filename)
    allKeys=data.keys()
    text, ok = QInputDialog.getText(self, 'Which matrix?',str(allKeys))
    if ok:
        M = data.get(str(text))
    else:
        return None
    data=np.zeros((M.shape[1],M[0,0].shape[0],M[0,0].shape[1]))

    for i in np.arange(0,M.shape[1],1):
        data[i]=M[0,i]

    return data

def saveSpacialScan(self,loadFilename,saveFilename,saveData):
    data=scipy.io.loadmat(loadFilename)
    allKeys=data.keys()
    text, ok = QInputDialog.getText(self, 'Which matrix',str(allKeys))
    if ok:
        data[str(text)]=np.swapaxes(saveData,0,2)
    else:
        return None
    scipy.io.savemat(saveFilename,data)
    
    return 0