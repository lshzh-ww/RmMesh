from matplotlib.pyplot import sci
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

def loadDA30scan(self,filename):
    data=np.fromfile(filename,dtype=np.float32,sep="")
    print(np.shape(data))
    data=np.reshape(data,(61,799,1005))
    #data=np.swapaxes(data,0,2)
    print(np.shape(data))
    return data[2:59,:,:]

def loadFile(self,filename):
    if filename[len(filename)-3:]=='mat':
        M=matFile(self,filename)
        return M
    if filename[len(filename)-3:]=='npy':
        M=np.load(filename)
        return M
    if filename[len(filename)-3:]=='bin':
        M=loadDA30scan(self,filename)
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

def saveFile(self,loadFilename,saveFilename,saveData):
    if saveFilename[len(saveFilename)-3:]=='mat':
        saveMatFile(self,loadFilename,saveFilename,saveData)
    if saveFilename[len(saveFilename)-3:]=='npy':
        np.save(saveFilename,saveData)

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