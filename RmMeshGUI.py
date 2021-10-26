import FileIO.loadData
import colorMap
import CustomWidgets.plot
import CompFunc.fft
import threading
from multiprocessing import Pool,cpu_count

import sys
import numpy as np
from math import log,isnan
import scipy as sp
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QFileDialog, QWidget, QInputDialog, QHBoxLayout, QFrame, QSplitter, QPushButton, QGridLayout
from PyQt5.QtGui import QIcon, QPainter, QColor, QFont, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
from pathlib import Path
from time import time



class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.status=None
        self.radius=1200
    
    def initUI(self):
        #quit 
        exitAct = QAction(QIcon(), '&Exit', self)
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)
        #load file
        loadFileAct = QAction(QIcon(),'Load...',self)
        loadFileAct.triggered.connect(self.loadFileDiag)
        #save file
        saveFileAct = QAction(QIcon(),'Save...',self)
        saveFileAct.triggered.connect(self.saveFileDiag)

        #restoreCurve
        restoreCurveAct=QAction(QIcon(),'Restore Curvature',self)
        restoreCurveAct.triggered.connect(self.restoreCurvatureFunc)

        #switch XY
        switchXY=QAction(QIcon(),'Switch X-Y',self)
        switchXY.triggered.connect(self.switchXYFunc)

        #test func
        testAct=QAction(QIcon(),'Optimize mesh template',self)
        testAct.triggered.connect(self.testFunc)

        #crop data
        cropDataAct=QAction(QIcon(),'Crop data',self)
        cropDataAct.triggered.connect(self.cropData)

        #set menu style
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)
        fileMenu.addAction(loadFileAct)
        fileMenu.addAction(saveFileAct)
        dataMenu = menubar.addMenu('&Data')
        dataMenu.addAction(switchXY)
        dataMenu.addAction(restoreCurveAct)
        dataMenu.addAction(testAct)
        dataMenu.addAction(cropDataAct)
        menubar.setNativeMenuBar(False)
        
        #set layout
        self.wid=QWidget(self)
        self.setCentralWidget(self.wid)
        hbox = QHBoxLayout(self.wid)

        topLeft = QFrame(self.wid)
        topLeft.setFrameShape(QFrame.StyledPanel)
        topRight = QFrame(self.wid)
        topRight.setFrameShape(QFrame.StyledPanel)
        bottomLeft = QFrame(self.wid)
        bottomLeft.setFrameShape(QFrame.StyledPanel)
        bottomRight = QFrame(self.wid)
        bottomRight.setFrameShape(QFrame.StyledPanel)

        splitter1 = QSplitter(Qt.Horizontal)
        splitter1.addWidget(topLeft)
        splitter1.addWidget(topRight)

        splitter3 = QSplitter(Qt.Horizontal)
        splitter3.addWidget(bottomLeft)
        splitter3.addWidget(bottomRight)
        splitter3.setSizes([737,738])

        splitter2 = QSplitter(Qt.Vertical)
        splitter2.addWidget(splitter1)
        splitter2.addWidget(splitter3)
        splitter2.setSizes([627,173])

        hbox.addWidget(splitter2)
        self.wid.setLayout(hbox)



        self.leftRemoveMeshButton=QPushButton('Remove Mesh')
        self.leftSelectPeakButton=QPushButton('Select peak')
        self.leftUndoButton=QPushButton('Undo')
        self.leftFinishButton=QPushButton('Finish')
        self.leftMeshPattern=QPushButton('Mesh Pattern')

        leftGrid=QGridLayout()
        leftGrid.setSpacing(10)
        leftGrid.addWidget(self.leftRemoveMeshButton,1,1)
        leftGrid.addWidget(self.leftMeshPattern,1,4)
        leftGrid.addWidget(self.leftSelectPeakButton,2,2)
        leftGrid.addWidget(self.leftUndoButton,2,3)
        leftGrid.addWidget(self.leftFinishButton,3,4)
        
        bottomLeft.setLayout(leftGrid)
        
        self.rightFixCurveButton=QPushButton('Fix Curvature')
        self.rightSelectPeakButton=QPushButton('Radius +100')
        self.rightUndoButton=QPushButton('Radius -100')
        self.rightFinishButton=QPushButton('Finish')

        rightGrid=QGridLayout()
        rightGrid.setSpacing(10)
        rightGrid.addWidget(self.rightFixCurveButton,1,1)
        rightGrid.addWidget(self.rightSelectPeakButton,2,2)
        rightGrid.addWidget(self.rightUndoButton,2,3)
        rightGrid.addWidget(self.rightFinishButton,3,4)
        
        bottomRight.setLayout(rightGrid)

        self.leftRemoveMeshButton.clicked.connect(self.startRemoveMesh)
        self.leftSelectPeakButton.clicked.connect(self.leftSelectedRegion)
        self.leftUndoButton.clicked.connect(self.leftUndoFunc)
        self.leftMeshPattern.clicked.connect(self.leftShowMesh)
        self.leftFinishButton.clicked.connect(self.leftFinish)

        self.rightFixCurveButton.clicked.connect(self.startFixCurvature)
        self.rightSelectPeakButton.clicked.connect(self.rightRadiusUp)
        self.rightUndoButton.clicked.connect(self.rightRadiusDown)
        self.rightFinishButton.clicked.connect(self.rightFinish)

        self.leftGraphWidget = CustomWidgets.plot.myImageView(topLeft,view=pg.PlotItem())
        self.leftGraphWidget.show()
        self.rightGraphWidget = CustomWidgets.plot.myImageView(topRight,view=pg.PlotItem())
        self.rightGraphWidget.show()
        self.leftGraphWidget.linkSlider(self.rightGraphWidget)
        self.rightGraphWidget.linkSlider(self.leftGraphWidget)
        self.leftGraphWidget.view.invertY(False)
        self.rightGraphWidget.view.invertY(False)
        self.rightGraphWidget.view.getViewBox().setAspectLocked(lock=False)

        
        self.leftSelectPeakButton.setEnabled(False)
        self.leftUndoButton.setEnabled(False)
        self.leftFinishButton.setEnabled(False)
        self.leftMeshPattern.setEnabled(False)
        self.rightSelectPeakButton.setEnabled(False)
        self.rightUndoButton.setEnabled(False)
        self.rightFinishButton.setEnabled(False)
        

        self.setGeometry(100, 100, 1475, 800)
        self.setWindowTitle('Remove Mesh')
        self.show()

    def loadFileDiag(self):
        home_dir = str(Path.home())
        self.fname = QFileDialog.getOpenFileName(self, 'Open file', home_dir)
        #self.rawData=FileIO.loadData.loadSpacialScan(self,self.fname[0])
        self.rawData=FileIO.loadData.matFile(self,self.fname[0])
        dataShape=np.shape(self.rawData)
        print(dataShape)
        self.displayData=self.rawData.copy()
        self.leftGraphWidget.setImage(self.displayData)
        self.fftData=CompFunc.fft.dataToFft(self.displayData)
        self.absFftData=CompFunc.fft.fftToAbs(self.fftData)
        self.rightGraphWidget.setImage(self.absFftData)
        self.rawIntensity=np.zeros(np.size(self.rawData,0))
        for i in np.arange(0,np.size(self.rawData,0),1):
            self.rawIntensity[i]=np.mean(self.rawData[i])
    
    def cropData(self):
        text, ok = QInputDialog.getText(self, 'Data Range','Xmin,Xmax,Ymin,Ymax')
        paraList=text.split(',')
        self.rawData=self.rawData[:,int(paraList[0]):int(paraList[1]),int(paraList[2]):int(paraList[3])]
        dataShape=np.shape(self.rawData)
        print(dataShape)
        self.displayData=self.rawData.copy()
        self.leftGraphWidget.setImage(self.displayData)
        self.fftData=CompFunc.fft.dataToFft(self.displayData)
        self.absFftData=CompFunc.fft.fftToAbs(self.fftData)
        self.rightGraphWidget.setImage(self.absFftData)
        self.rawIntensity=np.zeros(np.size(self.rawData,0))
        for i in np.arange(0,np.size(self.rawData,0),1):
            self.rawIntensity[i]=np.mean(self.rawData[i])

    def saveFileDiag(self):
        home_dir = str(Path.home())
        saveFilename = QFileDialog.getSaveFileName(self, 'Save File', home_dir)
        print(saveFilename)
        FileIO.loadData.saveMatFile(self, self.fname[0], saveFilename[0],self.displayData)

    def switchXYFunc(self):
        self.rawData=np.swapaxes(self.rawData,1,2)
        dataShape=np.shape(self.rawData)
        print(dataShape)
        self.displayData=self.rawData.copy()
        self.leftGraphWidget.setImage(self.displayData)
        self.fftData=CompFunc.fft.dataToFft(self.displayData)
        self.absFftData=CompFunc.fft.fftToAbs(self.fftData)
        self.rightGraphWidget.setImage(self.absFftData)
        self.rawIntensity=np.zeros(np.size(self.rawData,0))
        for i in np.arange(0,np.size(self.rawData,0),1):
            self.rawIntensity[i]=np.mean(self.rawData[i])

    def restoreCurvatureFunc(self):
        self.restoredData=np.zeros(np.shape(self.rawData))
        for i in np.arange(0,np.size(self.displayData,0),1):
            self.restoredData[i]=CompFunc.fft.restoreCurvature2D(self.displayData[i],self.radius)
        self.displayData=self.restoredData
        self.leftGraphWidget.setImage(self.displayData)
        self.finalMesh=np.zeros(np.shape(self.meshPattern))
        self.finalMesh[0]=CompFunc.fft.restoreCurvature2D(self.meshPattern[0],self.radius)



    def startRemoveMesh(self):
        self.status='RemoveMesh'
        self.selectedRegion=[]
        self.leftSelectPeakButton.setEnabled(True)
        self.leftUndoButton.setEnabled(True)
        self.avgRawData=np.zeros((1,np.shape(self.rawData[0])[0],np.shape(self.rawData[0])[1]))
        for i in np.arange(0,np.size(self.rawData,0),1):
            if isnan(np.sum(self.rawData[i])):
                print('image ' + str(i+1)+" contains NaN.")
                for index_1 in range(np.size(self.rawData,1)):
                    for index_2 in range(np.size(self.rawData,2)):
                        if isnan(self.rawData[i,index_1,index_2]):
                            self.rawData[i,index_1,index_2]=0
            self.avgRawData[0] = self.avgRawData[0] + self.rawData[i]
        self.avgRawData = self.avgRawData/np.size(self.rawData,0)
        self.leftGraphWidget.setImage(self.avgRawData)
        self.fftData=CompFunc.fft.dataToFft(self.avgRawData)
        self.absFftData=CompFunc.fft.fftToAbs(self.fftData)
        self.rightGraphWidget.setImage(self.absFftData)
        self.rightGraphWidget.view.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        print(self.rightGraphWidget.view.getViewBox().viewRange())

    def leftSelectedRegion(self):
        self.leftDisplay='ReconstructData'
        
        region=self.rightGraphWidget.view.getViewBox().viewRange()
        self.selectedRegion.append(region)
        print(self.selectedRegion[len(self.selectedRegion)-1])
        CompFunc.fft.maskFftData(self.fftData,region)
        self.absFftData=CompFunc.fft.fftToAbs(self.fftData)
        self.rightGraphWidget.setImage(self.absFftData)
        self.displayData=CompFunc.fft.fftToData(self.fftData,self.rawIntensity)

        self.leftGraphWidget.setImage(self.displayData)
        self.leftMeshPattern.setEnabled(True)

    def leftShowMesh(self):
        if self.leftDisplay=='ReconstructData':
            self.meshPattern=CompFunc.fft.getRatio(self.avgRawData,self.displayData)
            self.leftGraphWidget.setImage(self.meshPattern)
            self.leftGraphWidget.setLevels(min=0,max=3.)
            self.leftDisplay='MeshPattern'
            self.leftFinishButton.setEnabled(True)
        elif self.leftDisplay=='MeshPattern':
            self.leftGraphWidget.setImage(self.displayData)
            self.leftDisplay='ReconstructData'
        

    def leftUndoFunc(self):
        self.selectedRegion=self.selectedRegion[:len(self.selectedRegion)-1]
        self.fftData=CompFunc.fft.dataToFft(self.avgRawData)
        for i in np.arange(0,len(self.selectedRegion),1):
            CompFunc.fft.maskFftData(self.fftData,self.selectedRegion[i])
        self.absFftData=CompFunc.fft.fftToAbs(self.fftData)
        self.rightGraphWidget.setImage(self.absFftData)
        self.displayData=CompFunc.fft.fftToData(self.fftData,self.rawIntensity)
        self.leftGraphWidget.setImage(self.displayData)
        self.leftDisplay='ReconstructData'

    def leftFinish(self):
        self.sumRatio=np.zeros(np.shape(self.rawData[0]))
        for i in np.arange(0,np.size(self.meshPattern,0),1):
            self.sumRatio = self.sumRatio + self.meshPattern[i]
        self.sumRatio=self.sumRatio/np.size(self.meshPattern,0)
        self.displayData=self.rawData.copy()
        for i in np.arange(0,np.size(self.rawData,0),1):
            self.displayData[i] = self.displayData[i]*self.sumRatio
        self.leftGraphWidget.setImage(self.displayData)
        self.leftDisplay='ReconstructData'
        self.fftData=CompFunc.fft.dataToFft(self.displayData)
        self.absFftData=CompFunc.fft.fftToAbs(self.fftData)
        self.rightGraphWidget.setImage(self.absFftData)
        self.leftSelectPeakButton.setEnabled(False)
        self.leftUndoButton.setEnabled(False)
        self.leftFinishButton.setEnabled(False)
        return 0

    def startFixCurvature(self):
        self.status='FixCurvature'
        self.rightSelectPeakButton.setEnabled(True)
        self.rightUndoButton.setEnabled(True)
        self.selectedRegion=[]
        self.rightGraphWidget.view.getViewBox().setMouseMode(pg.ViewBox.RectMode)

    def rightSelectedRegion(self):
        region=self.rightGraphWidget.view.getViewBox().viewRange()
        self.selectedRegion.append(region)
        print(self.selectedRegion[len(self.selectedRegion)-1])
        CompFunc.fft.drawRect(self.absFftData,region)
        self.rightGraphWidget.setImage(self.absFftData)
        self.rightFinishButton.setEnabled(True)

    def rightUndoFunc(self):
        self.selectedRegion=self.selectedRegion[:len(self.selectedRegion)-1]
        self.absFftData=CompFunc.fft.fftToAbs(self.fftData)
        for i in np.arange(0,len(self.selectedRegion),1):
            CompFunc.fft.drawRect(self.absFftData,self.selectedRegion[i])
        self.rightGraphWidget.setImage(self.absFftData)
    

    def rightRadiusUp(self):
        self.radius = self.radius + 100
        print('radius=',self.radius)
        self.displayData=self.rawData.copy()
        self.displayData=CompFunc.fft.fixCurvature2D(self.displayData[0],self.radius)
        self.fftData=sp.fft.fft2(self.displayData)
        self.absFftData=np.log(np.abs(self.fftData)+1.)
        self.leftGraphWidget.setImage(self.displayData)
        self.rightGraphWidget.setImage(self.absFftData)
        self.rightFinishButton.setEnabled(True)

    def rightRadiusDown(self):
        self.radius = self.radius - 100
        print('radius=',self.radius)
        self.displayData=self.rawData.copy()
        self.displayData=CompFunc.fft.fixCurvature2D(self.displayData[0],self.radius)
        self.fftData=sp.fft.fft2(self.displayData)
        self.absFftData=np.log(np.abs(self.fftData)+1.)
        self.leftGraphWidget.setImage(self.displayData)
        self.rightGraphWidget.setImage(self.absFftData)
        self.rightFinishButton.setEnabled(True)

    def rightFinish(self):
        self.fixedData=np.zeros(np.shape(self.rawData))
        for i in np.arange(0,np.size(self.rawData,0),1):
            self.fixedData[i]=CompFunc.fft.fixCurvature2D(self.rawData[i],self.radius)
        self.rawData=self.fixedData
        self.displayData=self.rawData.copy()
        self.leftGraphWidget.setImage(self.displayData)
        self.fftData=CompFunc.fft.dataToFft(self.displayData)
        self.absFftData=CompFunc.fft.fftToAbs(self.fftData)
        print(np.shape(self.absFftData))
        self.rightGraphWidget.setImage(self.absFftData)
        self.rightSelectPeakButton.setEnabled(False)
        self.rightUndoButton.setEnabled(False)
        self.rightFinishButton.setEnabled(False)

    def seeAvgFunc(self):
        self.avgRawData=np.zeros((1,np.shape(self.rawData[0])[0],np.shape(self.rawData[0])[1]))
        for i in np.arange(0,np.size(self.rawData,0),1):
            self.avgRawData[0] = self.avgRawData[0] + self.rawData[i]
        self.avgRawData = self.avgRawData/np.size(self.rawData,0)
        self.leftGraphWidget.setImage(self.avgRawData)
        self.fftData=CompFunc.fft.dataToFft(self.avgRawData)
        self.absFftData=CompFunc.fft.fftToAbs(self.fftData)
        self.rightGraphWidget.setImage(self.absFftData)

    def testFunc(self):
        CPU_NUM=cpu_count()
        
        timeStart=time()
        blockSize=48
        
        M=len(self.meshPattern[0])*4//blockSize
        N=len(self.meshPattern[0][0])*4//blockSize
        print(str(M)+'by'+str(N))

        self.displayData=np.zeros((self.displayData.shape))
        self.smallImageArray=CompFunc.fft.splitLargeImage(self.meshPattern[0],M,N,blockSize)
        self.avgScanImageArray=CompFunc.fft.splitLargeImage(self.avgRawData[0],M,N,blockSize)
        self.optimizedMeshArray=np.zeros(self.smallImageArray.shape)
        self.optimizedMeshLoss=np.zeros(M*N)

        with Pool(processes=CPU_NUM) as pool:
            index=0
            while (index+CPU_NUM)<M*N:
                m_Results=[pool.apply_async(CompFunc.fft.optimizeMeshFromData,args=(self.smallImageArray[i],self.avgScanImageArray[i],)) for i in range(index,index+CPU_NUM)]
                for i in range(CPU_NUM):
                    self.optimizedMeshArray[index+i],self.optimizedMeshLoss[index+i]=m_Results[i].get()
                    print(str(index+i+1)+'/'+str(M*N))
                index+=CPU_NUM
            m_Results=[pool.apply_async(CompFunc.fft.optimizeMeshFromData,args=(self.smallImageArray[i],self.avgScanImageArray[i],)) for i in range(index,M*N)]
            for i in range(len(m_Results)):
                self.optimizedMeshArray[index+i],self.optimizedMeshLoss[index+i]=m_Results[i].get()
                print(str(index+i+1)+'/'+str(M*N))
            
        
        pool.close()
        pool.join()


        self.meshPattern[0]=CompFunc.fft.constructLargeImage(self.optimizedMeshArray,1./self.optimizedMeshLoss,M,N,blockSize,len(self.rawData[0]),len(self.rawData[0][0]))
        self.displayData=self.meshPattern
        
        self.leftGraphWidget.setImage(self.displayData)
        print(time()-timeStart)


        






        
def main():

    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

