
from os import XATTR_REPLACE
import numpy as np
from math import log, sqrt,sin,cos,pi,exp,atan,isinf,isnan
import scipy.fft
#import scipy.fft.fft2
from numba import jit





def dataToFft(inputdata):
    fftData=np.zeros(np.shape(inputdata),dtype=np.complex)
    M=len(inputdata[0])//2
    N=len(inputdata[0][0])//2
    for i in np.arange(0,np.size(inputdata,0),1):
        temp=scipy.fft.fft2(inputdata[i])
        fftData[i]=np.roll(np.roll(temp,M,axis=0),N,axis=1)
    print(np.shape(fftData))
    return fftData



def fftToAbs(fftData):
    return np.log(np.abs(fftData)+1.)

def fftToData(fftData,rawIntensity):
    reconstructData=np.zeros(np.shape(fftData),dtype=np.complex)
    for i in np.arange(0,np.size(fftData,0),1):
        reconstructData[i]=np.flipud(np.fliplr(scipy.fft.fft2(fftData[i])))
    reconstructData=np.abs(reconstructData)
    
    for i in np.arange(0,np.size(fftData,0),1):
        reconstructData[i]=reconstructData[i]*rawIntensity[i]/np.mean(reconstructData[i])

    return reconstructData

@jit(nopython=True)
def complex2angles(complexArray):
    newArray=np.zeros(2*len(complexArray),dtype=np.float64)
    for i in range(len(complexArray)):
        newArray[i]=abs(complexArray[i])

    for i in range(len(complexArray),2*len(complexArray)):
        newArray[i]=float(np.angle(complexArray[i-len(complexArray)]))
    return newArray

@jit(nopython=True)
def angles2complex(normalArray):
    newArray=np.zeros(len(normalArray)//2,dtype=np.complex128)
    for i in range(len(newArray)):
        newArray[i]=normalArray[i]*(complex(cos(normalArray[i+len(newArray)]))+(1j)*sin(normalArray[i+len(newArray)]))
    return newArray

def genHigherPeaks(firstOrder):
    peaks=np.zeros((25,2),dtype=np.int)
    peaks[0]=firstOrder[0]
    peaks[1]=firstOrder[1]
    peaks[2]=firstOrder[2]
    peaks[3]=2*firstOrder[0]-firstOrder[1]
    peaks[4]=2*firstOrder[0]-firstOrder[2]
    peaks[5]=-firstOrder[0]+2*firstOrder[2]
    peaks[6]=-firstOrder[0]+firstOrder[2]+firstOrder[1]
    peaks[7]=-firstOrder[0]+2*firstOrder[1]
    peaks[8]=firstOrder[1]+firstOrder[0]-firstOrder[2]
    peaks[9]=2*peaks[0]-peaks[5]
    peaks[10]=2*peaks[0]-peaks[6]
    peaks[11]=2*peaks[0]-peaks[7]
    peaks[12]=2*peaks[0]-peaks[8]
    peaks[13]=3*peaks[2]-2*peaks[0]
    peaks[14]=peaks[5]+peaks[1]-peaks[0]
    peaks[15]=peaks[2]+peaks[7]-peaks[0]
    peaks[16]=3*peaks[1]-2*peaks[0]
    peaks[17]=peaks[7]+peaks[4]-peaks[0]
    peaks[18]=peaks[9]+peaks[1]-peaks[0]
    peaks[19]=2*peaks[0]-peaks[13]
    peaks[20]=2*peaks[0]-peaks[14]
    peaks[21]=2*peaks[0]-peaks[15]
    peaks[22]=2*peaks[0]-peaks[16]
    peaks[23]=2*peaks[0]-peaks[17]
    peaks[24]=2*peaks[0]-peaks[18]

    return peaks

def simplexMethod(loss_function,alpha,startValues,para_start,para_stop,*loss_function_para):
    N=len(startValues)
    randomArray=np.ones(N)
    pointsMatrix=np.zeros((N+1,N))
    lossValues=np.zeros(N+1)
    
    for i in range(N+1):
        randomArray=np.ones(N)
        for j in range(para_start,para_stop):
            randomArray[j]=np.random.uniform(1./alpha,alpha)
        pointsMatrix[i]=startValues*randomArray
        lossValues[i]=loss_function(pointsMatrix[i],loss_function_para)
    
    for iter in range(200):
        max_loss=0
        for i in range(N+1):
            if (lossValues[i]>lossValues[max_loss]):
                max_loss=i
        pointCOM=np.zeros(N)
        for i in range(N+1):
            if(i!=max_loss):
                pointCOM+=pointsMatrix[i]
        pointCOM/=N
        newPoint=2.*pointCOM-pointsMatrix[max_loss]
        if (loss_function(newPoint,loss_function_para)>lossValues[max_loss]):
            pointsMatrix[max_loss]=pointCOM+0.7*(pointCOM-pointsMatrix[max_loss])
            lossValues[max_loss]=loss_function(pointsMatrix[max_loss],loss_function_para)
        else:
            pointsMatrix[max_loss]=pointCOM+1.2*(pointCOM-pointsMatrix[max_loss])
            lossValues[max_loss]=loss_function(pointsMatrix[max_loss],loss_function_para)
        #print(lossValues[max_loss])
    #print(lossValues[max_loss],pointCOM)
    for i in range(len(startValues)):
        startValues[i]=pointCOM[i]
    return lossValues[max_loss]

@jit(nopython=True)
def createMeshFromPeaks(inputArray,XYPos_1st,xLen,yLen):
    
    XYPoses=np.zeros((25,2))
    XYPoses[1,0]=XYPos_1st[0]
    XYPoses[1,1]=XYPos_1st[1]
    XYPoses[2,0]=XYPos_1st[2]
    XYPoses[2,1]=XYPos_1st[3]
    XYPoses[3]=-XYPoses[1]
    XYPoses[4]=-XYPoses[2]

    XYPoses[5]=2.*XYPoses[2]
    XYPoses[6]=XYPoses[2]+XYPoses[1]
    XYPoses[7]=XYPoses[1]+XYPoses[1]
    XYPoses[8]=XYPoses[4]+XYPoses[1]
    XYPoses[9]=-XYPoses[5]
    XYPoses[10]=-XYPoses[6]
    XYPoses[11]=-XYPoses[7]
    XYPoses[12]=-XYPoses[8]
    XYPoses[13]=3.*XYPoses[2]
    XYPoses[14]=XYPoses[5]+XYPoses[1]
    XYPoses[15]=XYPoses[2]+XYPoses[7]
    XYPoses[16]=3.*XYPoses[1]
    XYPoses[17]=XYPoses[4]+XYPoses[7]
    XYPoses[18]=XYPoses[1]+XYPoses[9]
    XYPoses[19]=-XYPoses[13]
    XYPoses[20]=-XYPoses[14]
    XYPoses[21]=-XYPoses[15]
    XYPoses[22]=-XYPoses[16]
    XYPoses[23]=-XYPoses[17]
    XYPoses[24]=-XYPoses[18]



    fourierPoints_compressed=angles2complex(inputArray)
    fourierPoints=np.zeros(25,dtype=np.complex128)
    fourierPoints[0]=fourierPoints_compressed[0]
    fourierPoints[1]=fourierPoints_compressed[1]
    fourierPoints[2]=fourierPoints_compressed[2]
    fourierPoints[3]=fourierPoints_compressed[1].conjugate()
    fourierPoints[4]=fourierPoints_compressed[2].conjugate()
    fourierPoints[5]=fourierPoints_compressed[3]
    fourierPoints[6]=fourierPoints_compressed[4]
    fourierPoints[7]=fourierPoints_compressed[5]
    fourierPoints[8]=fourierPoints_compressed[6]
    fourierPoints[9]=fourierPoints_compressed[3].conjugate()
    fourierPoints[10]=fourierPoints_compressed[4].conjugate()
    fourierPoints[11]=fourierPoints_compressed[5].conjugate()
    fourierPoints[12]=fourierPoints_compressed[6].conjugate()
    fourierPoints[13]=fourierPoints_compressed[7]
    fourierPoints[14]=fourierPoints_compressed[8]
    fourierPoints[15]=fourierPoints_compressed[9]
    fourierPoints[16]=fourierPoints_compressed[10]
    fourierPoints[17]=fourierPoints_compressed[11]
    fourierPoints[18]=fourierPoints_compressed[12]
    fourierPoints[19]=fourierPoints_compressed[7].conjugate()
    fourierPoints[20]=fourierPoints_compressed[8].conjugate()
    fourierPoints[21]=fourierPoints_compressed[9].conjugate()
    fourierPoints[22]=fourierPoints_compressed[10].conjugate()
    fourierPoints[23]=fourierPoints_compressed[11].conjugate()
    fourierPoints[24]=fourierPoints_compressed[12].conjugate()

    num=len(fourierPoints)
    #num=13
    recreatedMesh=np.zeros((xLen,yLen))
    for index in (0,1,2,5,6,7,8):
        for m in range(xLen):
            for n in range(yLen):
                recreatedMesh[m,n]+=2.*(fourierPoints[index]/(xLen*yLen)*(cos(2*pi*((XYPoses[index,0]/xLen)*m+(XYPoses[index,1]/yLen)*n))+(1.j)*sin(2*pi*((XYPoses[index,0]/xLen)*m+(XYPoses[index,1]/yLen)*n)))).real
    return recreatedMesh


@jit(nopython=True)
def normalizedAndProduct(recreatedMesh,standardMesh):
    sum=0.
    for m in range(len(standardMesh)):
        for n in range(len(standardMesh[0])):
            sum+=(recreatedMesh[m,n])**2
    if(sum==sum):
        sum =sum**0.5
        recreatedMesh/=sum
        sum=0.
        for m in range(len(standardMesh)):
            for n in range(len(standardMesh[0])):
                sum+=(recreatedMesh[m,n])*standardMesh[m,n]
    else:
        sum=0.
    return sum


@jit(nopython=True)
def meshLoss(fourierPoints_compressed_with_XyPos,allPara):
    loss=0.

    fourierPoints_compressed=fourierPoints_compressed_with_XyPos[:len(fourierPoints_compressed_with_XyPos)-4].copy()
    XYPoses=fourierPoints_compressed_with_XyPos[len(fourierPoints_compressed_with_XyPos)-4:].copy()


    standardMesh=allPara[0]
    
    xLen=len(standardMesh)
    yLen=len(standardMesh[0])
    recreatedMesh=createMeshFromPeaks(fourierPoints_compressed,XYPoses,xLen,yLen)
    
    sum=normalizedAndProduct(recreatedMesh,standardMesh)
    loss=1.-sum
    #print(loss)
    return loss

#@jit(nopython=True)
def meshLoss3(fourierPoints_compressed_with_XyPos,allPara):
    loss=0.

    fourierPoints_compressed=fourierPoints_compressed_with_XyPos[:len(fourierPoints_compressed_with_XyPos)-4].copy()
    XYPoses=fourierPoints_compressed_with_XyPos[len(fourierPoints_compressed_with_XyPos)-4:].copy()


    originalImage=allPara[0]
    
    xLen=len(originalImage)
    yLen=len(originalImage[0])
    recreatedMesh=createMeshFromPeaks(fourierPoints_compressed,XYPoses,xLen,yLen)

    reconstructData=recreatedMesh*originalImage

    inputdata=scipy.fft.fft2(reconstructData)
    M=len(inputdata)//2
    N=len(inputdata[0])//2

    inputdata=np.roll(np.roll(inputdata,M,axis=0),N,axis=1)
    sum=0.



    #for i in range(int(XYPoses[0])-2,int(XYPoses[0])+3):
    #    sum+=abs(inputdata[M+i,N+int(XYPoses[1])])
    #for j in range(int(XYPoses[1])-2,int(XYPoses[1])+3):
    #    sum+=abs(inputdata[M+int(XYPoses[0]),N+j])
    #sum-=abs(inputdata[M+int(XYPoses[0]),N+int(XYPoses[1])])

    for i in range(int(XYPoses[0])-2,int(XYPoses[0])+3):
        for j in range(int(XYPoses[1])-2,int(XYPoses[1])+3):
            sum+=abs(inputdata[M+i,N+j])
    
    #for i in range(int(XYPoses[2])-2,int(XYPoses[2])+3):
    #    sum+=abs(inputdata[M+i,N+int(XYPoses[3])])
    #for j in range(int(XYPoses[3])-2,int(XYPoses[3])+3):
    #    sum+=abs(inputdata[M+int(XYPoses[2]),N+j])
    #sum-=abs(inputdata[M+int(XYPoses[2]),N+int(XYPoses[3])])

    for i in range(int(XYPoses[2])-2,int(XYPoses[2])+3):
        for j in range(int(XYPoses[3])-2,int(XYPoses[3])+3):
            sum+=abs(inputdata[M+i,N+j])
    

    
    
    loss=sum/abs(inputdata[M,N])
    #print(loss)
    return loss

def meshLoss4(fourierPoints_compressed_with_XyPos,allPara):
    loss=0.

    fourierPoints_compressed=fourierPoints_compressed_with_XyPos[:len(fourierPoints_compressed_with_XyPos)-4].copy()
    XYPos_1st=fourierPoints_compressed_with_XyPos[len(fourierPoints_compressed_with_XyPos)-4:].copy()


    XYPoses=np.zeros((25,2))
    XYPoses[1,0]=XYPos_1st[0]
    XYPoses[1,1]=XYPos_1st[1]
    XYPoses[2,0]=XYPos_1st[2]
    XYPoses[2,1]=XYPos_1st[3]
    XYPoses[3]=-XYPoses[1]
    XYPoses[4]=-XYPoses[2]

    XYPoses[5]=2.*XYPoses[2]
    XYPoses[6]=XYPoses[2]+XYPoses[1]
    XYPoses[7]=XYPoses[1]+XYPoses[1]
    XYPoses[8]=XYPoses[4]+XYPoses[1]

    originalImage=allPara[0]
    
    xLen=len(originalImage)
    yLen=len(originalImage[0])
    recreatedMesh=createMeshFromPeaks(fourierPoints_compressed,XYPos_1st,xLen,yLen)

    reconstructData=recreatedMesh*originalImage

    inputdata=scipy.fft.fft2(reconstructData)
    M=len(inputdata)//2
    N=len(inputdata[0])//2

    inputdata=np.roll(np.roll(inputdata,M,axis=0),N,axis=1)
    sum=0.

    for index in range(5,9):
        for i in range(int(XYPoses[index,0])-2,int(XYPoses[index,0])+3):
            for j in range(int(XYPoses[index,1])-2,int(XYPoses[index,1])+3):
                if(M+i<len(inputdata) and M+i>-1 and N+j<len(inputdata[0]) and N+j>-1 and i*j!=0):
                    sum+=abs(inputdata[M+i,N+j])

    
    loss=sum/abs(inputdata[M,N])
    #print(loss)
    return loss

@jit(nopython=True)
def devidedAndVar(recreatedMesh,standardMesh):
    newMesh=standardMesh/recreatedMesh
    mean=np.mean(newMesh)
    stdVar=np.std(newMesh)

    
    return stdVar/mean

@jit(nopython=True)
def meshLoss2(fourierPoints_compressed_with_XyPos,allPara):
    loss=0.
    
    fourierPoints_compressed=fourierPoints_compressed_with_XyPos[:len(fourierPoints_compressed_with_XyPos)-4].copy()
    XYPoses=fourierPoints_compressed_with_XyPos[len(fourierPoints_compressed_with_XyPos)-4:].copy()
    
    standardMesh=allPara[0]
    
    xLen=len(standardMesh[0])
    yLen=len(standardMesh[0][0])
    recreatedMesh=createMeshFromPeaks(fourierPoints_compressed,XYPoses,xLen,yLen)
    
    loss=devidedAndVar(recreatedMesh,standardMesh)
    #print(loss)
    return loss   
    



def recreateMeshFft(fftData,meshPattern,paraList):
    M=len(fftData)//2
    N=len(fftData[0])//2
    xLen=len(fftData)
    yLen=len(fftData[0])
    XyPos=np.zeros((len(paraList),2))
    sum_expanded=np.zeros(len(paraList),dtype=np.complex)
    for i in range(len(paraList)):
        sum=0.+0.j
        weight=0.
        xPos=0.
        yPos=0.
        for m in range(paraList[i,0]-1,paraList[i,0]+2):
            for n in range(paraList[i,1]-1,paraList[i,1]+2):
                if(m<len(fftData) and m>-1 and n<len(fftData[0]) and n>-1):
                    sum+=fftData[m,n]
                    weight+=abs(fftData[m,n])
                    xPos+=m*abs(fftData[m,n])
                    yPos+=n*abs(fftData[m,n])
        if(i==1 or i==2):
            if weight==0:
                XyPos[i,0]=xPos
                XyPos[i,1]=yPos
            else:
                xPos/=weight
                yPos/=weight
                xPos-=M
                yPos-=N
                XyPos[i,0]=xPos
                XyPos[i,1]=yPos
        sum_expanded[i]=sum
            #for m in range(len(fftData[0])):
            #    for n in range(len(fftData[0][0])):
            #        meshPattern[index,m,n]+=sum/(xLen*yLen)*(cos(2*pi*((xPos/xLen)*m+(yPos/yLen)*n))+(1.j)*sin(2*pi*((xPos/xLen)*m+(yPos/yLen)*n)))

    sum_compressed=np.zeros(13,dtype=np.complex128)
    sum_compressed[0]=sum_expanded[0]
    sum_compressed[1]=sum_expanded[1]
    sum_compressed[2]=sum_expanded[2]
    sum_compressed[3]=sum_expanded[5]
    sum_compressed[4]=sum_expanded[6]
    sum_compressed[5]=sum_expanded[7]
    sum_compressed[6]=sum_expanded[8]
    sum_compressed[7]=sum_expanded[13]
    sum_compressed[8]=sum_expanded[14]
    sum_compressed[9]=sum_expanded[15]
    sum_compressed[10]=sum_expanded[16]
    sum_compressed[11]=sum_expanded[17]
    sum_compressed[12]=sum_expanded[18]
    #print(sum_compressed)
    #print(complex2angles(sum_compressed))
    sum_compressed_angles=complex2angles(sum_compressed)

    returnValues=np.zeros(len(sum_compressed_angles)+4)
    for i in range(len(sum_compressed_angles)):
        returnValues[i]=sum_compressed_angles[i]
    
    returnValues[len(sum_compressed_angles)]=XyPos[1,0]
    returnValues[len(sum_compressed_angles)+1]=XyPos[1,1]
    returnValues[len(sum_compressed_angles)+2]=XyPos[2,0]
    returnValues[len(sum_compressed_angles)+3]=XyPos[2,1]


    return returnValues

@jit(nopython=True)
def findPeakPos(data,peaksArray):
    firstValue=0.
    secondValue=0.
    for i in range(len(data)//2+1,len(data)):
        for j in range(len(data[0])//2+1,len(data[0])):
            if(data[i,j]>firstValue):
                peaksArray[1,0]=i
                peaksArray[1,1]=j
                firstValue=data[i,j]
    
    for i in range(len(data)//2):
        for j in range(len(data[0])//2+1,len(data[0])):
            if(data[i,j]>secondValue):
                peaksArray[2,0]=i
                peaksArray[2,1]=j
                secondValue=data[i,j]


    

    if (firstValue==0):
        peaksArray[1,0]=len(data)//2
        peaksArray[1,1]=len(data)//2
        peaksArray[2,0]=len(data)//2
        peaksArray[2,1]=len(data)//2



def optimizeMeshFromData(meshData,rawImage):

    sum=0.
    standardMesh=meshData.copy()
    for i in range(len(meshData)):
        for j in range(len(meshData[0])):
            sum+=standardMesh[i,j]**2
    if(sum>0.):
        sum=sum**0.5
        standardMesh/=sum  

        peaks=np.zeros((3,2),dtype=np.int)
        #peaks[0,0]=50
        #peaks[0,1]=50
        #peaks[1,0]=57
        #peaks[1,1]=60
        #peaks[2,0]=41
        #peaks[2,1]=58
        
        #peaks[2,0]=59
        #peaks[2,1]=60
        #peaks[1,0]=38
        #peaks[1,1]=58

        M=len(meshData)//2
        N=len(meshData[0])//2

        peaks[0,0]=M
        peaks[0,1]=N
        
        fftData=np.roll(np.roll(scipy.fft.fft2(meshData),M,axis=0),N,axis=1)

        findPeakPos(abs(fftData),peaks)
        #print(peaks)

        recreatedMesh=np.zeros(meshData.shape,dtype=np.complex)
        optimized_para=recreateMeshFft(fftData,recreatedMesh,genHigherPeaks(peaks))

        standardMesh3D=np.zeros((1,len(standardMesh),len(standardMesh[0])))
        standardMesh3D[0]=rawImage
        #print('start value',meshLoss3(optimized_para,standardMesh3D))

        simplexMethod(meshLoss,1.2,optimized_para,14,16,standardMesh)
        simplexMethod(meshLoss,0.2,optimized_para,16,20,standardMesh)
        simplexMethod(meshLoss,1.01,optimized_para,26,30,standardMesh)
        simplexMethod(meshLoss,1.05,optimized_para,14,16,standardMesh)
        simplexMethod(meshLoss,2.0,optimized_para,1,3,standardMesh)
        
        
        #simplexMethod(meshLoss,2.0,optimized_para,16,20,standardMesh)
        #simplexMethod(meshLoss,2.0,optimized_para,3,7,standardMesh)
        

        simplexMethod(meshLoss,1.001,optimized_para,26,30,standardMesh)
        #simplexMethod(meshLoss,1.2,optimized_para,3,7,standardMesh)
        simplexMethod(meshLoss3,1.2,optimized_para,1,3,rawImage)
        simplexMethod(meshLoss3,1.2,optimized_para,14,16,rawImage)
        simplexMethod(meshLoss4,1.1,optimized_para,16,20,rawImage)
        lossValues=simplexMethod(meshLoss4,1.1,optimized_para,3,7,rawImage)
        
        
        
        
        
        
        
        #simplexMethod(meshLoss,0.2,optimized_para,20,26,standardMesh)
        #simplexMethod(meshLoss,0.2,optimized_para,7,13,standardMesh)
        #print(optimized_para)
        #print(atan(optimized_para[26]/optimized_para[27]),atan(optimized_para[28]/optimized_para[29]))

        recreatedMesh=createMeshFromPeaks(optimized_para[:len(optimized_para)-4],optimized_para[len(optimized_para)-4:],len(meshData),len(meshData[0]))
        recreatedMesh=recreatedMesh*np.mean(meshData)/np.mean(recreatedMesh)
        return recreatedMesh,lossValues
    else:
        return meshData,1000000

     
def splitLargeImage(inputImage,M,N,size):
    imageArray=np.zeros((M*N,size,size))
    if M*N!=1:
        for i in range(M):
            for j in range(N):
                startX=i*(len(inputImage)-size)//(M-1)
                startY=j*(len(inputImage[0])-size)//(N-1)
                #print(startX,startY)
                for iter1 in range(size):
                    for iter2 in range(size):
                        imageArray[N*i+j,iter1,iter2]=inputImage[startX+iter1,startY+iter2]
    else:
        imageArray[0]=inputImage
    
    return imageArray

def constructLargeImage(imageArray,imageWeight,M,N,size,xLen,yLen):
    largeImage=np.zeros((xLen,yLen))
    datacount=np.zeros((xLen,yLen))
    if M*N!=1:
        for i in range(M):
            for j in range(N):
                startX=i*(xLen-size)//(M-1)
                startY=j*(yLen-size)//(N-1)
                for iter1 in range(size):
                    for iter2 in range(size):
                        largeImage[startX+iter1,startY+iter2]+=(imageArray[N*i+j,iter1,iter2]*imageWeight[N*i+j])
                        datacount[startX+iter1,startY+iter2]+=imageWeight[N*i+j]

        for i in range(xLen):
            for j in range(yLen):
                if (datacount[i,j]!=0):
                    largeImage[i,j]/=datacount[i,j]
                    if isnan(largeImage[i,j]):
                        largeImage[i,j]=0.
                    if isinf(largeImage[i,j]):
                        largeImage[i,j]=0.
    else:
        largeImage=imageArray[0]
    return largeImage

def maskFftData(fftData,paraList=[]):
    xRange=[0,0]
    yRange=[0,0]
    
    xRange[0]=int(paraList[0][0])
    xRange[1]=int(paraList[0][1])
    yRange[0]=int(paraList[1][0])
    yRange[1]=int(paraList[1][1])

    for index in np.arange(0,np.size(fftData,0),1):
        if (xRange[1]-xRange[0])>(yRange[1]-yRange[0]):
            for i in np.arange(xRange[0],xRange[1],1):
                for j in np.arange(yRange[0],yRange[1],1):
                    fftData[index,i,j]=fftData[index,i,yRange[0]] + (fftData[index,i,yRange[1]]-fftData[index,i,yRange[0]])*(j-yRange[0])/(yRange[1]-yRange[0])
                    fftData[index,np.size(fftData,1)-1-i,np.size(fftData,2)-1-j]=fftData[index,i,j]
        else:
            for i in np.arange(yRange[0],yRange[1],1):
                for j in np.arange(xRange[0],xRange[1],1):
                    fftData[index,j,i]=fftData[index,xRange[0],i] + (fftData[index,xRange[1],i]-fftData[index,xRange[0],i])*(j-xRange[0])/(xRange[1]-xRange[0])
                    fftData[index,np.size(fftData,1)-1-j,np.size(fftData,2)-1-i]=fftData[index,j,i]

@jit(nopython=True)
def getRatio(rawData,reconstructData):
    ratioData=np.zeros((len(rawData),len(rawData[0]),len(rawData[0,0])))
    for i in range(len(rawData)):
        for j in range(len(rawData[0])):
            for k in range(len(rawData[0,0])):
                if rawData[i,j,k]!=0:
                    ratioData[i,j,k]=reconstructData[i,j,k]/rawData[i,j,k]
                    if ratioData[i,j,k] > 3.:
                        ratioData[i,j,k]=0.
    return ratioData

def createMeshArray(rawImage,M,N,blockSize):
    meshArray=np.zeros((M*N,blockSize,blockSize))
    rawImageArray=splitLargeImage(rawImage,M,N,blockSize)
    for index in range(M*N):
        smallRawData=rawImageArray[index]
        smallfftData=scipy.fft.fft2(smallRawData)
        smallfftData=np.roll(np.roll(smallfftData,blockSize//2,axis=0),blockSize//2,axis=1)
        peaks=np.zeros((3,2),dtype=np.int)
        findPeakPos(abs(smallfftData),peaks)
        peaks=genHigherPeaks(peaks)
        
        for z in (1,2,5,6,7,8):
            xRange=[peaks[z][0]-1,peaks[z][0]+2]
            yRange=[peaks[z][1]-1,peaks[z][1]+2]
            for i in np.arange(xRange[0],xRange[1],1):
                for j in np.arange(yRange[0],yRange[1],1):
                    if (i>=0 and j>=0 and i< blockSize and j<blockSize and i!=blockSize//2 and j!=blockSize//2):
                        smallfftData[i,j]=smallfftData[i,yRange[0]] + (smallfftData[i,yRange[1]]-smallfftData[i,yRange[0]])*(j-yRange[0])/(yRange[1]-yRange[0])
                        smallfftData[blockSize-1-i,blockSize-1-j]=smallfftData[i,j]

        
        
        meshRemovedData=abs(np.flipud(np.fliplr(scipy.fft.fft2(smallfftData))))
        meshArray[index]=meshRemovedData/smallRawData
    return meshArray

def drawRect(data,paraList=[]):
    xRange=[0,0]
    yRange=[0,0]
    xRange[0]=int(paraList[0][0])
    xRange[1]=int(paraList[0][1])
    yRange[0]=int(paraList[1][0])
    yRange[1]=int(paraList[1][1])

    for index in np.arange(0,np.size(data,0),1):
        maxI=np.max(data[index])
        for i in np.arange(xRange[0],xRange[1]+1,1):
            data[index,i,yRange[0]]=maxI
            data[index,i,yRange[0]-1]=maxI
            data[index,i,yRange[1]]=maxI
            data[index,i,yRange[1]+1]=maxI

            data[index,np.size(data,1)-1-i,np.size(data,2)-1-yRange[0]]=maxI
            data[index,np.size(data,1)-1-i,np.size(data,2)-1-yRange[0]-1]=maxI
            data[index,np.size(data,1)-1-i,np.size(data,2)-1-yRange[1]]=maxI
            data[index,np.size(data,1)-1-i,np.size(data,2)-1-yRange[1]+1]=maxI
        for j in np.arange(yRange[0],yRange[1]+1,1):
            data[index,np.size(data,1)-1-xRange[0],np.size(data,2)-1-j]=maxI
            data[index,np.size(data,1)-1-xRange[1],np.size(data,2)-1-j]=maxI
            data[index,np.size(data,1)-1-xRange[0]-1,np.size(data,2)-1-j]=maxI
            data[index,np.size(data,1)-1-xRange[1]+1,np.size(data,2)-1-j]=maxI

            data[index,xRange[0],j]=maxI
            data[index,xRange[1],j]=maxI
            data[index,xRange[0]-1,j]=maxI
            data[index,xRange[1]+1,j]=maxI

def fixCurvature2D(rawData,radius):
    mid=np.size(rawData,0)/2
    ref=sqrt(radius**2-mid**2)
    fixedData=np.zeros(np.shape(rawData))

    for i in np.arange(0,np.size(rawData,0),1):
        deltaY=sqrt(radius**2-abs(i-mid)**2)-ref
        fixedData[i]=np.roll(rawData[i],int(deltaY))

    return fixedData

def restoreCurvature2D(rawData,radius):
    mid=np.size(rawData,0)/2
    ref=sqrt(radius**2-mid**2)
    fixedData=np.zeros(np.shape(rawData))

    for i in np.arange(0,np.size(rawData,0),1):
        deltaY=sqrt(radius**2-abs(i-mid)**2)-ref
        fixedData[i]=np.roll(rawData[i],-int(deltaY))

    return fixedData