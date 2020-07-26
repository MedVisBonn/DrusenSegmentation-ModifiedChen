# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:59:48 2019

@author: gorgi
"""


import os
import argparse
import numpy as np
from os import listdir
from skimage import io
from scipy import ndimage
from os.path import isfile, join
from matplotlib import pyplot as plt
            
def compute_OR_3D(gt, pr,measureFormat):
    
    denom = float(np.sum((np.logical_or(gt>0, pr>0)>0).astype('float')))
    nom   = float(np.sum((np.logical_and(gt>0, pr>0)>0).astype('float')))
    OR      = nom/denom if denom > 0 else 1.0
    return OR

def compute_ADAD_3D(gt,pr,measureFormat,resy=1.):
    
    adad = np.abs(np.sum(gt)-np.sum(pr))
    numAscans = float(np.sum(np.sum(gt, axis=0)>0)) if np.sum(gt)>0.0 else 1.0
    adad = float(adad)/numAscans 
    gtLoad=float(np.sum((gt>0).astype('int')))/numAscans if numAscans>0 else 1.0
    resolution= resy #um Height of druse -- 3.872 
    if(measureFormat=="percent"):
        adad=adad/gtLoad if gtLoad>0 else adad
        adad=adad*100
    elif(measureFormat=="um"):
        adad=adad*resolution
    else:
        pass
    return adad

def compute_ADAD(gt, pr,measureFormat, resy=1.):
    
    adad = np.abs(np.sum(gt)-np.sum(pr))
    numAscans = float(np.sum(np.sum(gt, axis=0)>0)) if np.sum(gt)>0.0 else 1.0
    adad = float(adad)/numAscans 
    gtLoad=float(np.sum((gt>0).astype('int')))/numAscans if numAscans>0 else 1.0
    resolution=resy #um Height of druse -- 3.872
    if(measureFormat=="percent"):
        adad=float(adad)/gtLoad if gtLoad>0 else adad
        adad=adad*100
    elif(measureFormat=="um"):
        adad=adad*resolution
    else:
        pass
    return adad 
        
def compute_OR(gt,pr,measureFormat):
    
    denom = float(np.sum((np.logical_or(gt>0, pr>0)>0).astype('float')))
    nom   = float(np.sum((np.logical_and(gt>0, pr>0)>0).astype('float')))
    gtLoad=float(np.sum((gt>0).astype('int')))
    OR      = nom/denom if denom > 0 else 1.0
    return OR
 
def read_b_scans( path , img_type = "None",toID=False,returnIds=False):
    
    d2 = [f for f in listdir(path) if isfile(join(path, f))]
    rawstack = list()
    ind = list()
    rawStackDict=dict()
    rawSize=()
    for fi in range(len(d2)):
         filename = path+'/'+d2[fi]
         ftype = d2[fi].split('-')[-1]
         if(ftype==img_type or img_type=="None"):
             ind.append(int(d2[fi].split('-')[0]))
             
             raw = io.imread(filename)
             rawSize = raw.shape
             rawStackDict[ind[-1]]=raw
    rawstack=np.empty((rawSize[0],rawSize[1],len(ind)))
   
    keys=rawStackDict.keys()
    keys = sorted(keys)
    i=0
    for k in keys:
        rawstack[:,:,i]=rawStackDict[k]
        i+=1
    if(returnIds):
        return rawstack,keys
    return rawstack

def filter_height_1(scan):
    height=np.sum(scan,axis=0)
    mask=(height>0).astype(int)
    mask[height==1]=0
    mask3D=np.stack([mask]*scan.shape[0],axis=0)
    return scan*mask3D

def filter_height_1_bscan(bscan):
    height=np.sum(bscan,axis=0)
    mask=(height>0).astype(int)
    mask[height==1]=0
    mask2D=np.stack([mask]*bscan.shape[0],axis=0)
    return bscan*mask2D

def compute_component_max_height(cca,heights,bgLbl):
    labels  = np.unique( cca )
    max_hs  = np.zeros( cca.shape[1] )
    for l in labels:
        if(l==bgLbl):
            continue
        region = np.sum((cca == l).astype(int),axis=0)>0
        max_hs[region] = np.max( region.astype(int) * heights )
    return max_hs

def get_label_of_largest_component(labels,mask ):
        
        bgInd=labels[np.where(mask==0)]
        if(len(bgInd)>0):
            return np.unique(bgInd)[0]
        size = np.bincount(labels.ravel())
        largest_comp_ind = size.argmax() 
        return largest_comp_ind
        
def filter_max_height_2_bscan(bscan):
    
    cca, num_drusen = ndimage.measurements.label( bscan )
    bgLbl=get_label_of_largest_component(cca,bscan)  
    max_hs=compute_component_max_height(cca,np.sum(bscan,axis=0),bgLbl)
    max_hs=np.tile(max_hs,bscan.shape[0]).reshape(bscan.shape)
    max_hs[max_hs<3]=0
    max_hs[max_hs>0]=1
    return bscan*max_hs
    
def get_measure_for_bScan_with_max_drusen(gt,pr,calcMeasure,measureFormat,resy):
    #Index of B-scan with max druse load
    loadPerBScan=np.sum(np.sum(gt,axis=0),axis=0)
    maxInd=np.argmax(loadPerBScan)
    localGt=gt[:,:,maxInd]
    if(np.sum(localGt)==0):
        return []

    localPr=pr[:,:,maxInd]
    if(calcMeasure=='OR'):
        return [compute_OR(localGt,localPr,measureFormat)]
    if(calcMeasure=='ADAD'):
        return [compute_ADAD(localGt,localPr,measureFormat,resy)]

def get_measure_for_druse_present_bScans(gt,pr,calcMeasure,measureFormat,resy):
    loadPerBScan=np.sum(np.sum(gt,axis=0),axis=0)
    m=[]
    for ind in np.where(loadPerBScan>0)[0]:
        localGt=gt[:,:,ind]
        if(np.sum(localGt)==0):
            continue
        localPr=pr[:,:,ind]
        if(calcMeasure=='OR'):
            m.append(compute_OR(localGt,localPr,measureFormat))
        if(calcMeasure=='ADAD'):
            m.append(compute_ADAD(localGt,localPr,measureFormat,resy))
    return m

def get_measure_for_all_bscans(gt,pr,calcMeasure,measureFormat,resy):
    m=[]
    for ind in np.arange(gt.shape[2]):
        localGt=gt[:,:,ind]
        localPr=pr[:,:,ind]
        if(calcMeasure=='OR'):
            m.append(compute_OR(localGt,localPr,measureFormat))
        if(calcMeasure=='ADAD'):
            m.append(compute_ADAD(localGt,localPr,measureFormat,resy))
    return m
    
def get_measure_for_3D_volume(gt,pr,calcMeasure,measureFormat,resy):
    #Index of B-scan with max druse load
    loadPerBScan=np.sum(np.sum(gt,axis=0),axis=0)
    maxInd=np.argmax(loadPerBScan)
    if(calcMeasure=='OR'):
        return [compute_OR_3D(gt,pr,measureFormat)]
    if(calcMeasure=='ADAD'):
        return [compute_ADAD_3D(gt,pr,measureFormat,resy)]

def create_directory(path):
        """
        Check if the directory exists. If not, create it.
        Input:
            path: the directory to create
        Output:
            None.
        """
        if not os.path.exists(path):
            os.makedirs(path)

def compute_drusen_load_distribution(path,savePath):
    loadsHighRes=[]
    loadsLowRes=[]
    for d1 in os.listdir(path):
        for d2 in os.listdir(path+'/'+d1):
            for d3 in os.listdir(path+'/'+d1+'/'+d2):
                localPath=path+'/'+d1+'/'+d2+'/'+d3+'/'
                md = [f for f in listdir(localPath) if isfile(join(localPath, f))]
                if(len(md)==0):
                    continue
                gt=read_b_scans(localPath,"drusen.png")
                
                gt = gt.astype('float')
                gt[gt>0]  = 1.0
                gt[gt<=0] = 0.0
                gt=filter_height_1(gt)  
                if(gt.shape[2]>70):
                    loadsHighRes.append(np.sum(gt))
                else:
                    loadsLowRes.append(np.sum(gt))
    np.savetxt(savePath+"/highResDrusenLoads.txt",np.asarray(loadsHighRes))
    np.savetxt(savePath+"/lowResDrusenLoads.txt",np.asarray(loadsLowRes))

def compute_drusen_load_distribution_refined_data(path,savePath,name=''):
    loadsHighRes=[]
    loadsEnface=[]
    
    f1=open(savePath+"/"+name+"highResDrusenLoads.txt","w")
    f2=open(savePath+"/"+name+"enfaceDrusenLoads.txt","w")
#    counter=0
    for d1 in os.listdir(path):
        
            localPath=path+'/'+d1+'/'
            md = [f for f in listdir(localPath) if isfile(join(localPath, f))]
            if(len(md)==0):
                continue
            gt=read_b_scans(localPath+'DrusenAfterFPE/',"binmask.png")
            
            gt = gt.astype('float')
            gt[gt>0]  = 1.0
            gt[gt<=0] = 0.0
            gt=filter_height_1(gt)  
           
            f1.write(d1+":"+str(np.sum(gt))+"\n")
           
            # Project on 2D
            gtProj=np.max(gt,axis=0)
            
            f2.write(d1+":"+str(np.sum(gtProj))+"\n")
        

def draw_histogram(dataPath):
    data=np.loadtxt(dataPath)
    sparse=data<=20000
    med=(data<=70000)*(data>20000)
    dense=data>70000
    return
    plt.hist(data, bins=1000)  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()
    
                
def evaluate_performance(gtPath, prPath,calcMeasure='OR',scope='druPresent',\
                         resx=1., resy=1., resz=1.,savePath="",measureFormat="um"):
    highResMeasure=[]
            
    md = [f for f in listdir(gtPath) if isfile(join(gtPath, f))]
    gtshape=len(md)
    saveFname=savePath+os.sep+scope+'-'+calcMeasure+'-'+measureFormat+'.txt'
    
    gt=read_b_scans(gtPath)
    gt = gt.astype('float')
    gt[gt>0]  = 1.0
    gt[gt<=0] = 0.0
    
    pr=read_b_scans(prPath) #binmask.png for chen and refined-Chen, drusen otherwise
    pr = pr.astype('float')
    pr[pr>0]  = 1.0
    pr[pr<=0] = 0.0
    
    
    if(scope=='maxDru'):
        m=get_measure_for_bScan_with_max_drusen(gt,pr,calcMeasure,measureFormat,resy)
        
    elif(scope=='druPresent'):
        m=get_measure_for_druse_present_bScans(gt,pr,calcMeasure,measureFormat,resy)
        
    elif(scope=='vol'):
        m=get_measure_for_3D_volume(gt,pr,calcMeasure,measureFormat,resy) 
        
    elif(scope=='all'):
        m=get_measure_for_all_bscans(gt,pr,calcMeasure,measureFormat,resy)    
    else:
        print("Warning: in evaluate_performance(...), unknown scope value encountered.")

    if(len(m)>0):
        create_directory(savePath)
        np.savetxt('.'+os.sep+saveFname, np.asarray(m))
        
    return m

def draw_evaluation_diagram(A,AErr,xA,B,BErr,xB,title,xTicks,ylabel):
    fig,ax=plt.subplots()
    ax.set_title(title)
    p1=plt.errorbar(xA,A,yerr=AErr,fmt='o',color='r',capsize=4)
    p2=plt.errorbar(xB,B,yerr=BErr,fmt='o',color='b',capsize=4)
    plt.xlabel("Drusen load(Num of Scans)")
    plt.ylabel(ylabel)
    plt.xticks([1,2,3],xTicks)
    plt.legend((p1[0],p2[0]),('Chen method','Modified Chen method'))
    
    plt.show()
    
def extract_info_from_file(path,highRes):   
    data=[]
    with open(path,'r') as f:
        data=f.readlines()
    
    hSize=int(data[0].split(":")[-1])    
    hMean=float(data[1].split(":")[-1])
    hStdv=float(data[2].split(":")[-1])
        
    lSize=float(data[5].split(":")[-1])
    lMean=float(data[6].split(":")[-1])
    lStdv=float(data[7].split(":")[-1])
    
    if(highRes):
        return (hMean,hStdv,hSize)
    return (lMean,lStdv,lSize)

def read_gt_load_from_file(filePath):
    gtLoadFile=open(filePath,'r')
    lines=gtLoadFile.readlines()
    gtLoadInfo=dict()
    for l in lines:
        l=l.rstrip()
        gtLoadInfo[l.split(':')[0]]=int(float(l.split(':')[1]))
    return gtLoadInfo

def compute_drusen_volume(gtPath, prPath, resx=1., resy=1., resz=1., savePath=""):
            
    md = [f for f in listdir(gtPath) if isfile(join(gtPath, f))]
    gtshape=len(md)
    
    gt=read_b_scans(gtPath)
    gt = gt.astype('float')
    gt[gt>0]  = 1.0
    gt[gt<=0] = 0.0
    
    pr=read_b_scans(prPath) #binmask.png for chen and refined-Chen, drusen otherwise
    pr = pr.astype('float')
    pr[pr>0]  = 1.0
    pr[pr<=0] = 0.0
    
    voxelSize=resx*resy*resz
    
    gtVolVx=np.sum(gt)
    prVolVx=np.sum(pr)
    
    create_directory(savePath)
    
    saveFname=savePath+os.sep+'Drusen-volume-GT.txt'
    cnt=['Voxel:'+str(gtVolVx)+'\n','Micometer3:',str(gtVolVx*voxelSize)+'\n']
    with open(saveFname,'w') as f:
        f.writelines(cnt)
    
    saveFname=savePath+os.sep+'Drusen-volume-PR.txt'
    cnt=['Voxel:'+str(prVolVx)+'\n','Micometer3:',str(prVolVx*voxelSize)+'\n']
    with open(saveFname,'w') as f:
        f.writelines(cnt)   
        
def main(args):
    
    measures=[('OR','percent'),('ADAD','percent'),('ADAD','um')]
    for m in measures:
        print("Computing:",m)
        ev=evaluate_performance(args.gtPath, args.prPath, m[0],\
                                args.scope, float(args.resx), float(args.resy),\
                                float(args.resz), args.savePath, m[1])  
    print("Computing drusen volume")
    compute_drusen_volume(args.gtPath, args.prPath, float(args.resx),\
                        float(args.resy), float(args.resz), args.savePath)
    print('Done!')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate drusen segmentation accuracy using the manual segmentation (gt)"+\
            " and automatically generated segmentation (pr).")
    parser.add_argument(
        "--gtPath", action="store", required=True, help="Path to the folder containing ground truth drusen segmentation.")
    parser.add_argument(
        "--prPath", action="store", required=True, help="Path to the folder that contains automated drusen segmentation for evaluation.")
    parser.add_argument(
        "--savePath", action="store", required=True, help="Path to save evaluation.")
    parser.add_argument(
        "--scope", action="store", required=True, help="Evaluation scope: {"+\
            "\nvol: volumetric computation,"+\
            "\nmaxDru: B-scan with largest drusen load in OCT volume,"+\
            "\ndruPresent: B-scans with drusen}.")
    parser.add_argument(
        "--resx", action="store", required=False, default=1.0, help="Pixel size in x direction (x resolution in each B-scan) in micrometer. ")    
    parser.add_argument(
        "--resy", action="store", required=False, default=1.0, help="Pixel size in y direction (y resolution in each B-scan) in micrometer. ")    
    parser.add_argument(
        "--resz", action="store", required=False, default=1.0, help="Distance between consecutive B-scans in micrometer. ")    
    args = parser.parse_args()
    main(args)
