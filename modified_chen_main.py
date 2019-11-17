import os
import cv2
import MAFOD as mf
import argparse
import numpy as np
import scipy as sc
from os import listdir
from skimage import io
import filterfunctions as FF
from skimage import img_as_uint
from os.path import isfile, join
from matplotlib import pyplot as plt

octParams = dict()
rpe_scan_list  = list()
nrpe_scan_list = list()

def init_meta_info():
    useChenDefault = False
    metaFile = open('OCT_info.txt','r')
    lines=metaFile.readlines()
    for l in lines:
        l = l.rstrip()
        l = l.strip()
        params=l.split('=')
        if(len(params)==2):
            paramName = params[0].strip()
            val = params[1].strip()
            if(paramName=='useDefaultFromChen'):
                useChenDefault = True if val.lower()=='yes' else False
            else:
                octParams[paramName] = float(val)
                
    if(useChenDefault):
        # Defaults w.r.t. Chen et al. paper
        octParams['sizeInZ']=6  # Size in mm
        octParams['sizeInX']=6 # Size in mm
        octParams['sizeInY']=2 # Size in mm
        
        octParams['widthOverHeightCoeff']=6
        octParams['intensity']=4
        octParams['smoothingSigma']=0.1

def show_image( image, block = True ):
    plt.imshow( image, cmap = plt.get_cmap('gray'))
    plt.show(block)

def show_images( images, r, c, titles = [], d = 0 , save_path = "" , block = True):
    i = 1
    for img in images:
        ax = plt.subplot( r, c, i )
        ax.xaxis.set_visible( False )
        ax.yaxis.set_visible( False )
        if( len(titles) != 0 ):
            ax.set_title( titles[i-1] )
        if( len(img.shape) > 2 ):
            plt.imshow( img )
        else:
            plt.imshow( img , cmap = plt.get_cmap('gray'))
        i += 1
    if( save_path != "" ):
        plt.savefig(save_path+".png")
        plt.close()
    else:
        plt.show(block)
    
def compute_heights(cca,mask ):
        max_hs  = np.zeros( cca.shape )  
        bg_lbl  = get_label_of_background_component( cca ,mask)
        labels=np.unique(cca)
        for l in labels:
            if( l != bg_lbl ):
                y, x = np.where( cca == l )
                h = np.max(y) - np.min(y) + 1
                max_hs[cca == l] = h
                
        return max_hs
    
def find_rel_maxima( arr ):
    val = []
    pre = -1
    for a in arr:
        if( a != pre ):
            val.append(a)
        pre = a
    val = np.asarray(val)
    return val[sc.signal.argrelextrema(val, np.greater)]
    
def compute_component_sum_local_max_height( cca,mask ):
    labels  = np.unique( cca )
    max_hs  = np.zeros( cca.shape )  
    bg_lbl  = get_label_of_background_component( cca,mask )
    heights = compute_heights( cca,mask )
    for l in labels:
        if( l != bg_lbl ):
            region = cca == l
            masked_heights = region * heights
            col_h = np.max( masked_heights, axis = 0 )
            local_maxima   = find_rel_maxima( col_h )
            if( len(local_maxima) == 0 ):
                local_maxima = np.asarray([np.max(masked_heights)])
            max_hs[region] = np.sum(local_maxima)        
    return max_hs

def compute_component_max_height( cca,mask ):
    labels  = np.unique( cca )
    max_hs  = np.zeros( cca.shape )  
    bg_lbl  = get_label_of_background_component( cca ,mask)
    heights = compute_heights( cca ,mask)
    for l in labels:
        if( l != bg_lbl ):
            region = cca == l
            max_hs[region] = np.max( region * heights )
    return max_hs

    
def filter_drusen_by_size( dmask, slice_num=-1 ):
    drusen_mask = np.copy( dmask )
    return drusen_mask
        
    cca, num_drusen = sc.ndimage.measurements.label( drusen_mask )
    # Find the size of each component    
    filtered_mask = np.ones( drusen_mask.shape )
    h  = compute_heights( cca,drusen_mask )
    filtered_mask[np.where( h <=  h_threshold )] = 0.0
    h  = compute_component_max_height( cca,drusen_mask )
    filtered_mask[np.where( h <=  max_h_t )] = 0.0
    
    cca, num_drusen = sc.ndimage.measurements.label( filtered_mask )
    
    # Find the ratio of height over component width and  maximum hight of each component
    w_o_h, height  = compute_width_height_ratio_height_local_max( cca,filtered_mask )
   
    filtered_mask = np.ones( drusen_mask.shape ).astype('float')
    filtered_mask[np.where(w_o_h  >  w_over_h_ratio_threshold)] = 0.0
    filtered_mask[np.where(w_o_h == 0.0)] = 0.0
   
    return filtered_mask
    
def remove_drusen_with_1slice_size( projection_mask ):
    mask = np.copy( projection_mask )
    cca, numL = sc.ndimage.measurements.label( mask )
    bgL = get_label_of_background_component( cca, mask )
    for l in np.unique( cca ):
        if( l != bgL ):
            
            y, x = np.where( cca == l )
            if(len(np.unique(y)) == 1 ):
                mask[y,x] = 0.0
    return mask
    
def read_b_scans( path , img_type = "None",toID=False,returnIds=False):
    d2 = [f for f in listdir(path) if isfile(join(path, f))]
    rawstack = list()
    ind = list()
    rawStackDict=dict()
    rawSize=()
    for fi in range(len(d2)):
         filename = path+os.sep+d2[fi]
         ftype = d2[fi].split('-')[-1]
         if(ftype==img_type or img_type=="None"):
             ind.append(int(d2[fi].split('-')[0]))
             
             raw = io.imread(filename)
             if(len(raw.shape)==3):
                 raw=raw.astype('float')
                 raw=(0.2989*raw[:,:,0])+(0.5870*raw[:,:,1])+(0.1140*raw[:,:,2])
                 if(np.mean(raw)>170):
                     raw=255.-raw
             if(toID):
                 raw=convert_label_uint_to_id(raw)
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
    
def get_label_of_background_component( labels,mask ):
    bgInd=labels[np.where(mask==0)]
    if(len(bgInd)>0):
        return np.unique(bgInd)[0]
    size = np.bincount(labels.ravel())
    largest_comp_ind = size.argmax() 
    return largest_comp_ind

def compute_component_width(cca,mask ):
        labels  = np.unique( cca )
        max_ws  = np.zeros( cca.shape )  
        bg_lbl  = get_label_of_background_component( cca ,mask)
        for l in labels:
            if( l != bg_lbl ):
                y, x = np.where( cca == l )
                w = np.max(x) - np.min(x) +1
                max_ws[cca == l] = w
        return max_ws
        
def compute_width_height_ratio_height_local_max(cca,mask ):
        mx_h = compute_heights( cca,mask )
        mx_w = compute_component_width( cca,mask )
        forDiv=np.copy(mx_h)
        forDiv[mx_h==0]=1
        return mx_w.astype('float')/(forDiv.astype('float')), mx_h
  
def produce_drusen_projection_image_chen( b_scans ):
    height,width,depth=b_scans.shape
    b_scans = b_scans.astype('float')
    masks=np.zeros(b_scans.shape)
    baseLines=np.zeros(b_scans.shape)
    projection = np.zeros((b_scans.shape[2], b_scans.shape[1]))
    projection2 = np.zeros((b_scans.shape[2], b_scans.shape[1]))
    total_y_max = 0
    max_i = 0
    img_max = np.zeros(b_scans[:,:,0].shape)
    for i in range(b_scans.shape[2]):
        b_scan = b_scans[:,:,i]
        b_scan = (b_scan - np.min(b_scan))/(np.max(b_scan)-np.min(b_scan)) if len(np.unique(b_scan))>1 else np.ones(b_scan.shape)
        rpe  = rpe_scan_list[i]
        nrpe = nrpe_scan_list[i]
        mask = draw_lines_on_mask( rpe, nrpe, b_scan.shape )
        area_mask = find_area_btw_RPE_normal_RPE( mask )
        baseLines[:,:,i]=(mask>=2).astype(int)
        area_mask=filter_drusen_by_size(area_mask)
        masks[:,:,i]=np.copy(area_mask)
        
        y_diff = np.sum(area_mask, axis=0)
        y_max    = np.max(y_diff)
        if( total_y_max < y_max ):
            rpe = rpe.astype('int')
            nrpe = nrpe.astype('int')
            img_max = np.copy(b_scan)
            img_max[nrpe[:,1],nrpe[:,0]] = 0.5
            img_max[rpe[:,1],rpe[:,0]] = 1.0
            kk = nrpe[:,1]-y_max
            img_max[kk.astype('int'),nrpe[:,0]] = 1.0
            total_y_max = y_max
    for i in range(b_scans.shape[2]):
        b_scan = b_scans[:,:,i]
        b_scan = (b_scan - np.min(b_scan))/(np.max(b_scan)-np.min(b_scan)) if len(np.unique(b_scan))>1 else np.ones(b_scan.shape)
        n_bscan  = np.copy(b_scan)
        rpe = rpe_scan_list[i]
        nrpe = nrpe_scan_list[i]

        rpe = fill_inner_gaps2(rpe)
        rpe = rpe.astype('int')
        nrpe = nrpe.astype('int')
        nrpe[np.where(nrpe[:,1]<0),1]=0
        nrpe[np.where(nrpe[:,1]>=height),1]=height-1
        drpe  = dict(rpe)
        dnrpe = dict(nrpe)
        upper_y  = np.copy(nrpe)        
        y_max    = total_y_max
        upper_y[:,1]  = (upper_y[:,1] - y_max)
        upper_y[np.where(upper_y[:,1]<0),1]=0
        durpe = dict(upper_y)
        for ix in range(b_scan.shape[1]):
            if( (ix in drpe.keys()) and (ix in dnrpe.keys())):
                n_bscan[drpe[ix]:dnrpe[ix]+1,ix] = np.max(b_scan[durpe[ix]:dnrpe[ix]+1,ix])
                projection[i,ix] =  np.sum(n_bscan[durpe[ix]:dnrpe[ix]+1,ix])            
                projection2[i,ix] =  np.sum(n_bscan[durpe[ix]:drpe[ix]+1,ix])*0.5 + np.sum(n_bscan[drpe[ix]:dnrpe[ix]+1,ix])
        n_bscan[upper_y[:,1].astype('int'),nrpe[:,0].astype('int')] = 1
        n_bscan[rpe[:,1].astype('int'),rpe[:,0]] = 1
        n_bscan[nrpe[:,1].astype('int'),nrpe[:,0].astype('int')] = 0.5
    return projection.astype('float'), masks,baseLines
    
def get_outer_boundary( mask, size ):
    a = sc.ndimage.binary_dilation(mask, iterations = size)
    return a.astype('float') - mask.astype('float')
    
def remove_non_bright_spots( projection_img, drusen_mask, threshold = 4 ):   
    projection_img = (projection_img.astype('float')-np.min(projection_img))/\
                     (np.max(projection_img)-np.min(projection_img))* 255.0
    cca, num_lbls = sc.ndimage.measurements.label( drusen_mask )
    bg_lbl = get_label_of_background_component( cca,drusen_mask )
    labels = np.unique( cca )
    res_m  = np.zeros( drusen_mask.shape, dtype='float' )
    
    for l in labels:
        if( l != bg_lbl ):
            reg_mask = cca == l
            vreg = projection_img[reg_mask>0.0]
            vreg = vreg[vreg>0.0]
            boundary = get_outer_boundary( reg_mask , size = 2 )
            vbnd = projection_img[boundary>0.0]
            vbnd = vbnd[vbnd>0.0]
            reg_mean = np.mean( vreg )
            bnd_mean = np.mean( vbnd )
            # If the mean intensity of the spot is larger by a threshold than the 
            # region then keep this spot, otherwise remove it
            if( (reg_mean - bnd_mean) > threshold ):
                res_m[cca==l] = 1.0
    return res_m

#==============================================================================
# Modified Chen Method
#==============================================================================
def seg_modified_chen( inputImage, useMAFOD, debug=False ):
  if(useMAFOD):
      filter1=mf.MAFOD_filter(inputImage,octParams['sizeInY']/float(inputImage.shape[0]),\
                              octParams['sizeInX']/float(inputImage.shape[1]),\
                              int(octParams['stoppingTime']),int(octParams['numberOfCycles']),\
                              octParams['lambda'])
  else:
      filter1=FF.FilterBilateral(inputImage)
  
  threshold=FF.computeHistogrammThreshold(filter1,octParams['sizeInY'])
  RNFL=FF.getTopRNFL(filter1,threshold,False)

  if(debug):
      #show top RNFL on input
      tmp=np.copy(filter1)
      tmp=tmp.astype(float)
      tmp=tmp/np.max(tmp)
      rgbTmp=np.empty((filter1.shape[0],filter1.shape[1],3))
      rgbTmp[:,:,0]=tmp
      rgbTmp[:,:,1]=tmp
      rgbTmp[:,:,2]=tmp
      
      rgbTmp[RNFL[:,1],RNFL[:,0],0]=1.
      rgbTmp[RNFL[:,1],RNFL[:,0],1]=0.
      rgbTmp[RNFL[:,1],RNFL[:,0],2]=0.
      
      rgbTmp[RNFL[:,1]+1,RNFL[:,0],0]=1.
      rgbTmp[RNFL[:,1]+1,RNFL[:,0],1]=0.
      rgbTmp[RNFL[:,1]+1,RNFL[:,0],2]=0.
      
      rgbTmp[RNFL[:,1]-1,RNFL[:,0],0]=1.
      rgbTmp[RNFL[:,1]-1,RNFL[:,0],1]=0.
      rgbTmp[RNFL[:,1]-1,RNFL[:,0],2]=0.
      
  if( len(RNFL) ==0):
        return [],[]
  
  
  mask=FF.thresholdImage(filter1,threshold)
  bandRadius=int(0.04/float(octParams['sizeInY']/float(inputImage.shape[0]))) # Must be 40um
  mask2=FF.removeTopRNFL(filter1,mask,RNFL,bandRadius=bandRadius)
  FF.extractRegionOfInteresst(filter1,mask2,bw=bandRadius);
  centerLine1=FF.getCenterLine(mask2)
  
  centerLine2,mask3=FF.segmentLines_new(filter1,centerLine1,debug)
  if(False):
      tmp=np.copy(mask3)
      tmp=tmp.astype(float)
      tmp=tmp/np.max(tmp)
      cl=centerLine2.astype(int)
      cl=connect_all(cl)
      rgbTmp=np.empty((filter1.shape[0],filter1.shape[1],3))
      rgbTmp[:,:,0]=tmp
      rgbTmp[:,:,1]=tmp
      rgbTmp[:,:,2]=tmp
      rgbTmp[cl[:,1],cl[:,0],0]=1.
      rgbTmp[cl[:,1],cl[:,0],1]=0.
      rgbTmp[cl[:,1],cl[:,0],2]=0.
      
      rgbTmp[cl[:,1]+1,cl[:,0],0]=1.
      rgbTmp[cl[:,1]+1,cl[:,0],1]=0.
      rgbTmp[cl[:,1]+1,cl[:,0],2]=0.
      
      rgbTmp[cl[:,1]-1,cl[:,0],0]=1.
      rgbTmp[cl[:,1]-1,cl[:,0],1]=0.
      rgbTmp[cl[:,1]-1,cl[:,0],2]=0.
      
#      show_images([rgbTmp],1,1)
      
  itterations=5;
  dursenDiff=5;
  idealRPE=FF.normal_RPE_estimation(filter1,centerLine2,itterations,dursenDiff)

  if(True):
      tmp=np.copy(filter1)
      tmp=tmp.astype(float)
      tmp=tmp/np.max(tmp)
      cl=centerLine2.astype(int)
      cl=fill_inner_gaps2(cl)
      cl=connect_all(cl)
      rgbTmp=np.empty((filter1.shape[0],filter1.shape[1],3))
      rgbTmp[:,:,0]=tmp
      rgbTmp[:,:,1]=tmp
      rgbTmp[:,:,2]=tmp
      rgbTmp[cl[:,1],cl[:,0],0]=1.
      rgbTmp[cl[:,1],cl[:,0],1]=0.
      rgbTmp[cl[:,1],cl[:,0],2]=0.
      
      rgbTmp[cl[:,1]+1,cl[:,0],0]=1.
      rgbTmp[cl[:,1]+1,cl[:,0],1]=0.
      rgbTmp[cl[:,1]+1,cl[:,0],2]=0.
      
      rgbTmp[cl[:,1]-1,cl[:,0],0]=1.
      rgbTmp[cl[:,1]-1,cl[:,0],1]=0.
      rgbTmp[cl[:,1]-1,cl[:,0],2]=0.
      
      cl2=idealRPE.astype(int)
      cl2=fill_inner_gaps2(cl2)
      cl2=connect_all(cl2)
      rgbTmp[cl2[:,1],cl2[:,0],0]=1.
      rgbTmp[cl2[:,1],cl2[:,0],1]=1.
      rgbTmp[cl2[:,1],cl2[:,0],2]=0.
      
      rgbTmp[cl2[:,1]+1,cl2[:,0],0]=1.
      rgbTmp[cl2[:,1]+1,cl2[:,0],1]=1.
      rgbTmp[cl2[:,1]+1,cl2[:,0],2]=0.
      
      rgbTmp[cl2[:,1]-1,cl2[:,0],0]=1.
      rgbTmp[cl2[:,1]-1,cl2[:,0],1]=1.
      rgbTmp[cl2[:,1]-1,cl2[:,0],2]=0.
      
      rgbTmp=np.empty((filter1.shape[0],filter1.shape[1],3))
      rgbTmp[:,:,0]=tmp
      rgbTmp[:,:,1]=tmp
      rgbTmp[:,:,2]=tmp
      
      rgbTmp[cl[:,1],cl[:,0],0]=1.
      rgbTmp[cl[:,1],cl[:,0],1]=0.
      rgbTmp[cl[:,1],cl[:,0],2]=0.
      
      rgbTmp[cl2[:,1],cl2[:,0],0]=1.
      rgbTmp[cl2[:,1],cl2[:,0],1]=1.
      rgbTmp[cl2[:,1],cl2[:,0],2]=0.
      
      
      mask = draw_lines_on_mask(cl, cl2, filter1.shape)
      area_mask = find_area_btw_RPE_normal_RPE( mask )
      yy,xx=np.where(area_mask>0)
      rgbTmp[yy,xx,0]=1.
      rgbTmp[yy,xx,1]=rgbTmp[yy,xx,1]*0.8
      rgbTmp[yy,xx,2]=rgbTmp[yy,xx,1]*0.8
      
  
  return centerLine2, idealRPE

def save_masks(masks,refPath,savePath,img_type="None"):
    d2 = [f for f in listdir(refPath) if isfile(join(refPath, f))]
    ind = list()
    for fi in range(len(d2)):
         ftype = d2[fi].split('-')[-1]
         if(ftype==img_type or img_type=="None"):
             ind.append(int(d2[fi].split('-')[0]))
    ind = np.asarray(ind)
    sin = ind[np.argsort(ind)]
    if(len(masks.shape)==3):
        for i in range(masks.shape[2]):
            fname=str(sin[i])+'-binmask.png'
#            print "Saving......",i
            lsav=masks[:,:,i]/float(np.max(masks[:,:,i])) if np.max(masks[:,:,i])!=0 else masks[:,:,i]
            lsav=(lsav*255).astype("uint8")
            cv2.imwrite(savePath+os.sep+fname,lsav)
    else:
        for i in range(masks.shape[3]):
            fname=str(sin[i])+'-binmask.png'
            lsav=masks[:,:,:,i]/float(np.max(masks[:,:,:,i])) if np.max(masks[:,:,:,i])!=0 else masks[:,:,:,i]
            lsav=(lsav*255).astype("uint8")
            cv2.imwrite(savePath+os.sep+fname,lsav)
        

def get_label_from_projection_image_chen(projected_labels, labels):
    lbls = np.copy(labels)
    
    for i in range( labels.shape[2] ):
        valid_drusens = np.tile(projected_labels[i, :], labels.shape[0]).\
                        reshape(labels.shape[0],labels.shape[1])
                        
        l_area = lbls[:,:,i] 
        lbls[:,:,i] = l_area * valid_drusens
    return lbls

def save_b_scans(bscans,savePath):
	create_directory(savePath)
	for i in range(bscans.shape[2]):
		io.imsave(savePath+str(i+39)+"-Input.tif",bscans[:,:,i].astype("uint8"))
	exit()	   
def segment_drusen_using_chen_modified_chen_method(path,savePath, method,layerPath="",drusenPath="",enfacePath=""):
    print("Eliminating false positives...")
    bs = read_b_scans( path, "Input.tif")
    
    postProcessedMasks=run_chen_modified_chen_method(bs,method=method,\
                        layerPath=layerPath,drusenPath=drusenPath,\
                        enfacePath=enfacePath)
    create_directory(savePath)
    save_masks(postProcessedMasks, refPath=path,savePath=savePath,img_type="Input.tif")              
    
    enfaceDrusenMask=(np.sum(postProcessedMasks,axis=0)>0).astype(int).T
    io.imsave(enfacePath+os.sep+"enface-drusen-afterFPE.png",(enfaceDrusenMask*255).astype("uint8"))
    print("Done.")
    
def remove_false_positives_from_gt(drusenPath,savePath,enfacePath=""):
    gt=read_b_scans(drusenPath,"drusen.png")
    
    masks=(gt>0).astype(int)
    projection=read_enface_from_image(enfacePath)
    projection=projection.astype(float)
    projection /= np.max(projection) if np.max(projection) != 0.0 else 1.0
    beforeMask=(np.sum(masks,axis=0)>0).astype(int)
    enFaceMask=remove_false_positives_gt( projection, gt)
    filteredMasks=get_label_from_projection_image_chen(enFaceMask, masks)
    # Add smoothing
    postProcessedMasks=smooth_drusen_gt(filteredMasks,sigma=octParams['smoothingSigma'])
    afterMask=(np.sum(filteredMasks,axis=0)>0).astype(int).T
    create_directory(savePath)
    save_masks(postProcessedMasks, refPath=drusenPath,savePath=savePath,img_type="drusen.png")
            
def draw_lines_on_mask(rpe , nrpe, shape ):
    mask = np.zeros(shape)
    if(rpe is None):
        rpe=[]
    if(nrpe is None):
        nrpe=[]
    if(len(rpe)==0):
        return mask
    
    rpearr = np.asarray(rpe)
    nrpearr = np.asarray(nrpe)
    
    checkRpe = np.abs(list(rpearr))
    checkNrpe = np.abs(list(nrpearr))
    if(  np.array_equal(rpearr,checkRpe) and np.array_equal(nrpearr,checkNrpe)):
        mask[rpe[:,1].astype('int'),rpe[:,0].astype('int')] += 1.0
        mask[nrpe[:,1].astype('int'),nrpe[:,0].astype('int')] += 2.0
    return mask
    
def find_area_btw_RPE_normal_RPE( mask ):
    area_mask = np.zeros(mask.shape)
    for i in range( mask.shape[1] ):
        col = mask[:,i]
        v1  = np.where(col==1.0)
        v2  = np.where(col==2.0)
        v3  = np.where(col==3.0)
        v1 = np.min(v1[0]) if len(v1[0]) > 0  else -1
        v2 = np.max(v2[0]) if len(v2[0]) > 0  else -1
        v3 = np.min(v3[0]) if len(v3[0]) > 0  else -1
        if( v1 >= 0 and v2 >= 0 ):
            area_mask[v1:v2,i] = 1
    return area_mask
        
def seg_chen( inputImage, useMAFOD,debug=False):
    if(useMAFOD):
        filter1=mf.MAFOD_filter(inputImage,octParams['sizeInY']/float(inputImage.shape[0]),\
                              octParams['sizeInX']/float(inputImage.shape[1]),\
                              int(octParams['stoppingTime']),int(octParams['numberOfCycles']),\
                              octParams['lambda'])
    else:
        filter1=FF.FilterBilateral(inputImage)
    threshold=FF.computeHistogrammThreshold(filter1,octParams['sizeInY'])
    RNFL=FF.getTopRNFL(filter1,threshold,False)
    if(debug):
      #show top RNFL on input
      tmp=np.copy(filter1)
      tmp=tmp.astype(float)
      tmp=tmp/np.max(tmp)
      rgbTmp=np.empty((filter1.shape[0],filter1.shape[1],3))
      rgbTmp[:,:,0]=tmp
      rgbTmp[:,:,1]=tmp
      rgbTmp[:,:,2]=tmp
      
      rgbTmp[RNFL[:,1],RNFL[:,0],0]=1.
      rgbTmp[RNFL[:,1],RNFL[:,0],1]=0.
      rgbTmp[RNFL[:,1],RNFL[:,0],2]=0.
      
      rgbTmp[RNFL[:,1]+1,RNFL[:,0],0]=1.
      rgbTmp[RNFL[:,1]+1,RNFL[:,0],1]=0.
      rgbTmp[RNFL[:,1]+1,RNFL[:,0],2]=0.
      
      rgbTmp[RNFL[:,1]-1,RNFL[:,0],0]=1.
      rgbTmp[RNFL[:,1]-1,RNFL[:,0],1]=0.
      rgbTmp[RNFL[:,1]-1,RNFL[:,0],2]=0.
      
#      show_images([rgbTmp],1,1)
      
    mask=FF.thresholdImage(filter1,threshold)
    if( len(RNFL) ==0):
        return [],[]
        
    mask[RNFL[:,1],RNFL[:,0]]=255
    bandRadius=int(0.04/float(octParams['sizeInY']/float(inputImage.shape[0]))) # Must be 40um
    mask2=FF.removeTopRNFL(filter1,mask,RNFL,bandRadius)
    
    centerLine1=FF.getCenterLine(mask2)
    
    if(False):
      tmp=np.copy(mask2)
      tmp=tmp.astype(float)
      tmp=tmp/np.max(tmp)
      cl=centerLine1.astype(int)
      cl=fill_inner_gaps2(cl)
      rgbTmp=np.empty((filter1.shape[0],filter1.shape[1],3))
      rgbTmp[:,:,0]=tmp
      rgbTmp[:,:,1]=tmp
      rgbTmp[:,:,2]=tmp
      rgbTmp[cl[:,1],cl[:,0],0]=1.
      rgbTmp[cl[:,1],cl[:,0],1]=0.
      rgbTmp[cl[:,1],cl[:,0],2]=0.
      
      rgbTmp[cl[:,1]+1,cl[:,0],0]=1.
      rgbTmp[cl[:,1]+1,cl[:,0],1]=0.
      rgbTmp[cl[:,1]+1,cl[:,0],2]=0.
      
      rgbTmp[cl[:,1]-1,cl[:,0],0]=1.
      rgbTmp[cl[:,1]-1,cl[:,0],1]=0.
      rgbTmp[cl[:,1]-1,cl[:,0],2]=0.
      
#      show_images([rgbTmp],1,1)
      
    itterations=5;
    dursenDiff=5;
    idealRPE=FF.normal_RPE_estimation(filter1,centerLine1,itterations,dursenDiff)
    
    if(True):
      tmp=np.copy(inputImage)
      tmp=tmp.astype(float)
      tmp=tmp/np.max(tmp)
      cl=centerLine1.astype(int)
      cl=fill_inner_gaps2(cl)
      cl=connect_all(cl)
      rgbTmp=np.empty((filter1.shape[0],filter1.shape[1],3))
      rgbTmp[:,:,0]=tmp
      rgbTmp[:,:,1]=tmp
      rgbTmp[:,:,2]=tmp
      rgbTmp[cl[:,1],cl[:,0],0]=1.
      rgbTmp[cl[:,1],cl[:,0],1]=0.
      rgbTmp[cl[:,1],cl[:,0],2]=0.
      
      rgbTmp[cl[:,1]+1,cl[:,0],0]=1.
      rgbTmp[cl[:,1]+1,cl[:,0],1]=0.
      rgbTmp[cl[:,1]+1,cl[:,0],2]=0.
      
      rgbTmp[cl[:,1]-1,cl[:,0],0]=1.
      rgbTmp[cl[:,1]-1,cl[:,0],1]=0.
      rgbTmp[cl[:,1]-1,cl[:,0],2]=0.
      
      cl2=idealRPE.astype(int)
      cl2=fill_inner_gaps2(cl2)
      cl2=connect_all(cl2)
      rgbTmp[cl2[:,1],cl2[:,0],0]=1.
      rgbTmp[cl2[:,1],cl2[:,0],1]=1.
      rgbTmp[cl2[:,1],cl2[:,0],2]=0.
      
      rgbTmp[cl2[:,1]+1,cl2[:,0],0]=1.
      rgbTmp[cl2[:,1]+1,cl2[:,0],1]=1.
      rgbTmp[cl2[:,1]+1,cl2[:,0],2]=0.
      
      rgbTmp[cl2[:,1]-1,cl2[:,0],0]=1.
      rgbTmp[cl2[:,1]-1,cl2[:,0],1]=1.
      rgbTmp[cl2[:,1]-1,cl2[:,0],2]=0.
      
      
#      show_images([rgbTmp],1,1)
      
      rgbTmp=np.empty((filter1.shape[0],filter1.shape[1],3))
      rgbTmp[:,:,0]=tmp
      rgbTmp[:,:,1]=tmp
      rgbTmp[:,:,2]=tmp
      
      rgbTmp[cl[:,1],cl[:,0],0]=1.
      rgbTmp[cl[:,1],cl[:,0],1]=0.
      rgbTmp[cl[:,1],cl[:,0],2]=0.
  
      
      rgbTmp[cl2[:,1],cl2[:,0],0]=1.
      rgbTmp[cl2[:,1],cl2[:,0],1]=1.
      rgbTmp[cl2[:,1],cl2[:,0],2]=0.
      
      mask = draw_lines_on_mask(cl, cl2, filter1.shape)
      area_mask = find_area_btw_RPE_normal_RPE( mask )
      yy,xx=np.where(area_mask>0)
      rgbTmp[yy,xx,0]=1.
      rgbTmp[yy,xx,1]=rgbTmp[yy,xx,1]*0.8
      rgbTmp[yy,xx,2]=rgbTmp[yy,xx,1]*0.8
      
#      show_images([rgbTmp],1,1)
    return centerLine1, idealRPE
    
def fill_inner_gaps( layer ):
    d_layer = dict(layer)
    prev = -1
    if( len(d_layer.keys())>0):
        maxId=int(np.max(np.asarray(list(d_layer.keys()))))
        for i in range(maxId):
            if( not i in d_layer.keys() and prev!=-1):
                d_layer[i] = prev
                
            if( i in d_layer.keys() ):
                prev = d_layer[i]
        return np.asarray([list(d_layer.keys()),list(d_layer.values())]).T
    else:
        return np.asarray([])

def connect_all( layer ):
    d_layer = dict(layer)
    connectedLayer=[]
    if( len(d_layer.keys())>0):
        maxId=int(np.max(np.asarray(list(d_layer.keys()))))
        minId=int(np.min(np.asarray(list(d_layer.keys()))))
        for i in range(minId,maxId+1):
            if( i in d_layer.keys() ):
                currY=d_layer[i]
                connectedLayer.append([i,currY])
                if((i+1) in d_layer.keys()):
                    nextY=d_layer[i+1]
                    for j in range(abs(nextY-currY)):
                        if(nextY>currY):
                            connectedLayer.append([i,currY+j])
                        else:
                            connectedLayer.append([i,currY-j])
                
        return np.asarray(connectedLayer)
    else:
        return np.asarray([])
    
def fill_inner_gaps2( layer ):
    d_layer = dict(layer)
    prev = -1
    minY=int(np.min(np.asarray(list(d_layer.values()))))
    maxY=int(np.max(np.asarray(list(d_layer.values()))))
    if( len(d_layer.keys())>0):
        maxId=int(np.max(np.asarray(list(d_layer.keys()))))
        minId=int(np.min(np.asarray(list(d_layer.keys()))))
        for i in range(minId,maxId+1):
            if( not i in d_layer.keys() ):
                endP=-1
                startingP=prev
                startingI=i-1
                while(endP==-1):
                    i=i+1
                    if( i in d_layer.keys()):
                        endP=d_layer[i]
                        endI=i
                m=(endP-startingP)/(endI-startingI)
                for j in range(startingI,endI+1):
                    d_layer[j] = max(minY,min(maxY,int(m*(j-startingI)+startingP)))
                
            if( not i in d_layer.keys() and prev!=-1):
                d_layer[i] = prev
                
            if( i in d_layer.keys() ):
                prev = d_layer[i]
        return np.asarray([list(d_layer.keys()),list(d_layer.values())]).T
    else:
        return np.asarray([])
        
def find_drusen_in_stacked_slices_chen( b_scans ):
    hmask = np.zeros((b_scans.shape[2], b_scans.shape[1]))
    for i in range(b_scans.shape[2]):
        b_scan = b_scans[:,:,i]
        rpe  = rpe_scan_list[i]
        nrpe = nrpe_scan_list[i]
        mask = draw_lines_on_mask(rpe, nrpe, b_scan.shape)
        area_mask = find_area_btw_RPE_normal_RPE( mask )
        hmask[i,:] = np.sum(area_mask, axis=0)
    return hmask

def remove_false_positives_chen( projection_image, b_scans): 
    height_mask = find_drusen_in_stacked_slices_chen( b_scans )
    height_mask[height_mask<=1]=0.
    mask = (height_mask>0.0).astype('int')
    cca, num_drusen = sc.ndimage.measurements.label( mask )
    bgLbl=get_label_of_background_component(cca,mask)
    woh,heights=compute_width_height_ratio_height_local_max(cca,mask)
    diffBScanLocation=octParams['sizeInZ']/float(b_scans.shape[2]-1)
    isHighRes = diffBScanLocation <  0.1 # Diff less than 100 um
    wohT=octParams['widthOverHeightCoeff']*((octParams['sizeInZ']/float(b_scans.shape[2]))/(octParams['sizeInX']/float(b_scans.shape[1])))
    mask3=None
    if( isHighRes ):
#         Remove drusen that appear in 1 slice
        cca2=np.copy(cca)
        mask2=np.copy(mask)
        cca2[np.where(heights==1)]=bgLbl
        mask2[np.where(heights==1)]=0
        # Remove thin drusen based on w/h
        cca2[np.where(woh>wohT)]=bgLbl
        mask2[np.where(woh>wohT)]=0
        # Remove based on brightness
        mask3 = (mask2>0.0).astype('float')
        mask3 = remove_non_bright_spots(projection_image,mask2,octParams['intensity'])
    else:
        cca2=np.copy(cca)
        mask2=np.copy(mask)
        # Remove thin drusen based on w/h
        cca2[np.where(woh>(wohT))]=bgLbl
        mask2[np.where(woh>(wohT))]=0
        # Remove based on brightness
        mask3 = (mask2>0.0).astype('float')
        mask3 = remove_non_bright_spots(projection_image,mask2,octParams['intensity'])
    return mask3


def remove_false_positives_gt( projection_image, gt): 
    
    height_mask = np.sum(gt,axis=0).T
    mask = (height_mask>0.0).astype('int')
    
    cca, num_drusen = sc.ndimage.measurements.label( mask )
    bgLbl=get_label_of_background_component(cca,mask)
    
    woh,heights=compute_width_height_ratio_height_local_max(cca,mask)
    
    diffBScanLocation=octParams['sizeInZ']/float(gt.shape[2]-1)
    isHighRes = diffBScanLocation <  0.1 # Diff less than 100 um
    
    wohT=octParams['widthOverHeightCoeff']*((octParams['sizeInZ']/float(gt.shape[2]))/(octParams['sizeInX']/float(gt.shape[1])))
    
    mask3=None
    if( isHighRes ):
#         Remove drusen that appear in 1 slice
        cca2=np.copy(cca)
        mask2=np.copy(mask)
        cca2[np.where(heights==1)]=bgLbl
        mask2[np.where(heights==1)]=0
        # Remove thin drusen based on w/h
        cca2[np.where(woh>wohT)]=bgLbl
        mask2[np.where(woh>wohT)]=0
        # Remove based on brightness
        mask3 = (mask2>0.0).astype('float')
        mask3 = remove_non_bright_spots(projection_image,mask2,octParams['intensity'])
    else:
        cca2=np.copy(cca)
        mask2=np.copy(mask)
         
        # Remove thin drusen based on w/h
        cca2[np.where(woh>(wohT))]=bgLbl
        mask2[np.where(woh>(wohT))]=0
         
        # Remove based on brightness
        mask3 = (mask2>0.0).astype('float')
        mask3 = remove_non_bright_spots(projection_image,mask2,octParams['intensity'])
    return mask3
    
def delete_rpe_nrpe_lists(  ):
    del rpe_scan_list[:]
    del nrpe_scan_list[:]  
    
def initialize_rpe_nrpe_lists( b_scans,method='chen',useMAFOD=False):
    debug=False
    for i in range(b_scans.shape[2]):
        print("Segmenting Drusen in B-scan:#",i)
        if(method=='chen'):
            rpe, nrpe = seg_chen(b_scans[:,:,i],useMAFOD)
        elif(method=='modifiedChen'):
            rpe, nrpe = seg_modified_chen(b_scans[:,:,i],useMAFOD)
        if(len(rpe)==0):
            xr=np.arange(b_scans.shape[1])
            yr=np.ones(xr.shape)
            rpe=np.zeros((b_scans.shape[1],2))
            rpe[:,0]=xr
            rpe[:,1]=yr
            nrpe=np.copy(rpe)
        nrpe[np.where(nrpe[:,1]>=b_scans.shape[0]),1]=b_scans.shape[0]-1
        rpe_scan_list.append(rpe)
        nrpe_scan_list.append(nrpe)
        if(debug):
            mm = draw_lines_on_mask(rpe , nrpe, shape=b_scans[:,:,0].shape )
#            show_images([b_scans[:,:,i],mm],1,2)

    
def smooth_drusen(masks,baseLines,sigma=0.):
    smoothedDrusen=sc.ndimage.gaussian_filter(masks,sigma)
    smoothedDrusen=(smoothedDrusen>0).astype(int)
    filterMask=np.zeros((masks.shape[0],masks.shape[1]))
            
    for i in range(smoothedDrusen.shape[2]):
        nrpe=nrpe_scan_list[i]
        nrpe = nrpe.astype('int')
        filterMask=baseLines[:,:,i]
        #Flip
        filterMask=np.flipud(filterMask)
        filterMask=(np.cumsum(filterMask,axis=0)>0).astype('int')
        filterMask=np.flipud(filterMask)
        
        smoothedDrusen[:,:,i]=smoothedDrusen[:,:,i]*filterMask
        filterMask.fill(0)
    return smoothedDrusen
 
def smooth_drusen_gt(masks,sigma=0.):
    smoothedDrusen=sc.ndimage.gaussian_filter(masks,sigma)
    smoothedDrusen=(smoothedDrusen>0).astype(int)
    return smoothedDrusen
    
def remove_dangling_components(filteredMasks):
    for i in range(filteredMasks.shape[2]):
        cy,cx=np.where(filteredMasks[:,:,i]>0)
        nextInd=min(i+1,filteredMasks.shape[2]-1)
        prevInd=max(i-1,0)
        #currentCount
        c=np.sum((filteredMasks[:,:,i]>0).astype(int))
        if(c==0): # Drusen free B-scan
            continue
        sumNext=np.sum((filteredMasks[cy,cx,nextInd]>0).astype(int))
        sumPrev=np.sum((filteredMasks[cy,cx,prevInd]>0).astype(int))
        if(sumNext==0 and sumPrev==0): # If the current drusen are dangling, remove them
            filteredMasks[:,:,i].fill(0.)
    return filteredMasks
    
def run_chen_modified_chen_method(b_scans,method,savePath="",layerPath="",\
                                            drusenPath="",enfacePath=""):
    if(layerPath!=""):
        baseLines=read_rpe_nrpe_from_image(layerPath)
        masks=read_drusen_from_image(drusenPath)
        projection=read_enface_from_image(enfacePath)
    else:
        initialize_rpe_nrpe_lists(b_scans,method)
        projection,masks,baseLines = produce_drusen_projection_image_chen( b_scans )
        projection /= np.max(projection) if np.max(projection) != 0.0 else 1.0
        
    enFaceMask=remove_false_positives_chen( projection, b_scans)
    filteredMasks=get_label_from_projection_image_chen(enFaceMask, masks)
    # Add smoothing
    filteredMasks=smooth_drusen(filteredMasks,baseLines,sigma=octParams['smoothingSigma'])
    delete_rpe_nrpe_lists()
    enFaceMask=(np.sum(filteredMasks,axis=0)>0).astype(int).T
    return filteredMasks

def save_rpe_nrpe_drusen(layerPath,druPath,enfacePath,refPath,method,useMAFOD):
    bScans = read_b_scans( refPath, "Input.tif")
    #save_b_scans(bScans,savePath="C:\\Users\\shekoufeh\\Desktop\\Manuscript\\ChenGit\\DrusenSegmentation-ModifiedChen\\volume2\\")
    initialize_rpe_nrpe_lists(bScans,method,useMAFOD)
    hmask = np.zeros((bScans.shape[2], bScans.shape[1]))
    rpeNrpe=np.empty(bScans.shape)
    drusen=np.empty(bScans.shape)
    for i in range(bScans.shape[2]):
        b_scan = bScans[:,:,i]
        rpe  = rpe_scan_list[i]
        nrpe = nrpe_scan_list[i]
        rpe = fill_inner_gaps2(rpe)
        nrpe = fill_inner_gaps2(nrpe)
        mask = draw_lines_on_mask(rpe, nrpe, b_scan.shape)
        rpeNrpe[:,:,i]=mask
        area_mask = find_area_btw_RPE_normal_RPE( mask )
        drusen[:,:,i]=area_mask
        hmask[i,:] = np.sum(area_mask, axis=0)
    print("Saving...")
    create_directory(layerPath)    
    save_masks(rpeNrpe, refPath=refPath,savePath=layerPath,img_type="Input.tif")              
    create_directory(druPath)  
    save_masks(drusen, refPath=refPath,savePath=druPath,img_type="Input.tif")        
    
    # Create projectoion image
    projection,masks,baseLines = produce_drusen_projection_image_chen( bScans )
    projection /= np.max(projection) if np.max(projection) != 0.0 else 1.0
    create_directory(enfacePath)
    
    io.imsave(enfacePath+os.sep+"enface.png",(projection*255).astype("uint8"))
    
    enfaceDrusenMask=(np.sum(masks,axis=0)>0).astype(int).T
    io.imsave(enfacePath+os.sep+"enface-drusen-withoutFPE.png",(enfaceDrusenMask*255).astype("uint8"))
    
    print("Done.")
def read_rpe_nrpe_from_image(readPath):
    layers=read_b_scans(readPath,"binmask.png")
    baseLines=np.empty(layers.shape)
    for i in range(layers.shape[2]):
        layers[:,:,i]=convert_label_uint_to_id(layers[:,:,i])
        rpeImg=(layers[:,:,i]==1).astype(int)
        rpeImg[np.where(layers[:,:,i]==3)]=1
        nrpeImg=(layers[:,:,i]>1).astype(int)
        baseLines[:,:,i]=nrpeImg
        yrpe,xrpe=np.where(rpeImg>0)
        rpe=np.asarray(list(zip(xrpe,yrpe)))

        ynrpe,xnrpe=np.where(nrpeImg>0)
        nrpe=np.asarray(list(zip(xnrpe,ynrpe)))
        if(len(rpe)==0):
            xr=np.arange(layers.shape[1])
            yr=np.ones(xr.shape)
            rpe=np.zeros((layers.shape[1],2))
            rpe[:,0]=xr
            rpe[:,1]=yr
            nrpe=np.copy(rpe)
        rpe_scan_list.append(rpe)
        nrpe_scan_list.append(nrpe)

    return baseLines
    
def read_drusen_from_image(readPath):
    drusen=read_b_scans(readPath,"binmask.png")
    drusen=(drusen).astype(int)
    return drusen

def read_enface_from_image(readPath):
    return io.imread(readPath+os.sep+"enface.png")    
    
#========================================= end of Chen related method functions                   
    
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

def compute_height(layerImg,tId=255):
    if(len(np.unique(layerImg))>3):
        tmp=(layerImg==tId).astype(int)
        if(tId==127):
            tmp=(layerImg==85).astype(int)
        tmp[layerImg==170]=1
    else:
        tmp=(layerImg==tId).astype(int)
    vec=np.abs((np.sum(tmp,axis=0)>0).astype(int)-1)
    tmp[-1,vec==1]=1
    orig=np.empty(vec.shape)
    orig.fill(tmp.shape[0]-1)
    res=np.copy(orig)
    for i in range(orig.shape[0]):
        res[i]=orig[i]-np.min(np.where(tmp[:,i]==1))
    return res

def im2double(img):
    return (img.astype('float64') ) / 255.0

def permute(A, dim):
    return np.transpose( A , dim )
    
def imread(filename):
    return io.imread(filename)
    
def convert_label_uint_to_id(label):
    return label/85 if len(np.unique(label))==4 else label/127

def main_refined_data_set(method, path, savePath, applyFPE, useMAFOD):
    init_meta_info()
    numSubjects=0
    skipProcessedFiles=False
    # Iterate in folders     
    for d1 in os.listdir(path):
        
            numSubjects+=1
            print("Processing...",d1)
                 
            refPath=path+os.sep+d1+os.sep
            
            md = [f for f in listdir(refPath) if isfile(join(refPath, f))]
            if(len(md)==0):
                continue
                
            p1=savePath+os.sep+"metaData"+os.sep+"layer"+os.sep+d1+os.sep
            p2=savePath+os.sep+"withoutFPE"+os.sep+d1+os.sep
            p3=savePath+os.sep+"metaData"+os.sep+"enface"+os.sep+d1+os.sep
            p4=savePath+os.sep+"afterFPE"+os.sep+d1+os.sep
            
            # Skip already processed OCT scans
            if (skipProcessedFiles and (os.path.exists(p4) and\
                len([f for f in listdir(refPath) if isfile(join(refPath, f))])>0)):
                    print("skip ",refPath)
                    continue
            
            # Create segmentations and store them
            save_rpe_nrpe_drusen(p1,p2,p3,refPath,method,useMAFOD)
            
            # FPE STEP
            if(applyFPE):
                segment_drusen_using_chen_modified_chen_method(refPath,p4,method,p1,p2,p3)

def main(args):
    
    main_refined_data_set(args.method, args.source, args.dest, args.fpe,args.mafod)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drusen segmentation using the algorithm by Chen et al. or its modified version.")
    parser.add_argument(
        "--method", action="store", required=True, help="Method name. Use either of the following keywords: {Chen, modifiedChen}")
    parser.add_argument(
        "--source", action="store", required=True, help="Path to the folder that contains OCT volumes.")
    parser.add_argument(
        "--dest", action="store", required=True, help="Path to save drusen segmentation into.")
    parser.add_argument(
        "--fpe", action="store_true", required=False, default=False, help="Automatically eliminate falsely detected drusen.")
    parser.add_argument(
        "--mafod", action="store_true", required=False, default=False, help="Uses MAFOD filter instead of Bilateral.")
    args = parser.parse_args()
    main(args)
