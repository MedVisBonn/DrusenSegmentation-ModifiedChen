import time
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.filters.rank import mean_bilateral
from skimage.restoration import denoise_bilateral

from subfunctions import * 

def show_image( image, block = True ):
    plt.imshow( image, cmap = plt.get_cmap('gray'))
    plt.show(block)

def biggestRegs(img):
    lbim, numL= ndimage.label(img)
    hist = np.histogram(lbim,bins=np.arange(numL+5))
    ls=hist[0]
    ln=hist[1]
    ln=ln[:-1]
    ## handle the 0 value label
    q=np.where(img==0)
    py=q[0][0]
    px=q[1][0]

    zeroL=lbim[py][px]
    ls[zeroL]=0
    max=np.amax(ls)
    ls[ls<max]=0
    l2k=ln[ls>0]

    for i in range(0,len(l2k)):
      ll=l2k[i]

      lbim[lbim==ll]=-1
    img[lbim>=0]=0

def removeSmallRegs(img,regSize):
    lbim, numL= ndimage.label(img)
    hist = np.histogram(lbim,bins=np.arange(numL+10))
    ls=hist[0]
    ln=hist[1]
    ln=ln[:-1]
    ls[0]=0
    ls[ls<regSize]=0
    l2k=ln[ls>0]
    for i in range(0,len(l2k)):
      ll=l2k[i]
      lbim[lbim==ll]=-1
    img[lbim>=0]=0

def  FilterBilateral(img, stdFilter=False,winh=5, winw=15,S0=20,S1=50):
    
    if stdFilter==False:
      winsize=(winh,winw)
      win=np.ones(winsize)
      win=255*win
      img255=np.copy(img)
      img=img/255.
      result=mean_bilateral(img, win, shift_x=False,shift_y=False,s0=S0,s1=S1)
      img=img255
      return result;
    else:
      prog= 255*denoise_bilateral(img)
      return prog
      
def computeHistogrammThreshold(img,d):
    h=np.histogram(img,bins=np.arange(256))
    hist=h[0]
    h,w=img.shape
    k=20 # original value is 5
    tr=20e-6 #20 my meter
    res= d/h
    c=w*(tr/res + k)
    threshold=0;
    for i in range(1,255):
        s_t1=np.sum(hist[i-1:255])
        s_t2=np.sum(hist[i:255])
        if s_t1 > c and s_t2< c :
          threshold=i
          break
    return threshold
    
def getTopRNFL(img,threshold=150,debug=False):
   #threshold image
    th=threshold*0.3 # heuristik
    temp=np.copy(img)
    temp[temp<=th]=0
    temp[temp>th]=255


    removeSmallRegs(temp,50)
    biggestRegs(temp)
    height,width=temp.shape
    
    pointList=[]
    for x in range(0,width):
      #extract a column
      column=temp[:,x]
      y=np.argmax(column)
      if y==0:
        continue
      pointList.append([x,y])
    pts=np.array(pointList)
    
    xR=pts[:,0]
    yR=pts[:,1]
    offset=1
    x=xR[offset:-offset:]
    y=yR[offset:-offset:]
    z = np.polyfit(x, y, 10)
    newPoints=np.poly1d(z)
    pointList=[]

    for i in range(0,width):
        xval=i
        yval=max(0,min(int(newPoints(xval)),img.shape[0]-1))
        pointList.append([xval,yval])

    ar=np.array(pointList)
    if debug==True:
        plt.imshow(temp,cmap='gray')
        xpos=ar[:,0]
        ypos=ar[:,1]
        plt.plot(xpos,ypos,'g',lw=3)
        plt.show()
    return ar
  
def thresholdImage(img,threshold):
    mask=np.copy(img)
    mask[img<=threshold]=0
    mask[img>threshold]=255
    return mask;
    
def normal_RPE_estimation(img,centerline,itterations,minDrusenDiff):   
    current=np.copy(centerline)
    height,width=img.shape
    if(len(centerline)==0):
        return centerline
    x=current[:,0]
    y=current[:,1]
    
    tmpx = np.copy(x)
    tmpy = np.copy(y)
    origy = np.copy(y)
    origx = np.copy(x)
    finalx = np.copy(tmpx)
    finaly = tmpy
    for i in range(itterations):
        z = np.polyfit(tmpx, tmpy, deg = 3)
        p = np.poly1d(z)
        new_y = p(finalx).astype('int')
        if( True ):
            tmpx = []
            tmpy = []
            for i in range(0,len(origx)):
              diff=new_y[i]-origy[i]
              if diff<minDrusenDiff:
                  tmpx.append(origx[i])
                  tmpy.append(origy[i])
        else:
            tmpy = np.maximum(new_y, tmpy)
        finaly = new_y
    pointList2=[]
    for i in range(0,len(finalx)):
        
        pointList2.append([finalx[i],finaly[i]])
    current=np.array(pointList2)
    return current
        
def getIdealRPELine(img,centerline,itterations,minDrusenDiff):
    
    current=np.copy(centerline)
    height,width=img.shape
    if(len(centerline)==0):
        return centerline
    
    drusenDiff=height
    stepSize=(height-minDrusenDiff)/(itterations-1)
    for kk in range(0,itterations):
      x=current[:,0]
      y=current[:,1]
      offset=1
      x=x[::offset]
      y=y[::offset]
      z = np.polyfit(x, y, 3)
      newPoints=np.poly1d(z)
      
      pointList=[]
      for i in range(0,width):
        xval=i
        yval=int(newPoints(xval))
        pointList.append([xval,yval])
      npPoly3=np.array(pointList)
      
      yPol=npPoly3[:,1]

      xMid=centerline[:,0]
      yMid=centerline[:,1]
      
      # remove far points out of the interpolation
      newCenter=[]
      for i in range(0,len(xMid)):
          diff=yPol[i]-yMid[i]

          if diff<drusenDiff:
              newCenter.append([xMid[i],yMid[i]])

      offset=1
      ar=np.array(newCenter)
      if( len(ar)==0):
          kk=kk-1
          drusenDiff+=1
          continue
      x=ar[:,0]
      y=ar[:,1]
      x=x[::offset]
      y=y[::offset]
      z = np.polyfit(x, y, 3)
      newPoints=np.poly1d(z)
      pointList2=[]

      for i in range(0,width):
        xval=i
        yval=newPoints(xval)
        pointList2.append([xval,yval])
      current=np.array(pointList2)
      drusenDiff=max(minDrusenDiff,drusenDiff-stepSize)
#      print kk,drusenDiff
    return current;
  
def getCenterLine(mask):
    height,width=mask.shape
    pts=[]
    for x in range(0,width):
      column=mask[:,x]
      cinv=column[::-1]
      ymin=np.argmax(column)
      if ymin==0:
        continue
      ymax=height-1-np.argmax(cinv)
      yval=int(ymin+0.5*(ymax-ymin))
      pts.append([x,yval])
    result=np.array(pts)
    return result

def removeOutliers(img,bandRadius):
    temp=np.copy(img)
    height,width=temp.shape
    #find line
    pts=list()
    for x in range(0,width):
      #extract a column
      column=temp[:,x]
      y=np.argmax(column[::-1])
      y=height-1-y
      if y==height-1:
        continue
      pts.append([x,y])
    if(len(pts)==0):
        return temp
    #interpolate
    ar=np.array(pts)
    x=ar[:,0]
    y=ar[:,1]
    offset=1
    z = np.polyfit(x, y, 3)
    newPoints=np.poly1d(z)
    pointList=[]

    for i in range(0,width):
        xval=i
        yval=int(newPoints(xval))
        pointList.append([xval,yval])
    
    curve=np.array(pointList)
    xx=curve[:,0]
    yy=curve[:,1]

    lbim, numL= ndimage.label(temp)
    l2k=[]
    for rr in range(0,len(yy)):
      #get yvalnualThreshold
      yrange=np.arange(max(0,yy[rr]-bandRadius),min(yy[rr]+bandRadius,height),1)
      labels=set(lbim[yrange,rr])
      l2k=list(set(l2k+list(labels)))


    for i in range(0,len(l2k)):
      ll=l2k[i]
      lbim[lbim==ll]=-1
    temp[lbim>=0]=0
    return temp
  
  
def removeTopRNFL(img,mask,line,bandRadius):
    height,width=img.shape
    temp=np.copy(mask)
    removeSmallRegs(temp,50)

    labelsToRemove=[]

    curve=np.copy(line)
    xpos=curve[:,0]
    ypos=curve[:,1]

    debug=0
    lbim, numL= ndimage.label(temp)
    hist = np.histogram(lbim,bins=np.arange(numL+10))
    for x in range(0,len(ypos)):
      #get yvalnualThreshold
      yrange=np.arange(max(0,ypos[x]-bandRadius),min(ypos[x]+bandRadius,height),1)
      labels=set(lbim[yrange,x])
      labelsToRemove=labelsToRemove+list(labels)

    labelsToRemove=list(set(labelsToRemove))
    for x in range(0,len(labelsToRemove)):
        l=labelsToRemove[x]
        lbim[lbim==l]=0

    temp[lbim==0]=0
    if debug==1:
        f, axarr = plt.subplots(1,2, sharex=True)
        axarr[0].imshow(mask)
        axarr[1].imshow(temp)
        plt.show()
    remOut=removeOutliers(temp,bandRadius=bandRadius)
    if debug==1:
        f, axarr = plt.subplots(1,2, sharex=True)
        axarr[0].imshow(remOut,cmap='gray')
        axarr[1].imshow(temp,cmap='gray')
        plt.show(True)
    return remOut


def extractRegionOfInteresst(img,mask,bw):
    #compute top points of the mask
    height,width=mask.shape
    #find line
    top_pts=list()
    bot_pts=list()
   
    for x in range(0,width):
      #extract a column
      column=mask[:,x]
      y_t=np.argmax(column)
      y_b=np.argmax(column[::-1])
      y_b=height-1-y_b
      if y_b!=height-1:
        bot_pts.append([x,y_b])
      if y_t!=0:
          top_pts.append([x,y_t])

    drawTop=np.array(top_pts)
    drawBot=np.array(bot_pts)
    if(len(top_pts)==0 or len(bot_pts)==0):
        return img
    #Fitting
    f_top=np.array(top_pts)
    f_bot=np.array(bot_pts)
    x_t=f_top[:,0]
    y_t=f_top[:,1]
    z_t = np.polyfit(x_t, y_t, 3)

    x_b=f_bot[:,0]
    y_b=f_bot[:,1]
    z_b = np.polyfit(x_b, y_b, 3)

    eval_t=np.poly1d(z_t)
    eval_b=np.poly1d(z_b)

    fitted_t=[]      #compute top points of the mask
    #find line
    top_pts=[]
    bot_pts=[]
    for x in range(0,width):
      #extract a column
      column=mask[:,x]
      y_t=np.argmax(column)
      y_b=np.argmax(column[::-1])
      y_b=height-1-y_b
      if y_b!=height-1:
        bot_pts.append([x,y_b+bw])
      if y_t!=0:
          top_pts.append([x,y_t-bw])
          
    if(len(top_pts)==0 or len(bot_pts)==0):
        return img
    #Fitting
    f_top=np.array(top_pts)
    f_bot=np.array(bot_pts)
    x_t=f_top[:,0]
    y_t=f_top[:,1]
    z_t = np.polyfit(x_t, y_t, 3)

    x_b=f_bot[:,0]
    y_b=f_bot[:,1]
    z_b = np.polyfit(x_b, y_b, 3)

    eval_t=np.poly1d(z_t)
    eval_b=np.poly1d(z_b)

    fitted_t=[]
    fitted_b=[]
    bw=0
    for i in range(0,width):
      xval=i
      y_t=int(eval_t(xval))
      y_b=int(eval_b(xval))
      fitted_t.append([xval,y_t-bw])
      fitted_b.append([xval,y_b+bw])

    foundT=[]
    foundB=[]

    toptop=np.array(top_pts)
    botbot=np.array(bot_pts)

    top_x=toptop[:,0]
    bot_x=botbot[:,0]

    for i in range(0,width):
      xval=i
      rt=top_x[top_x==i]
      yt_est=10000000000
      if len(rt)!=0:
          id=np.where(top_x==i)
          idx=id[0][0]
          yt_est=top_pts[idx][1]

      yt_fit=fitted_t[i][1]
      foundT.append([xval,min(yt_est,yt_fit)])

      rb=bot_x[bot_x==i]
      yb_est=0
      if len(rb)!=0:
        id=np.where(bot_x==i)
        idx=id[0][0]
        yb_est=bot_pts[idx][1]

      yb_fit=fitted_b[i][1]
      foundB.append([xval,max(yb_est,yb_fit)])
   
    ft=np.array(foundT)
    fb=np.array(foundB)
    ftX=ft[:,0]
    ftY=ft[:,1]
    fbX=fb[:,0]
    fbY=fb[:,1]
    plt.imshow(mask,cmap='gray')
    plt.plot(fbX,fbY,color='r',lw=3)
    plt.plot(ftX,ftY,color='g',lw=3)
    plt.close("all")
    return img


def segmentLines_new(inputImage,centerLine,debug=False):
      img=inputImage
      backupImg=np.copy(img)
      center=np.array(centerLine)
      full= ndimage.generate_binary_structure(2, 2)
      cross=ndimage.generate_binary_structure(2, 1)

      #extrapolate the centerline
      height,width=img.shape
      rx=np.arange(width)
      if(len(centerLine)==0):
          return centerLine
      f1 = np.interp(rx,center[:,0], center[:,1])
      
      ### INITIAL RPE AND ELLIPSOID ------------------------------
      selem=np.ones([80,1],dtype=np.uint8)
      img_eq = rank.equalize(img, selem=selem)
      img_eq=img_eq/255.0
      scaledImg=img/255.0
      input=rescale(scaledImg)
      scalarRange(img)

      x=rx
      y=f1
      backX=np.copy(x)
      backY=np.copy(y)

      wimg=1-img_eq
      wimg[scaledImg<0.2]=1
      wimg[wimg>0.1]=1
      wimg[wimg<=0.1]=0
      rpe=1-wimg;
      if(debug):
          show_image(rpe)
      removeAbouve(rpe,backX,backY,0,20)
      if(debug):
          show_image(rpe)
      removeBelow(rpe,backX,backY,0,20)
      if(debug):
          show_image(rpe)
      removeSmallRegs(rpe,3)
      if(debug):
          show_image(rpe)
    
      rpe=dilation(rpe,cross)
      rpe=dilation(rpe,cross)
      cleaned=rpe
      if(np.sum(cleaned.astype(int))==0):
          return centerLine
          
      upX,upY=getUpperLine(cleaned,debug)
      upXq=np.copy(upX)
      upYq=np.copy(upY)
      doX,doY=getLowerLine(cleaned)

### FIRST IMPROVEMENT

      selem=np.ones([10,1],dtype=np.uint8)
      img_eq2 = rank.equalize(img_eq, selem=selem)
      img_eq2=img_eq2/255.0
      wimg2=1-img_eq2
      wimg2[scaledImg<0.2]=1
      wimg2[wimg2<=0.2]=0
      wimg2[wimg2>=0.8]=1

      rimg=np.copy(wimg2)
      rimg[:,:]=0
      rimg[wimg2==0]=1
      removeAbouve(rimg,upX,upY,0.0,0)
      removeBelow(rimg,doX,doY,0.0,0)

      rimg=dilation(rimg,cross)
      rimg=binClose(rimg,cross)
      upX,upY=getUpperLine(rimg)
      doX,doY=getLowerLine(rimg)

      showIm=np.copy(rimg).astype(np.float64)

### GAUSSIAN IMPROVEMENT FOR LOWER LINE
      selem=np.ones([10,1],dtype=np.uint8)
      gausImg=gaussian_filter(img,sigma=2,order=0)
      qimg= rank.equalize(gausImg, selem=selem)
      qimg=qimg/255.0
      qqimg=np.copy(qimg)
      plt.imshow(qimg,cmap="gray",interpolation="None")
      mng = plt.get_current_fig_manager()
      plt.plot(doX,doY,color='g',lw=3)
      plt.plot(upX,upY,color='r',lw=3)

      qimg[qimg>=0.8]=1
      qimg[qimg<0.8]=0

      removeAbouve(qimg,upX,upY,0.0,0)
      removeBelow(qimg,doX,doY,0.0,0)
      qimg=dilation(qimg,cross)
      plt.close("all")
      plt.title('qimg')
      plt.imshow(qimg,cmap="gray",interpolation="None")
      mng = plt.get_current_fig_manager()
      plt.plot(doX,doY,color='g',lw=3)
      plt.plot(upX,upY,color='r',lw=3)
      plt.close("all")

      rimg=improveDown(qimg,0)
      restoreBigLabels(rimg,qimg,200)
      upX,upY=getUpperLine(rimg)
      doX,doY=getLowerLine(rimg)

      restoreBigLabels(rimg,qimg,200)
      doX,doY=getLowerLine(rimg)

      plt.close("all")
      plt.imshow(rimg,cmap="gray",interpolation="None")
      mng = plt.get_current_fig_manager()
      plt.plot(upX,upY,color='g',lw=2)
      plt.plot(doX,doY,color='r',lw=2)
      plt.close("all")


############### IMPROVE TOP LINE
      selem=np.ones([10,1],dtype=np.uint8)
      gausImg=gaussian_filter(img,sigma=2,order=0)
      qimg= rank.equalize(gausImg, selem=selem)
      qimg=qimg/255.0
      qimg[qimg>=0.8]=1
      qimg[qimg<0.8]=0

      removeAbouve(qimg,upX,upY,0.0,0)
      removeBelow(qimg,doX,doY,0.0,0)
      plt.imshow(qimg,cmap='gray')
      mng = plt.get_current_fig_manager()
      plt.plot(doX,doY,color='g',lw=3)
      plt.plot(upX,upY,color='r',lw=3)

      nupX,nupY=improveTop(qimg,upX,upY)
      plt.imshow(qimg,cmap='gray')
      mng = plt.get_current_fig_manager()
      plt.plot(doX,doY,color='g',lw=3)
      plt.plot(nupX,nupY,color='r',lw=3)
      plt.close('all')


################# ACT SCNAKE STUFF $$$$$$$$
      fX=np.arange(0,width)
      fY = np.interp(fX,nupX,nupY)
      fY2 = np.interp(fX,doX,doY)
      CRX=fY+fY2
      MID=0.5*CRX
     
      plt.imshow(img,cmap='gray')
      mng = plt.get_current_fig_manager()
      plt.plot(fX,MID,color='g',lw=3)
      plt.close('all')
      CENTER=np.array([fX,MID]).T
      return CENTER
