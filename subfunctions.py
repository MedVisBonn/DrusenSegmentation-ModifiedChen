import numpy as np
from scipy import ndimage
from skimage import exposure
import matplotlib.pyplot as plt
from skimage.filters import rank
from scipy.ndimage.filters import gaussian_filter


def restoreBigLabels(out,input,regsize):
    #labelsn
    #label the image
    lbim, numL= ndimage.label(input)
    hist = np.histogram(lbim,bins=np.arange(numL+5))
    ls=hist[0]
    ln=hist[1]
    ln=ln[:-1]
    ls[0]=0
    l2r=ln[ls>regsize]
    for i in range(0,len(l2r)):
        lab=l2r[i]
        out[lbim==lab]=1


def bumpRemovalreverse(ux,uy,dx,dy,img,debug=0):
    #create working img
    wimg=np.copy(img)
    removeAbouve(wimg,ux,uy,0)
    removeBelow(wimg,dx,dy,0)
    wimg[wimg<0.7]=0
    wimg[wimg>=0.7]=1

    #this pushes it simply up to the current white regs
    dx,dy=nearestSearch(dx,dy,1-wimg,0)

    dx=dx[::-1]
    dy=dy[::-1]
    for i in range(1,len(dx)):
        py=dy[i-1]
        px=dx[i-1]

        cy=dy[i]
        cx=dx[i]
        dist=abs(py-cy)
        if dist>2:
            #check if we could have imporoved that one
            #local nn of px
            uusv = wimg[py-2,cx]
            usv  = wimg[py-1,cx]
            csv  = wimg[py,cx]
            dsv  = wimg[py+1,cx]
            ddsv = wimg[py+2,cx]

            energy=[uusv,usv,csv,dsv,ddsv]
            energy=np.array(energy)
            w=np.where(energy>0.5)[0]
            if len(w)==0:
               continue
            else:
              min=w[-1]
              dy[i]=dy[i-1]+(min-2)
              if (debug==1 and i %1 ==0 ):
                  del ax1.lines[0]
                  ax1.plot(dx,dy,color='r',lw=3)
                  fig.canvas.draw()
                  plt.pause(0.001)
    return dx,dy
    
def bumpRemoval(ux,uy,dx,dy,img,debug=0):
    #create working img
    wimg=np.copy(img)
    removeAbouve(wimg,ux,uy,0)
    removeBelow(wimg,dx,dy,0)
    wimg[wimg<0.7]=0
    wimg[wimg>=0.7]=1

    #this pushes it simply up to the current white regs
    dx,dy=nearestSearch(dx,dy,1-wimg,0)
    for i in range(1,len(dx)):
        py=dy[i-1]
        px=dx[i-1]

        cy=dy[i]
        cx=dx[i]
        dist=abs(py-cy)
        if dist>2:
            #check if we could have imporoved that one
            #local nn of px
            uusv = wimg[py-2,cx]
            usv  = wimg[py-1,cx]
            csv  = wimg[py,cx]
            dsv  = wimg[py+1,cx]
            ddsv = wimg[py+2,cx]
            energy=[uusv,usv,csv,dsv,ddsv]
            energy=np.array(energy)
            w=np.where(energy>0.5)[0]
            if len(w)==0:
              continue
            else:
              min=w[-1]
              dy[i]=dy[i-1]+(min-2)
    return dx,dy

def improveCenterline(cx,cy,img,debug=0):
    ix=np.copy(cx)
    iy=np.copy(cy)
    #label the image
    lbim, numL= ndimage.label(img)
    hist = np.histogram(lbim,bins=np.arange(numL+5))
    ls=hist[0]
    ln=hist[1]
    ln=ln[:-1]
    ls[0]=0
   
    for i in range(0,len(ix)):
        xval=ix[i]
        yval=iy[i]

        #get column;
        col=lbim[:,xval]
        vals=col[col!=0]
        if len(vals)==0:
            continue
        else:
            #search for the interesting labels
            #create unique labels
            uniqVals=np.unique(vals)
            if (len(uniqVals))==1:
                continue
            sizes=[];
            for v in range(0,len(uniqVals)):
                sizes.append(ls[uniqVals[v]])

            #get best label
            maxV=np.max(sizes)
            maxId=np.where(sizes==maxV)[0]
            bestL=uniqVals[maxId]
            #now improve the centerline
            w=np.where(col==bestL)[0]
            nyv=np.sum(w)/len(w)
            iy[i]=nyv

    return ix,iy

def rimpBot(img):
    height,width=img.shape
    img2ret=np.copy(img)
    img2ret[:,:]=0
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    for i in range(0,width):
        col=img[:,i]
        #label it
        lbim, numL= ndimage.label(col)
        hist = np.histogram(lbim,bins=np.arange(numL+5))
        ls=hist[0]
        ln=hist[1]
        ln=ln[:-1]
        #remove black labels
        l2r=lbim[col==0]
        for l in range(0,len(l2r)):
            ls[l2r[l]]=0
        bestL=np.argmax(ls)
        ncol=np.copy(col)
        ncol[:]=0
        ncol[lbim==bestL]=1
        img2ret[:,i]=ncol
    return img2ret

def binClose(mask,struct):
    return ndimage.binary_closing(mask, structure=struct).astype(np.float64)

def dilation(mask,struct):
    return ndimage.binary_dilation(mask, structure=struct).astype(np.float64)
def erosion(mask,struct):
    return ndimage.binary_erosion(mask, structure=struct).astype(np.float64)

def improveBottom(rpeImg,bx,by,offset=5,debug=0):
    # push line 5 pix up
    cx=np.copy(bx)
    cy=np.copy(by)
    cy=cy-offset
    resImg=np.copy(rpeImg)
    for i in range(0,len(cx)):
        x=cx[i]
        y=cy[i]
        line=resImg[y:,x]
        vals=np.where(line==1)[0]
        if len(vals)==0:
            continue
        toremove=vals[0]
        resImg[y+toremove:,x]=0.5
    return resImg

def improveDown(rpeImg,debug=0):
    full= ndimage.generate_binary_structure(2, 2)
    cross=ndimage.generate_binary_structure(2, 1)
    px=[]
    py=[]

    px2=[]
    py2=[]
    idoX,idoY=getLowerLine(rpeImg)

    height,width=rpeImg.shape
    # get first inital point;
    px.append(idoX[0])
    py.append(idoY[0])
    #label the image
    lbim, numL= ndimage.label(rpeImg)
    hist = np.histogram(lbim,bins=np.arange(numL+5))
    ls=hist[0]
    ln=hist[1]
    ln=ln[:-1]
    ls[0]=0

    currentLabel=lbim[idoY[0],idoX[0]]

    for i in range(px[0]+1,width):
      col=lbim[:,i]
      vals=np.where(col==currentLabel)[0]
      if len(vals)==0:
          #no label found use the lowest whitePixel
          guessvals=np.where(col!=0)[0]
          if len(guessvals)==0:
              continue
          else:
            maxval=np.max(guessvals)
            px.append(i)
            py.append(maxval)
            currentLabel=col[maxval]
      else:
          maxval=np.max(vals)
          px.append(i)
          py.append(maxval)
    # now run backwards
    px2.append(px[-1])
    py2.append(py[-1])

    for i in range(px[0]+1,width):
      curX=width-i
      col=lbim[:,curX]
      vals=np.where(col==currentLabel)[0]
      if len(vals)==0:
          #no label found use the lowest whitePixel
          guessvals=np.where(col!=0)[0]
          if len(guessvals)==0:
              continue
          else:
            maxval=np.max(guessvals)
            px2.append(curX)
            py2.append(maxval)
            currentLabel=col[maxval]
      else:
          maxval=np.max(vals)
          px2.append(curX)
          py2.append(maxval)

    s1=keepLabels(rpeImg,px,py)
    s2=keepLabels(rpeImg,px2,py2)
    s2=dilation(s2,cross)
    upX,upY=getUpperLine(s2)
    doX,doY=getLowerLine(s2)
    return s2

def growdown(img,dox,doy,debug=0):
    upx=np.copy(dox)
    upy=np.copy(doy)
    if debug==1:
      fig = plt.gcf()
      fig.clf()
      ax1 = fig.add_subplot(1,1,1)
      ax1.imshow(img,cmap='gray',interpolation='None')
      ax1.plot(upx,upy,color='r')
      plt.pause(0.01)
      mng = plt.get_current_fig_manager()
      mng.resize(*mng.window.maxsize())

    for i in range(0,len(upx)):
        xval=upx[i]
        yval=upy[i]
        column=img[:,xval]
        #// get lowest white pixel
        vals=np.where(column==1)[0]
        if len(vals)==0:
            continue
        white=1;
        while white==1:
            scalar=img[yval,xval]
            if (scalar==1):
              white=1
              yval+=1
              upy[i]=yval
            else :
                white=0
        if (debug==1 and i%10==0):
          del ax1.lines[0]
          ax1.plot(upx,upy,color='r')
          fig.canvas.draw()
          plt.pause(0.001)
    plt.close("all")
    return upx,upy


def improveTop(img,qx,qy,debug=0):
    height,width=img.shape
    upx=np.copy(qx)
    upy=np.copy(qy)
    for i in range(0,len(qx)):
        xval=qx[i]
        yval=qy[i]
        col=img[:,xval]
        lbim, numL= ndimage.label(col)
        hist = np.histogram(lbim,bins=np.arange(numL+5))
        ls=hist[0]
        ln=hist[1]
        ln=ln[:-1]
        ls[0]=0
        diffLabelIds=np.where(ls!=0)[0]
        numDifLabels=len(diffLabelIds)

        if numDifLabels==1:
            currLabel=ln[diffLabelIds[0]]
            where=np.where(lbim==currLabel)[0]
            ny=where[0]
            upy[i]=ny
        elif numDifLabels>1 : # pick the lowest region here
          currLabel=ln[diffLabelIds[-1]]
          where=np.where(lbim==currLabel)[0]
          ny=where[0]
          upy[i]=ny
    return upx,upy

def grow(img,dox,doy,debug=0):
    upx=np.copy(dox)
    upy=np.copy(doy)
    if debug==1:
      fig = plt.gcf()
      fig.clf()
      ax1 = fig.add_subplot(1,1,1)
      ax1.imshow(img,cmap='gray',interpolation='None')
      ax1.plot(upx,upy,color='r')
      plt.pause(0.01)
      mng = plt.get_current_fig_manager()
      mng.resize(*mng.window.maxsize())

    for i in range(0,len(upx)):
        xval=upx[i]
        yval=upy[i]
        column=img[:,xval]
        #// get lowest white pixel
        vals=np.where(column==1)[0]
        if len(vals)==0:
            continue
        first=np.max(vals)
        yval=first
        upy[i]=yval
        white=1;
        while white==1:
            scalar=img[yval,xval]
            if (scalar==1):
              white=1
              yval-=1
              upy[i]=yval
            else :
                white=0
        if (debug==1 and i%25==0):
          del ax1.lines[0]
          ax1.plot(upx,upy,color='r')
          fig.canvas.draw()
          plt.pause(0.001)

    plt.close("all")
    return upx,upy


def moveLastWhite(img,tx,ty):
    rx=np.copy(tx)
    ry=np.copy(ty)

    for x in range(0,len(tx)):
      xval=tx[x]
      yval=ty[x]
      line=img[:,xval]
      vals=np.where(line==1)[0]
      if len(vals)>0:
          newpos=np.max(vals)
          ry[x]=newpos

    return rx,ry

def magicRemoval(img):
    lbim, numL= ndimage.label(img)
    hist = np.histogram(lbim,bins=np.arange(numL+5))
    ls=hist[0]
    ln=hist[1]
    ln=ln[:-1]
    ls[0]=0
    height,width=img.shape
    l2k=[]
    for x in range(0,width):
        col=lbim[:,x]
        vals=col[col!=0]
        vals=np.unique(vals)
        if len(vals)==0:
            continue
        if len(vals)>1:
          #find the biggest label
          bestSize=-1
          bestLabel=0
          for i in range(0,len(vals)):
              label=vals[i]
              if(ls[label]>bestSize):
                  bestSize=ls[label]
                  bestLabel=label
          l2k.append(bestLabel)
          l2k=list(np.unique(l2k))
        else:
            l2k.append(vals[0])
            l2k=list(np.unique(l2k))
    l2r = range(len(ln))
    for x in range(0,len(l2k)):
      l2r.remove(l2k[x])

    for x in range(0,len(l2r)):
        label=l2r[x]
        lbim[lbim==label]=0
    resImg=np.copy(lbim)
    resImg[:,:]=1
    resImg[lbim==0]=0
    return resImg

def removeAbouve(img,x,y,val=0,offset=2):
    heigth,width=img.shape
    fupX=range(0,width)
    fupY=np.interp(fupX,x,y).astype(np.int32)
    for i in range(0,len(fupY)):
      xx =int(fupX[i])
      column=img[:,xx]
      upTo=fupY[i]-offset
      column[:upTo]=val

def removeBelow(img,x,y,val=0,offset=2):
    heigth,width=img.shape
    fupX=range(0,width)
    fupY=np.interp(fupX,x,y).astype(np.int32)
    for i in range(0,len(fupY)):
      xx =int(fupX[i])
      column=img[:,xx]
      upTo=fupY[i]+offset
      column[upTo:]=val

def rageProve(img,x,y):
    height,width=img.shape
    for xx in range(0,len(x)):
       col=img[:,x[xx]]

       vals=np.where(col==1)[0]
       if len(vals)==0:
           continue

       subcol=img[:y[xx],x[xx]]
       vals=np.where(subcol==1)[0]
       if len(vals)==0:
           continue

       isBlack=1
       while isBlack==1:
           if col[y[xx]]!=1:
               #moveup
               y[xx]=y[xx]-1
           else:
               isBlack=0
    fupX=range(0,width)
    fupY=np.interp(fupX,x,y)
    #clear the image
    for q in range(0,len(fupY)):
      xx =int(fupX[q])
      column=img[:,xx]
      upTo=fupY[q]-1
      column[:upTo]=0.5

def repairTop(img,cx,cy,minSize):
    lbim, numL= ndimage.label(img)
    hist = np.histogram(lbim,bins=np.arange(numL+5))
    ls=hist[0]
    ln=hist[1]
    ln=ln[:-1]
    ls[0]=0
    height,width=img.shape
    l2r=[]
    resimg=np.copy(img)
    return resimg
    for i in range(0,len(cx)):
        yval=cy[i]
        xval=cx[i]
        col=lbim[:,xval]
        coords=np.where(col!=0)[0]
        values=col[col!=0]
        vals=np.unique(values)
        if len(vals)>1:
            needStop=0
            for q in range(0,len(vals)):
                if ls[vals[q]]<minSize:
                    needStop=1
            if needStop==1:
                continue
            #remove that labelel abouve the centerline
            l2rinCol=lbim[coords[0],xval]
            resImgCol=resimg[:,xval]
            resImgCol[:yval]=0

    return resimg

def computeCenterline(img):
    height,width=img.shape
    points=[]
    for x in range(0,width):
        col=img[:,x]
        vals=np.where(col==1)[0]
        if len(vals)==0:
            continue
        center=np.sum(vals)/len(vals)
        points.append([x,center])
    arPts=np.array(points)
    return arPts[:,0],arPts[:,1]
    
def nearestSearch(xx,yy,img,debug=0):
    ix=np.copy(xx)
    iy=np.copy(yy)
    for i in range (0,len(ix)):
        currX=int(ix[i])
        col=img[:,currX]
        possiblePos=np.where(col==0)[0]
        if len(possiblePos)==0:
            continue
        currentY=int(iy[i])
        distances=abs(possiblePos-iy[i])
        minId=np.argmin(distances)
        newPos=possiblePos[minId]
        iy[i]=newPos
    return ix,iy

def shortestPathForward(xx,yy,img,debug=0):
    ix=np.copy(xx)
    iy=np.copy(yy)
    iy=iy
    for i in range (0,len(ix)-1):
        currX=int(ix[i-1])
        nextX=int(ix[i])
        nnextX=int(ix[i+1])
        nextCol=img[:,nextX]
        nnextCol=img[:,nnextX]
        possiblePos=np.where(nextCol==0)[0]
        npossiblePos=np.where(nnextCol==0)[0]
        testNext=0
        if len(npossiblePos)!=0:
            testNext=1
        if len(possiblePos)==0:
            continue
        currentY=int(iy[i-1])
        distances=abs(possiblePos-currentY)
        minId=np.argmin(distances)

        ndistances=0
        nminId=0
        testPos=1000000000000
        if (testNext==1):
          ndistances=abs(npossiblePos-currentY)
          nminId=np.argmin(ndistances)
          testPos=npossiblePos[nminId]
        newPos=0;
        if testNext==1:
          if ndistances[nminId]<distances[minId]:
             newPos=npossiblePos[nminId]
          else:
           newPos=possiblePos[minId]
        else:
           newPos=possiblePos[minId]

        iy[i]=newPos
    iy[len(iy)-1]=iy[len(iy)-2]
    return ix,iy

def rageSearch(img):
    height,width = img.shape
    rimg=np.copy(img)
    for x in range(0,width):
        col=rimg[:,x]
        vals=np.where(col==1)[0]
        if (len(vals)==0):
            continue
        zeros=np.where(col==0)[0]
        searchSpace=col[zeros[0]:zeros[-1]]
        if len(np.where(searchSpace==0.5)[0])==0:
          continue

        order=[]
        order.append(searchSpace[0])
        for i in range(1,len(searchSpace)):
            curVal=order[len(order)-1]
            searchVal=searchSpace[i]
            if curVal==searchVal:
                continue
            order.append(searchVal)

def shortestPathBackward(xx,yy,img,debug=0):
    ix=np.copy(xx)
    iy=np.copy(yy)
    iy=iy
    narf=range(1,len(iy))
    narf=narf[::-1]
    for i in narf:
        currX=i
        nextX=i-1
        nextCol=img[:,nextX]
        possiblePos=np.where(nextCol==0)[0]
        if len(possiblePos)==0:
            continue
        currentY=int(iy[currX])
        distances=abs(possiblePos-currentY)
        minId=np.argmin(distances)
        newPos=possiblePos[minId]

        iy[nextX]=newPos
    iy[0]=iy[1]
    return ix,iy


def removeMoreLabels2(img,x,y):
    lbim, numL= ndimage.label(img)
    hist = np.histogram(lbim,bins=np.arange(numL+5))
    ls=hist[0]
    ln=hist[1]
    ln=ln[:-1]
    ls[0]=0
    height,width=img.shape
    l2k=[]
    seenLabels=np.zeros(len(ln))
    for i in range(0,len(x)):
        label=lbim[y[i],x[i]]
        seenLabels[label]+=1

    l2r=np.where(seenLabels==1)[0]
    if len(l2r)==0:
        return img
    else:
        for i in range(0,len(l2r)):
          lbim[lbim==l2r[i]]=0
        retIm=np.copy(img)
        retIm[:,:]=0
        retIm[lbim!=0]=1
        return retIm

def removeMoreLabels(img):
    lbim, numL= ndimage.label(img)
    hist = np.histogram(lbim,bins=np.arange(numL+5))
    ls=hist[0]
    ln=hist[1]
    ln=ln[:-1]
    ls[0]=0
    height,width=img.shape
    l2k=[]
    for x in range(0,width):
        col=lbim[:,x]
        labels=col[col!=0]
        if len(labels)==0:
            continue
        unique=np.unique(labels)
        keeper=0
        size=0
        for y in range(0,len(unique)):
            al=labels[y]
            val=ls[al]
            if (val>size):
              val=size
              keeper=al
        l2k.append(keeper)
        l2k=list(np.unique(l2k))
    l2r = range(len(ln))
    for x in range(0,len(l2k)):
      l2r.remove(l2k[x])
    for x in range(0,len(l2r)):
      label=l2r[x]
      lbim[lbim==label]=0
    resImg=np.copy(lbim)
    resImg[:,:]=1
    resImg[lbim==0]=0
    return resImg

def magicKeepLabels(mask,scalar,size):
    lbim, numL= ndimage.label(mask)
    hist = np.histogram(lbim,bins=np.arange(numL+5))
    ls=hist[0]
    ln=hist[1]
    ln=ln[:-1]

    #for each label look if it has black values
    l2r=[]
    for i in range(0,len(ln)):
        label=ln[i]
        if ls[label]==0:
            continue
        search=scalar[lbim==label]
        vals=np.where(search==0)[0]
        if len(vals)<=size:
            l2r.append(label)
    for i in range(0,len(l2r)):
        lbim[lbim==l2r[i]]=0

    #newMask
    retImg=np.copy(mask)
    retImg[:,:]=1
    retImg[lbim==0]=0
    return retImg

def keepLabels(img,ix,iy):
    lbim, numL= ndimage.label(img)
    hist = np.histogram(lbim,bins=np.arange(numL+5))
    ls=hist[0]
    ln=hist[1]
    ln=ln[:-1]
    l2k=[]
    for x in range(0,len(ix)):
        yval=int(iy[x])
        label=lbim[yval,ix[x]]
        l2k.append(label)
        l2k=list(np.unique(l2k))
    l2r = range(len(ln))
    for x in range(0,len(l2k)):
        l2r.remove(l2k[x])

    for x in range(0,len(l2r)):
        label=l2r[x]
        lbim[lbim==label]=0

    resImg=np.copy(lbim)
    resImg[:,:]=1
    resImg[lbim==0]=0
    return resImg

def getLowerLine(img):
  height,width=img.shape
  points=[]
  for x in range(0,width):
     #extract a column
     column=img[:,x]
     colInv=column[::-1]
     yd=height-1-np.argmax(colInv)
     if yd==height-1:
       continue
     points.append([x,yd])
  asd=np.array(points)
  return asd[:,0],asd[:,1]

def getUpperLine(img,debug=False):
  height,width=img.shape
  if(debug):
      showImg(img)
  points=[]
  for x in range(0,width):
     #extract a column
     column=img[:,x]

     yd=np.argmax(column)
     if yd==0:
       continue
     points.append([x,yd])
  asd=np.array(points)
  return asd[:,0],asd[:,1]


def rescale(img):
  t=np.copy(img)
  p2, p98 = np.percentile(t, (2, 98))
  img_rescale = exposure.rescale_intensity(t, in_range=(p2, p98))
  return img_rescale

def magic(img):
  t=img*img
  p2, p98 = np.percentile(t, (2, 98))
  img_rescale = exposure.rescale_intensity(t, in_range=(p2, p98))
  return img_rescale


def computeHistogrammThreshold(img):
    res=img
    if np.max(img)<100:
      res=img*255;
    h=np.histogram(res,bins=np.arange(256))
    hist=h[0]
    h,w=res.shape
    k=20 # original value is 5
    d=1.8e-3 # 2mm in depth
    tr=20e-6 #20 my meter
    res= d/h
    c=w*(tr/res + k)
    sum=0;
    threshold=0;
    for i in range(1,255):
        s_t1=np.sum(hist[i-1:255])
        s_t2=np.sum(hist[i:255])
        if s_t1 > c and s_t2< c :
          threshold=i
          break
    return threshold

def rescaleImage(img):
  resc=np.copy(img)
  mx=np.max(img)
  mn=np.min(img)
  resc=img/float(mx)
  return resc

def computeInitialYTop(img):
  temp=np.copy(img)
  th=0.5
  temp[temp<th]=0
  temp[temp>=th]=1
  #get top
  removeSmallRegs(temp,50)
  h,w=img.shape
  pointList=[]
  for x in range(0,w):
    #extract a column
    column=temp[:,x]
    colInv=column[::-1]
    y=np.argmax(column)
    if y==0:
      continue
    pointList.append([x,y])
  pts=np.array(pointList)
  xx=pts[:,0]
  yy=pts[:,1]
  rx=np.arange(w)
  f2 = np.interp(rx,xx, yy)
  return rx,f2


def computeExtImage(img):
  #rescale
  p2, p98 = np.percentile(img, (2, 98))
  img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
  #hist eq
  selem=np.ones([20,5],dtype=np.uint8)
  img_eq = rank.equalize(img, selem=selem)
  img_eq=img_eq/255.0

  img_eq_smooth=gaussian_filter(img_eq,sigma=2,order=0)
 
  img_smooth=gaussian_filter(img,sigma=2,order=0)
  gy,gx=np.gradient(img_smooth)
 
  wi=2.0;
  weq=1.0;
  wed=25.0;
  result=wi*img_rescale+weq*img_eq_smooth+wed*gy
  res=gaussian_filter(result,sigma=2.0,order=0)
  res_rescaled=rescaleImage(res)
  return res_rescaled
  
def scalarRange(img):
  min=np.min(img)
  max=np.max(img)
  return min,max

def showImg(img,cmap=None,interpolation='None',text='not given'):
  plt.imshow(img,cmap=cmap,interpolation='None')
  plt.show(True)

def computeExtImage2(img):
  img=plt.imread('images/img1.png')
  img=img[:,:,0]
  showImg(img,text='original')

  p2, p98 = np.percentile(img, (2, 98))
  img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
  showImg(img_rescale,text='rescaled')

  selem=np.ones([50,50],dtype=np.uint8)
  img_eq = rank.equalize(img, selem=selem)
  img_eq=img_eq/255.0
  showImg(img_eq,text='histeq')
  return img_eq

def removeSmallRegs(img,regSize,debug=0):
  lbim, numL= ndimage.label(img)
  hist = np.histogram(lbim,bins=np.arange(numL+10))
  ls=hist[0]
  ln=hist[1]
  ls[0]=0
  ls[ls<regSize]=0
  l2k=ln[ls>0]
  for i in range(0,len(l2k)):
    ll=l2k[i]

    lbim[lbim==ll]=-1
  img[lbim>=0]=0
  
def closing(img):
      watch=np.copy(img)
      struct2 = ndimage.generate_binary_structure(2, 2)
      # ndimage.binary_dilation(watch, structure=struct2)
      watch=ndimage.binary_closing(watch).astype(watch.dtype)
      watch=255*watch
      removeSmallRegs(watch,200)
      return watch
      
def computeInitialYBot(img):
  temp=np.copy(img)
  th=0.5
  temp[temp<th]=0
  temp[temp>=th]=1
  #get top
  removeSmallRegs(temp,50)
  h,w=img.shape
  pointList=[]
  for x in range(0,w):
    #extract a column
    column=temp[:,x]
    colInv=column[::-1]
    y=h-1-np.argmax(colInv)
    if y==h-1:
      continue
    pointList.append([x,y])
  pts=np.array(pointList)
  xx=pts[:,0]
  yy=pts[:,1]

  rx=np.arange(w)
  f2 = np.interp(rx,xx, yy)
  return rx,f2
