# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:47:34 2016

@author: gorgi
"""

import numpy as np
from Image import Image
        
def MAFOD_filter(inputImg,hx,hy,stoppingTime,numberOfCycles,varLambda):
    img = Image()
    noisy=img.normalize(inputImg)
    print("Applying MAFOD filter...")
###############################################################################
########                          Filtering                               #####
###############################################################################
     
#    hy=200./17.
#    hx=200./51.
    
    minL=min(hx,hy)
    
    hx=hx/minL
    hy=hy/minL
    
#    n = 10
#    T = n * 0.04
  
    max_val = np.max(noisy)
    min_val = np.min(noisy)
    noisy2 = np.copy(noisy)
    noisy2 = noisy2 - min_val # Shift the minimum to 0
    if(max_val != 0):
        noisy2 = noisy2/max_val # Normalize to range [0,1]
                  
    # MAFOD filter
    m_mts_orig = \
        img.multiscale_fourth_order_anisotropic_diffusion_filter(\
        noisy, np.copy(noisy), hx, hy,\
        [0.2,0.3,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0],\
        0.0, 0.5, varLambda,\
        T = stoppingTime, M = numberOfCycles, c_vars = [], betta_var = 0.5, theta_var = 0.13,\
        crease_type = 'r', auto_stop = False, EULER_EX_STEP = False)
    
    m_mts_orig=m_mts_orig-np.min(m_mts_orig)
    m_mts_orig=m_mts_orig/(np.max(m_mts_orig).astype(float)) if (np.max(m_mts_orig)>0) else m_mts_orig
#    io.imsave(savePath+'/'+imageName+'.tif',(m_mts_orig*255).astype("uint8"))
    return (m_mts_orig*255).astype("uint8")
    
    
    
  