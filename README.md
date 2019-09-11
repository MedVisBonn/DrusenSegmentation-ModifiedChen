# Drusen Segmentation

We have reproduced and modified the algorithm proposed by Chen et al. [1], which can be used for segmenting drusen in Optical Coherence Tomography (OCT) images. Python scripts in this repository can be used to run both algorithms on OCT volumes.


Requirements
---------------

The scripts require

* Python 3.7: https://www.python.org/downloads/

* Other python packages such as numpy, scipy, skimage, argparse, os, sys, and matplotlib should be pre-installed 

to run.

Reproduction
---------------

After package installation, you can simply run the code with a demo dataset (which will be uploaded soon). The running command is as below.

```
python --method chen --source <path to OCT volume> --dest <path to destination folder> # for Chen et al. [1] algorithm
python --method modifiedChen --source <path to OCT volume> --dest <path to destination folder> # for the modified Chen [...] algorithm
```
Per B-scan drusen maps will be saved under ```<dest>/withoutFPE```. An en-face projection of drusen segmentation will be saved under ```<dest>/metaData/[scanName]/enface```. In order to automatically eliminate falsely detected drusen, use ```--fpe``` flag in the command line. The results will be saved under ```<dest>/afterFPE```. Example

```
python --method chen --source <path to OCT volume> --dest <path to destination folder> --fpe 
python --method modifiedChen --source <path to OCT volume> --dest <path to destination folder> --fpe 
```

Before running the algorithms on other OCT volumes, edit ```OCT_info.txt``` respectively.

References
----------

[1] Chen, Qiang, et al. "Automated drusen segmentation and quantification in SD-OCT images." Medical image analysis 17.8 (2013): 1058-1072.
  

%Cite
%----------
%You are welcome to cite the corresponding publication

%* Our paper

  
Git Authors
----------

* Shekoufeh Gorgi Zadeh (https://github.com/shekoufeh)
