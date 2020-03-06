# Drusen Segmentation

We have reproduced and modified the algorithm proposed by Chen et al. [1], which can be used for segmenting drusen in Optical Coherence Tomography (OCT) images. Python scripts in this repository can be used to run both algorithms on OCT volumes.


Requirements
---------------

The scripts require

* Python 3.7: https://www.python.org/downloads/

* Other python packages such as opencv, cffi, numpy, scipy, skimage, and matplotlib should be pre-installed 

to run.

Reproduction
---------------

After package installation, you can simply run the code with the demo dataset in ```sample-scan```. The running command is as below.

```
python modified_chen_main.py --method chen --source <path to OCT volumes> --dest <path to destination folder> # for Chen et al. [1] algorithm
python modified_chen_main.py --method modifiedChen --source <path to OCT volumes> --dest <path to destination folder> # for the modified Chen algorithm
```
Per B-scan drusen maps will be saved under ```<dest>/withoutFPE```. An en-face projection of drusen segmentation will be saved under ```<dest>/metaData/[scanName]/enface```. In order to automatically eliminate falsely detected drusen, use ```--fpe``` flag in the command line. The results will be saved under ```<dest>/afterFPE```. Example

```
python modified_chen_main.py --method chen --source <path to OCT volumes> --dest <path to destination folder> --fpe 
python modified_chen_main.py --method modifiedChen --source <path to OCT volumes> --dest <path to destination folder> --fpe 
```
In order to use multi-scale anisotropic fourth-order diffusion (MAFOD) filter proposed by Gorgi Zadeh et al. [2], instead of bilateral filter, type ```--mafod``` in the command line. The FED library (in fedfjlib folder), by Grewenig et al. [3] was used to implement MAFOD filter with the fast explicit diffusion (FED) scheme.

Before running the algorithms on other OCT volumes, edit ```OCT_info.txt``` respectively.

Dataset
------------------

The exemplary OCT volume scan in the ``sample-scan/volume1`` folder consists of 145 B-scans. You can download the complete version of this dataset that provides ground truth for RPE and BM layers per B-scan from this link https://github.com/MedVisBonn/OCT-Annotation-Tool/tree/master/data . The RPE+BM layer segmentation is carefully performed by a medical expert, who manually has corrected an initial segmentation, spending about 10 minutes per B-scan on average, adding up to a total of 26 hours. Also, ``sample-scan/volume1/OCT-Info.xml`` file includes additional information about the OCT volume scan such as B-scan quality (marked by ``<ImageQuality>`` tag), and size of each B-scan pixel in micrometer (marked by ``<ScaleX>``, ``<ScaleY>`` tags).

License
----------
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but without any warranty. See the GNU General Public License under http://www.gnu.org/licenses/ for more details.


References
----------

[1] Chen, Qiang, et al. "Automated drusen segmentation and quantification in SD-OCT images." Medical image analysis 17.8 (2013): 1058-1072.
  
[2] Gorgi Zadeh, Shekoufeh, et al. "Multi-scale Anisotropic Fourth-Order Diffusion Improves Ridge and Valley Localization." Journal of Mathematical Imaging and Vision 59.2 (2017): 257-269.

[3] Grewenig, Sven, et al. "From box filtering to fast explicit diffusion." In Joint Pattern Recognition Symposium, pp. 533-542. Springer, Berlin, Heidelberg, 2010.

Cite
----------
The manuscript corresponding to this repository is currently undergoing peer review.

  
Git Authors
----------

* Shekoufeh Gorgi Zadeh (https://github.com/shekoufeh)
